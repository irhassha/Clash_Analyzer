import streamlit as st
import pandas as pd
import os
from datetime import datetime, timedelta
import json
import io
import warnings
import numpy as np
import plotly.express as px
from itertools import combinations

# --- Libraries for Machine Learning ---
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

# Import necessary libraries
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode

# Ignore non-critical warnings
warnings.filterwarnings("ignore", category=UserWarning)

# --- Page Configuration & Title ---
st.set_page_config(page_title="Yard Cluster Monitoring", layout="wide")
st.title("Yard Cluster Monitoring")

# --- Function to reset data in memory ---
def reset_data():
    """Clears session state and all of Streamlit's internal caches."""
    keys_to_clear = ['processed_df', 'summary_display', 'vessel_area_slots']
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    st.cache_data.clear()
    st.cache_resource.clear()

# --- FORECASTING FUNCTIONS (No changes here) ---
# ... (All forecasting functions are assumed to be here) ...

def render_clash_tab():
    """Function to display the entire content of the Clash Analysis tab."""
    st.sidebar.header("⚙️ Upload Your Files")
    schedule_file = st.sidebar.file_uploader("1. Upload Vessel Schedule", type=['xlsx', 'csv'])
    unit_list_file = st.sidebar.file_uploader("2. Upload Unit List", type=['xlsx', 'csv'])
    min_clash_distance = st.sidebar.number_input("Minimum Safe Distance (slots)", min_value=0, value=5, step=1, help="Clash is detected if the distance between vessel allocations is less than or equal to this value.")
    
    process_button = st.sidebar.button("🚀 Process Data", use_container_width=True, type="primary")
    st.sidebar.button("Reset Data", on_click=reset_data, use_container_width=True, help="Clear all processed data and caches to start fresh.")

    # Initialize session state keys
    for key in ['processed_df', 'summary_display', 'vessel_area_slots']:
        if key not in st.session_state:
            st.session_state[key] = None

    df_vessel_codes = load_vessel_codes_from_repo() # Assuming this function exists
    if process_button:
        if schedule_file and unit_list_file and (df_vessel_codes is not None and not df_vessel_codes.empty):
            with st.spinner('Loading and processing data...'):
                try:
                    # Data loading and processing logic...
                    if schedule_file.name.lower().endswith(('.xls', '.xlsx')): df_schedule = pd.read_excel(schedule_file)
                    else: df_schedule = pd.read_csv(schedule_file)
                    df_schedule.columns = [col.strip().upper() for col in df_schedule.columns]

                    if unit_list_file.name.lower().endswith(('.xls', '.xlsx')): df_unit_list = pd.read_excel(unit_list_file)
                    else: df_unit_list = pd.read_csv(unit_list_file)
                    df_unit_list.columns = [col.strip() for col in df_unit_list.columns]

                    for col in ['ETA', 'ETD', 'CLOSING PHYSIC']:
                        if col in df_schedule.columns:
                            df_schedule[col] = pd.to_datetime(df_schedule[col], dayfirst=True, errors='coerce')
                    df_schedule.dropna(subset=['ETA', 'ETD'], inplace=True)
                    
                    if 'Row/bay (EXE)' not in df_unit_list.columns:
                        st.error("File 'Unit List' must contain a 'Row/bay (EXE)' column for detailed clash detection.")
                        st.stop()
                    df_unit_list['SLOT'] = df_unit_list['Row/bay (EXE)'].astype(str).str.split('-').str[-1]
                    df_unit_list['SLOT'] = pd.to_numeric(df_unit_list['SLOT'], errors='coerce')
                    df_unit_list.dropna(subset=['SLOT'], inplace=True)
                    df_unit_list['SLOT'] = df_unit_list['SLOT'].astype(int)
                    
                    df_schedule_with_code = pd.merge(df_schedule, df_vessel_codes, left_on="VESSEL", right_on="Description", how="left").rename(columns={"Value": "CODE"})
                    merged_df = pd.merge(df_schedule_with_code, df_unit_list, left_on=['CODE', 'VOY_OUT'], right_on=['Carrier Out', 'Voyage Out'], how='inner')
                    if merged_df.empty: st.warning("No matching data found."); st.session_state.processed_df = None; st.stop()
                    
                    original_vessels_list = df_schedule['VESSEL'].unique().tolist()
                    merged_df = merged_df[merged_df['VESSEL'].isin(original_vessels_list)]
                    excluded_areas = [str(i) for i in range(801, 809)]
                    merged_df['Area (EXE)'] = merged_df['Area (EXE)'].astype(str)
                    filtered_data = merged_df[~merged_df['Area (EXE)'].isin(excluded_areas)]
                    if filtered_data.empty: st.warning("No data left after filtering."); st.session_state.processed_df = None; st.stop()
                    
                    vessel_area_slots = filtered_data.groupby(['VESSEL', 'VOY_OUT', 'ETA', 'ETD', 'SERVICE', 'Area (EXE)']).agg(
                        MIN_SLOT=('SLOT', 'min'),
                        MAX_SLOT=('SLOT', 'max'),
                        BOX_COUNT=('SLOT', 'count')
                    ).reset_index()

                    pivot_for_display = vessel_area_slots.pivot_table(
                        index=['VESSEL', 'VOY_OUT', 'ETA', 'ETD', 'SERVICE'],
                        columns='Area (EXE)',
                        values='BOX_COUNT',
                        fill_value=0)
                    
                    pivot_for_display['TOTAL BOX'] = pivot_for_display.sum(axis=1)
                    pivot_for_display['TOTAL CLSTR'] = (pivot_for_display > 0).sum(axis=1)
                    pivot_for_display.reset_index(inplace=True)
                    pivot_for_display['ETA_Display'] = pivot_for_display['ETA'].dt.strftime('%d/%m/%Y %H:%M')
                    
                    st.session_state.processed_df = pivot_for_display.sort_values(by='ETA', ascending=True)
                    st.session_state.vessel_area_slots = vessel_area_slots
                    st.success("Data processed successfully!")

                except Exception as e:
                    st.error(f"An error occurred during processing: {e}")
                    st.session_state.processed_df = None
        else:
            st.warning("Please upload both files.")

    if st.session_state.get('processed_df') is not None:
        display_df = st.session_state.processed_df.copy()
        vessel_area_slots_df = st.session_state.vessel_area_slots.copy()
        
        # ... (Other UI sections like Upcoming Summary and Chart can be placed here) ...

        st.markdown("---")
        st.header("💥 Potential Clash Summary")
        
        clash_details = {}
        active_vessels = vessel_area_slots_df[['VESSEL', 'VOY_OUT', 'ETA', 'ETD']].drop_duplicates()
        summary_exclude_blocks = ['BR9', 'RC9', 'C01', 'C02', 'D01', 'OOG']

        for (idx1, vessel1), (idx2, vessel2) in combinations(active_vessels.iterrows(), 2):
            if (vessel1['ETA'] < vessel2['ETD']) and (vessel2['ETA'] < vessel1['ETD']):
                v1_slots = vessel_area_slots_df[(vessel_area_slots_df['VESSEL'] == vessel1['VESSEL']) & (vessel_area_slots_df['VOY_OUT'] == vessel1['VOY_OUT'])]
                v2_slots = vessel_area_slots_df[(vessel_area_slots_df['VESSEL'] == vessel2['VESSEL']) & (vessel_area_slots_df['VOY_OUT'] == vessel2['VOY_OUT'])]
                common_areas = pd.merge(v1_slots, v2_slots, on='Area (EXE)', suffixes=('_v1', '_v2'))

                for _, row in common_areas.iterrows():
                    area = row['Area (EXE)']
                    if area in summary_exclude_blocks: continue
                    
                    range1 = (row['MIN_SLOT_v1'], row['MAX_SLOT_v1'])
                    range2 = (row['MIN_SLOT_v2'], row['MAX_SLOT_v2'])
                    gap = max(range1[0], range2[0]) - min(range1[1], range2[1]) - 1

                    if gap <= min_clash_distance:
                        clash_date = max(vessel1['ETA'], vessel2['ETA']).normalize()
                        date_key = clash_date.strftime('%d/%m/%Y')
                        if date_key not in clash_details:
                            clash_details[date_key] = []
                        
                        clash_info = {
                            "block": area,
                            "vessel1_name": vessel1['VESSEL'], "vessel1_slots": f"{range1[0]}-{range1[1]}", "vessel1_box": row['BOX_COUNT_v1'],
                            "vessel2_name": vessel2['VESSEL'], "vessel2_slots": f"{range2[0]}-{range2[1]}", "vessel2_box": row['BOX_COUNT_v2'],
                            "gap": gap
                        }
                        # Cek duplikasi
                        is_duplicate = False
                        for existing_clash in clash_details[date_key]:
                            if (existing_clash['block'] == area and existing_clash['vessel1_name'] in [vessel1['VESSEL'], vessel2['VESSEL']] and existing_clash['vessel2_name'] in [vessel1['VESSEL'], vessel2['VESSEL']]):
                                is_duplicate = True
                                break
                        if not is_duplicate:
                            clash_details[date_key].append(clash_info)
        
        # --- PERUBAHAN TAMPILAN CLASH SUMMARY MENJADI st.container ---
        if not clash_details:
            st.info(f"No potential clashes found with a minimum distance of {min_clash_distance} slots.")
        else:
            total_clash_days = len(clash_details)
            st.markdown(f"**🔥 Found {total_clash_days} day(s) with potential clashes.**")
            clash_dates = sorted(clash_details.keys(), key=lambda x: datetime.strptime(x, '%d/%m/%Y'))
            cols = st.columns(len(clash_dates) or 1)
            
            for i, date_key in enumerate(clash_dates):
                with cols[i]:
                    with st.container(border=True): # Membuat kartu dengan bingkai
                        st.markdown(f"**Potential Clash on: {date_key}**")
                        
                        clashes_on_day = clash_details.get(date_key, [])
                        for clash in clashes_on_day:
                            st.divider() # Garis pemisah antar clash
                            st.markdown(f"**Block {clash['block']}** (Gap: `{clash['gap']}` slots)")
                            st.markdown(f"↳ **{clash['vessel1_name']}**: `{clash['vessel1_box']}` boxes (Slots: `{clash['vessel1_slots']}`)")
                            st.markdown(f"↳ **{clash['vessel2_name']}**: `{clash['vessel2_box']}` boxes (Slots: `{clash['vessel2_slots']}`)")
        # --- AKHIR PERUBAHAN TAMPILAN ---

        st.markdown("---")
        st.header("📋 Detailed Analysis Results")
        st.dataframe(display_df) # Placeholder display for now

    else:
        st.info("Welcome! Please upload your files and click 'Process Data' to begin.")

# --- MAIN STRUCTURE WITH TABS ---
tab1, tab2 = st.tabs(["🚨 Clash Analysis", "📈 Loading Forecast"])
with tab1:
    render_clash_tab()
with tab2:
    st.info("Forecasting tab is available.")
