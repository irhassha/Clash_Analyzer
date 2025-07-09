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
    keys_to_clear = ['processed_df', 'clash_summary_df', 'summary_display']
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    st.cache_data.clear()
    st.cache_resource.clear()

# --- FORECASTING FUNCTIONS (No changes here) ---
# ... (All forecasting functions remain the same)

def render_clash_tab():
    """Function to display the entire content of the Clash Analysis tab."""
    st.sidebar.header("⚙️ Upload Your Files")
    schedule_file = st.sidebar.file_uploader("1. Upload Vessel Schedule", type=['xlsx', 'csv'])
    unit_list_file = st.sidebar.file_uploader("2. Upload Unit List", type=['xlsx', 'csv'])
    
    # --- PERUBAHAN DI SINI: Tambahkan input untuk jarak minimum ---
    min_clash_distance = st.sidebar.number_input("Minimum Safe Distance (slots)", min_value=0, value=5, step=1, help="Clash is detected if the distance between vessel allocations is less than or equal to this value.")
    
    process_button = st.sidebar.button("🚀 Process Data", use_container_width=True, type="primary")
    st.sidebar.button("Reset Data", on_click=reset_data, use_container_width=True, help="Clear all processed data and caches to start fresh.")

    if 'processed_df' not in st.session_state:
        st.session_state.processed_df = None
    if 'summary_display' not in st.session_state:
        st.session_state.summary_display = None

    df_vessel_codes = load_vessel_codes_from_repo()
    if process_button:
        if schedule_file and unit_list_file and (df_vessel_codes is not None and not df_vessel_codes.empty):
            with st.spinner('Loading and processing data...'):
                try:
                    # --- Data Loading and Initial Cleaning ---
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

                    # --- PERUBAHAN LOGIKA UTAMA: Membaca dan Memproses Row/Bay ---
                    # 1. Parse 'Row/bay (EXE)' to get numeric slot
                    if 'Row/bay (EXE)' in df_unit_list.columns:
                        df_unit_list['SLOT'] = df_unit_list['Row/bay (EXE)'].astype(str).str.split('-').str[-1]
                        df_unit_list['SLOT'] = pd.to_numeric(df_unit_list['SLOT'], errors='coerce')
                        df_unit_list.dropna(subset=['SLOT'], inplace=True)
                        df_unit_list['SLOT'] = df_unit_list['SLOT'].astype(int)
                    else:
                        st.error("File 'Unit List' must contain a 'Row/bay (EXE)' column for detailed clash detection.")
                        st.stop()
                    
                    # --- Gabungkan semua data ---
                    df_schedule_with_code = pd.merge(df_schedule, df_vessel_codes, left_on="VESSEL", right_on="Description", how="left").rename(columns={"Value": "CODE"})
                    merged_df = pd.merge(df_schedule_with_code, df_unit_list, left_on=['CODE', 'VOY_OUT'], right_on=['Carrier Out', 'Voyage Out'], how='inner')
                    if merged_df.empty: st.warning("No matching data found."); st.session_state.processed_df = None; st.stop()
                    
                    original_vessels_list = df_schedule['VESSEL'].unique().tolist()
                    merged_df = merged_df[merged_df['VESSEL'].isin(original_vessels_list)]
                    excluded_areas = [str(i) for i in range(801, 809)]
                    merged_df['Area (EXE)'] = merged_df['Area (EXE)'].astype(str)
                    filtered_data = merged_df[~merged_df['Area (EXE)'].isin(excluded_areas)]
                    if filtered_data.empty: st.warning("No data left after filtering."); st.session_state.processed_df = None; st.stop()
                    
                    # 2. Buat data frame utama yang berisi informasi slot min/max per kapal per area
                    vessel_area_slots = filtered_data.groupby(['VESSEL', 'VOY_OUT', 'ETA', 'ETD', 'SERVICE', 'Area (EXE)']).agg(
                        MIN_SLOT=('SLOT', 'min'),
                        MAX_SLOT=('SLOT', 'max'),
                        BOX_COUNT=('SLOT', 'count')
                    ).reset_index()

                    # 3. Pivot untuk mendapatkan tampilan seperti sebelumnya, tapi ini hanya untuk AgGrid & Chart
                    pivot_for_display = vessel_area_slots.pivot_table(
                        index=['VESSEL', 'VOY_OUT', 'ETA', 'ETD', 'SERVICE'],
                        columns='Area (EXE)',
                        values='BOX_COUNT',
                        fill_value=0
                    ).reset_index()
                    pivot_for_display['TOTAL BOX'] = pivot_for_display.iloc[:, 5:].sum(axis=1)
                    pivot_for_display['TOTAL CLSTR'] = (pivot_for_display.iloc[:, 5:] > 0).sum(axis=1)

                    st.session_state.processed_df = pivot_for_display.sort_values(by='ETA', ascending=True)
                    st.session_state.vessel_area_slots = vessel_area_slots # Simpan data detail untuk kalkulasi clash
                    st.success("Data processed successfully!")

                except Exception as e:
                    st.error(f"An error occurred during processing: {e}")
                    st.session_state.processed_df = None
        else:
            st.warning("Please upload both files.")

    if st.session_state.get('processed_df') is not None:
        display_df = st.session_state.processed_df.copy()
        vessel_area_slots_df = st.session_state.vessel_area_slots.copy()
        
        # ... (Kode untuk Upcoming Vessel Summary dan Spreading Visualization tidak ditampilkan untuk keringkasan)

        # --- LOGIKA BARU: DETEKSI CLASH BERDASARKAN JARAK SLOT ---
        st.markdown("---")
        st.header("💥 Potential Clash Summary (Based on Slot Distance)")
        
        clash_details = {}
        
        # Ambil daftar unik kapal yang aktif
        active_vessels = vessel_area_slots_df[['VESSEL', 'VOY_OUT', 'ETA', 'ETD']].drop_duplicates()

        # Bandingkan setiap pasang kapal
        for (idx1, vessel1), (idx2, vessel2) in combinations(active_vessels.iterrows(), 2):
            # Cek jika periode waktu tumpang-tindih
            if (vessel1['ETA'] < vessel2['ETD']) and (vessel2['ETA'] < vessel1['ETD']):
                
                # Ambil data slot untuk kedua kapal
                v1_slots = vessel_area_slots_df[vessel_area_slots_df['VESSEL'] == vessel1['VESSEL']]
                v2_slots = vessel_area_slots_df[vessel_area_slots_df['VESSEL'] == vessel2['VESSEL']]
                
                # Cari area yang sama-sama mereka gunakan
                common_areas = pd.merge(v1_slots, v2_slots, on='Area (EXE)', suffixes=('_v1', '_v2'))

                for _, row in common_areas.iterrows():
                    area = row['Area (EXE)']
                    
                    # Tentukan range slot kapal 1 dan 2
                    range1 = (row['MIN_SLOT_v1'], row['MAX_SLOT_v1'])
                    range2 = (row['MIN_SLOT_v2'], row['MAX_SLOT_v2'])
                    
                    # Hitung jarak
                    gap = max(range1[0], range2[0]) - min(range1[1], range2[1]) - 1

                    if gap <= min_clash_distance:
                        period_key = f"Vessels Active During Overlapping Times"
                        if period_key not in clash_details:
                            clash_details[period_key] = []
                        
                        # Hindari duplikasi laporan clash
                        clash_exists = False
                        for existing_clash in clash_details[period_key]:
                            if existing_clash['block'] == area and vessel1['VESSEL'] in existing_clash['vessels'] and vessel2['VESSEL'] in existing_clash['vessels']:
                                clash_exists = True
                                break
                        
                        if not clash_exists:
                             clash_details[period_key].append({
                                "block": area,
                                "vessel1": f"{vessel1['VESSEL']} (Slots: {range1[0]}-{range1[1]})",
                                "vessel2": f"{vessel2['VESSEL']} (Slots: {range2[0]}-{range2[1]})",
                                "gap": gap
                            })

        # --- Tampilan Summary ---
        if not clash_details:
            st.info(f"No potential clashes found with a minimum distance of {min_clash_distance} slots.")
        else:
            for period, clashes in clash_details.items():
                st.markdown(f"**🔥 {period}:**")
                for clash in clashes:
                    st.warning(f"**Block {clash['block']}:** Clash detected between **{clash['vessel1']}** and **{clash['vessel2']}**. (Gap: {clash['gap']} slots)")

        # --- Tabel AgGrid dan komponen lainnya ---
        # ... (Kode AgGrid, Download, dll. perlu disesuaikan dengan struktur data 'pivot_for_display')

    else:
        st.info("Welcome! Please upload your files and click 'Process Data' to begin.")

# --- MAIN STRUCTURE WITH TABS ---
tab1, tab2 = st.tabs(["🚨 Clash Analysis", "📈 Loading Forecast"])
with tab1:
    render_clash_tab()
with tab2:
    st.info("Forecasting tab is available.") # Placeholder, karena fungsi forecast tidak ditampilkan di sini
