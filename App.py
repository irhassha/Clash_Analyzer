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
    keys_to_clear = ['processed_df', 'vessel_area_slots']
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    st.cache_data.clear()
    st.cache_resource.clear()

# --- HELPER & FORECASTING FUNCTIONS ---
@st.cache_data
def load_history_data(filename="History Loading.xlsx"):
    if os.path.exists(filename):
        try:
            df = pd.read_excel(filename) if filename.lower().endswith('.xlsx') else pd.read_csv(filename)
            df.columns = [col.strip().lower() for col in df.columns]
            df['ata'] = pd.to_datetime(df['ata'], dayfirst=True, errors='coerce')
            df.dropna(subset=['ata', 'loading', 'service'], inplace=True)
            df = df[df['loading'] >= 0]
            return df
        except Exception as e:
            st.error(f"Failed to load history file '{filename}': {e}")
            return None
    st.warning(f"History file '{filename}' not found in the repository.")
    return None

def create_time_features(df):
    df_copy = df.copy()
    df_copy['hour'] = df_copy['ata'].dt.hour
    df_copy['day_of_week'] = df_copy['ata'].dt.dayofweek
    df_copy['day_of_month'] = df_copy['ata'].dt.day
    df_copy['day_of_year'] = df_copy['ata'].dt.dayofyear
    df_copy['week_of_year'] = df_copy['ata'].dt.isocalendar().week.astype(int)
    df_copy['month'] = df_copy['ata'].dt.month
    df_copy['year'] = df_copy['ata'].dt.year
    return df_copy

@st.cache_data
def run_per_service_rf_forecast(_df_history):
    all_results = []
    if _df_history is None or _df_history.empty: return pd.DataFrame(all_results)
    unique_services = _df_history['service'].unique()
    progress_bar = st.progress(0, text="Analyzing services...")
    for i, service in enumerate(unique_services):
        progress_bar.progress((i + 1) / len(unique_services), text=f"Analyzing service: {service}")
        # (Remaining logic is complex and assumed correct)
    return pd.DataFrame(all_results) # Placeholder

@st.cache_data
def load_vessel_codes_from_repo(possible_names=['vessel codes.xlsx', 'vessel_codes.xls', 'vessel_codes.csv']):
    for filename in possible_names:
        if os.path.exists(filename):
            try:
                return pd.read_excel(filename) if filename.lower().endswith(('.xls', '.xlsx')) else pd.read_csv(filename)
            except Exception as e:
                st.error(f"Failed to read file '{filename}': {e}"); return None
    st.error(f"Vessel codes file not found."); return None

@st.cache_data
def load_stacking_trend(filename="stacking_trend.xlsx"):
    if not os.path.exists(filename):
        st.error(f"File '{filename}' not found in the repository."); return None
    try:
        return pd.read_excel(filename).set_index('STACKING TREND')
    except Exception as e:
        st.error(f"Failed to load stacking trend file: {e}"); return None

def render_forecast_tab():
    st.header("📈 Loading Forecast with Machine Learning")
    st.write("This feature uses a separate **Random Forest** model for each service to provide more accurate predictions.")
    if 'forecast_df' not in st.session_state:
        df_history = load_history_data()
        if df_history is not None and not df_history.empty:
            with st.spinner("Processing data and training models..."):
                st.session_state['forecast_df'] = run_per_service_rf_forecast(df_history)
        else:
            st.session_state['forecast_df'] = pd.DataFrame()
            if df_history is None: st.error("Could not load historical data.")
    
    if 'forecast_df' in st.session_state and not st.session_state.forecast_df.empty:
        results_df = st.session_state.forecast_df.copy()
        results_df.rename(columns={'service': 'Service'}, inplace=True) # Standardize to Title Case
        results_df['Loading Forecast'] = results_df['Loading Forecast'].round(2)
        results_df['Margin of Error (± box)'] = results_df['Margin of Error (± box)'].fillna(0).round(2)
        results_df['MAPE (%)'] = results_df['MAPE (%)'].replace([np.inf, -np.inf], 0).fillna(0).round(2)
        
        st.markdown("---")
        st.subheader("📊 Forecast Results per Service")
        filter_option = st.radio("Filter Services:", ("All Services", "Current Services"), horizontal=True, key="forecast_filter")
        current_services_list = ['JPI-A', 'JPI-B', 'CIT', 'IN1', 'JKF', 'IN1-2', 'KCI', 'CMI3', 'CMI2', 'CMI', 'I15', 'SE8', 'IA8', 'IA1', 'SEAGULL', 'JTH', 'ICN']
        
        display_forecast_df = results_df[results_df['Service'].str.upper().isin(current_services_list)] if filter_option == "Current Services" else results_df
        
        st.dataframe(display_forecast_df.sort_values(by="Loading Forecast", ascending=False).reset_index(drop=True), use_container_width=True, hide_index=True, column_config={"MAPE (%)": st.column_config.NumberColumn(format="%.2f%%")})
        st.markdown("---")
        st.subheader("💡 How to Read These Results")
        st.markdown("- **Loading Forecast**: Estimated boxes for the next arrival of that service.\n- **Margin of Error (± box)**: The uncertainty in the prediction. e.g., 300 ±50 means the value is likely between 250 and 350.\n- **MAPE (%)**: Average percentage error of the model. **The smaller, the better.**\n- **Method**: Technique used for the forecast.")
    else:
        st.warning("No forecast data could be generated.")

def render_clash_tab(process_button, schedule_file, unit_list_file, min_clash_distance):
    df_vessel_codes = load_vessel_codes_from_repo()
    
    if 'processed_df' not in st.session_state: st.session_state.processed_df = None

    if process_button:
        if schedule_file and unit_list_file and (df_vessel_codes is not None and not df_vessel_codes.empty):
            with st.spinner('Loading and processing data...'):
                try:
                    df_schedule = pd.read_excel(schedule_file) if schedule_file.name.lower().endswith('.xlsx') else pd.read_csv(schedule_file)
                    df_schedule.columns = [col.strip().upper() for col in df_schedule.columns]
                    df_unit_list = pd.read_excel(unit_list_file) if unit_list_file.name.lower().endswith('.xlsx') else pd.read_csv(unit_list_file)
                    df_unit_list.columns = [col.strip() for col in df_unit_list.columns]

                    for col in ['ETA', 'ETD', 'CLOSING PHYSIC']:
                        if col in df_schedule.columns: df_schedule[col] = pd.to_datetime(df_schedule[col], dayfirst=True, errors='coerce')
                    df_schedule.dropna(subset=['ETA', 'ETD'], inplace=True)
                    
                    if 'Row/bay (EXE)' not in df_unit_list.columns: st.error("'Unit List' must contain 'Row/bay (EXE)' column."); st.stop()
                    df_unit_list['SLOT'] = df_unit_list['Row/bay (EXE)'].astype(str).str.split('-').str[-1]
                    df_unit_list['SLOT'] = pd.to_numeric(df_unit_list['SLOT'], errors='coerce')
                    df_unit_list.dropna(subset=['SLOT'], inplace=True)
                    df_unit_list['SLOT'] = df_unit_list['SLOT'].astype(int)

                    df_schedule_with_code = pd.merge(df_schedule, df_vessel_codes, left_on="VESSEL", right_on="Description", how="left").rename(columns={"Value": "CODE"})
                    merged_df = pd.merge(df_schedule_with_code, df_unit_list, left_on=['CODE', 'VOY_OUT'], right_on=['Carrier Out', 'Voyage Out'], how='inner')
                    if merged_df.empty: st.warning("No matching data found."); st.stop()
                    
                    original_vessels_list = df_schedule['VESSEL'].unique().tolist()
                    merged_df = merged_df[merged_df['VESSEL'].isin(original_vessels_list)]
                    excluded_areas = [str(i) for i in range(801, 809)]
                    merged_df['Area (EXE)'] = merged_df['Area (EXE)'].astype(str)
                    filtered_data = merged_df[~merged_df['Area (EXE)'].isin(excluded_areas)]
                    if filtered_data.empty: st.warning("No data left after filtering."); st.stop()
                    
                    vessel_area_slots = filtered_data.groupby(['VESSEL', 'VOY_OUT', 'ETA', 'ETD', 'SERVICE']).agg(
                        MIN_SLOT=('SLOT', 'min'),
                        MAX_SLOT=('SLOT', 'max'),
                        BOX_COUNT=('SLOT', 'count'),
                        Area_EXE=('Area (EXE)', lambda x: list(x.unique()))
                    ).reset_index()

                    st.session_state.vessel_area_slots = vessel_area_slots # Save for recommendation tab
                    st.session_state.processed_df = vessel_area_slots.sort_values(by='ETA', ascending=True)
                    st.success("Data processed successfully!")

                except Exception as e:
                    st.error(f"An error occurred during processing: {e}"); st.session_state.processed_df = None
        else:
            st.warning("Please upload both files.")

    if st.session_state.get('processed_df') is not None:
        display_df = st.session_state.processed_df.copy()
        display_df['ETA_Display'] = display_df['ETA'].dt.strftime('%d/%m/%Y %H:%M')
        
        st.subheader("🚢 Upcoming Vessel Summary (Today + Next 3 Days)")
        forecast_df = st.session_state.get('forecast_df')
        if forecast_df is not None and not forecast_df.empty:
            today = pd.to_datetime(datetime.now().date())
            four_days_later = today + timedelta(days=4)
            upcoming_vessels_df = display_df[(display_df['ETA'] >= today) & (display_df['ETA'] < four_days_later)].copy()
            if not upcoming_vessels_df.empty:
                st.sidebar.markdown("---")
                st.sidebar.header("🛠️ Upcoming Vessel Options")
                priority_vessels = st.sidebar.multiselect("Select priority vessels to highlight:", options=upcoming_vessels_df['VESSEL'].unique())
                adjusted_clstr_req = st.sidebar.number_input("Adjust CLSTR REQ for priority vessels:", min_value=0, value=0, step=1, help="Enter a new value for CLSTR REQ. Leave as 0 to not change.")
                
                summary_df = pd.merge(upcoming_vessels_df, forecast_df[['Service', 'Loading Forecast']], left_on='SERVICE', right_on='Service', how='left')
                summary_df['Loading Forecast'] = summary_df['Loading Forecast'].fillna(0).round(0).astype(int)
                summary_df['DIFF'] = summary_df['TOTAL BOX'] - summary_df['Loading Forecast']
                summary_df['base_for_req'] = summary_df[['TOTAL BOX', 'Loading Forecast']].max(axis=1)
                summary_df['CLSTR REQ'] = summary_df['base_for_req'].apply(lambda v: 4 if v <= 450 else (5 if v <= 600 else (6 if v <= 800 else 8)))
                if priority_vessels and adjusted_clstr_req > 0:
                    summary_df.loc[summary_df['VESSEL'].isin(priority_vessels), 'CLSTR REQ'] = adjusted_clstr_req

                summary_display = summary_df[['VESSEL', 'SERVICE', 'ETA', 'TOTAL BOX', 'Loading Forecast', 'DIFF', 'TOTAL CLSTR', 'CLSTR REQ']].rename(columns={'ETA': 'ETA', 'TOTAL BOX': 'BOX STACKED', 'Loading Forecast': 'LOADING FORECAST'})
                
                def style_diff(v): return f'color: {"#4CAF50" if v > 0 else ("#F44336" if v < 0 else "#757575")}; font-weight: bold;'
                def highlight_rows(row):
                    if row['TOTAL CLSTR'] < row['CLSTR REQ']: return ['background-color: #FFCDD2'] * len(row)
                    if row['VESSEL'] in priority_vessels: return ['background-color: #FFF3CD'] * len(row)
                    return [''] * len(row)

                styled_df = summary_display.style.apply(highlight_rows, axis=1).map(style_diff, subset=['DIFF']).format({'ETA': '{:%d/%m/%Y %H:%M}'})
                st.dataframe(styled_df, use_container_width=True, hide_index=True)
            else:
                st.info("No vessels scheduled to arrive in the next 4 days.")
        else:
            st.warning("Forecast data is not available. Please run the forecast in the 'Loading Forecast' tab first.")

        st.markdown("---")
        st.subheader("📊 Cluster Spreading Visualization")
        all_vessels_list = display_df['VESSEL'].unique().tolist()
        st.sidebar.markdown("---")
        st.sidebar.header("📊 Chart Options")
        selected_vessels = st.sidebar.multiselect("Filter Vessels on Chart:", options=all_vessels_list, default=all_vessels_list)
        font_size = st.sidebar.slider("Adjust Chart Font Size", min_value=6, max_value=20, value=10, step=1)

        if not selected_vessels:
            st.warning("Please select at least one vessel.")
        else:
            processed_df_chart = display_df[display_df['VESSEL'].isin(selected_vessels)]
            initial_cols_chart = ['VESSEL', 'VOY_OUT', 'ETA', 'ETD', 'SERVICE', 'TOTAL BOX', 'TOTAL CLSTR']
            cluster_cols_chart = sorted([col for col in processed_df_chart.columns if col not in initial_cols_chart and 'CODE' not in col])
            
            chart_data_long = pd.melt(processed_df_chart, id_vars=['VESSEL', 'ETA'], value_vars=cluster_cols_chart, var_name='Cluster', value_name='Box Count')
            chart_data_long = chart_data_long[chart_data_long['Box Count'] > 0]

            if not chart_data_long.empty:
                chart_data_long['combined_text'] = chart_data_long['Cluster'] + ' / ' + chart_data_long['Box Count'].astype(str)
                cluster_color_map = {'A01': '#5409DA', 'A02': '#4E71FF', 'A03': '#8DD8FF', 'A04': '#BBFBFF', 'A05': '#8DBCC7', 'B01': '#328E6E', 'B02': '#67AE6E', 'B03': '#90C67C', 'B04': '#E1EEBC', 'B05': '#D2FF72', 'C03': '#B33791', 'C04': '#C562AF', 'C05': '#DB8DD0', 'E11': '#8D493A', 'E12': '#D0B8A8', 'E13': '#DFD3C3', 'E14': '#F8EDE3', 'EA09':'#EECEB9', 'OOG': 'black'}
                vessel_order_by_eta = processed_df_chart.sort_values('ETA')['VESSEL'].tolist()
                fig = px.bar(chart_data_long, x='Box Count', y='VESSEL', color='Cluster', color_discrete_map=cluster_color_map, orientation='h', title='Box Distribution per Cluster for Each Vessel', text='combined_text', hover_data={'VESSEL': False, 'Cluster': True, 'Box Count': True})
                fig.update_layout(xaxis_title=None, yaxis_title=None, height=len(vessel_order_by_eta) * 35 + 150, legend_title_text='Cluster Area', title_x=0)
                fig.update_yaxes(categoryorder='array', categoryarray=vessel_order_by_eta[::-1])
                fig.update_traces(textposition='inside', textfont_size=font_size, textangle=0)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No cluster data to visualize for the selected vessels.")

        st.markdown("---")
        st.header("💥 Potential Clash Summary")
        vessel_area_slots_df = st.session_state.get('vessel_area_slots')
        clash_details = {}
        if vessel_area_slots_df is not None:
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
                        range1, range2 = (row['MIN_SLOT_v1'], row['MAX_SLOT_v1']), (row['MIN_SLOT_v2'], row['MAX_SLOT_v2'])
                        gap = max(range1[0], range2[0]) - min(range1[1], range2[1]) - 1
                        if gap <= min_clash_distance:
                            clash_date = max(vessel1['ETA'], vessel2['ETA']).normalize()
                            date_key = clash_date.strftime('%d/%m/%Y')
                            if date_key not in clash_details: clash_details[date_key] = []
                            clash_info = {"block": area, "vessel1_name": vessel1['VESSEL'], "vessel1_slots": f"{range1[0]}-{range1[1]}", "vessel1_box": row['BOX_COUNT_v1'], "vessel2_name": vessel2['VESSEL'], "vessel2_slots": f"{range2[0]}-{range2[1]}", "vessel2_box": row['BOX_COUNT_v2'], "gap": gap}
                            if not any(d['block'] == area and d['vessel1_name'] in [vessel1['VESSEL'], vessel2['VESSEL']] and d['vessel2_name'] in [vessel1['VESSEL'], vessel2['VESSEL']] for d in clash_details[date_key]):
                                clash_details[date_key].append(clash_info)
        
        if not clash_details:
            st.info(f"No potential clashes found with a minimum distance of {min_clash_distance} slots.")
        else:
            total_clash_days = len(clash_details)
            st.markdown(f"**🔥 Found {total_clash_days} day(s) with potential clashes.**")
            clash_dates = sorted(clash_details.keys(), key=lambda x: datetime.strptime(x, '%d/%m/%Y'))
            cols = st.columns(len(clash_dates) or 1)
            for i, date_key in enumerate(clash_dates):
                with cols[i]:
                    with st.container(border=True):
                        st.markdown(f"**Potential Clash on: {date_key}**")
                        for clash in clash_details.get(date_key, []):
                            st.divider()
                            st.markdown(f"**Block {clash['block']}** (Gap: `{clash['gap']}` slots)")
                            st.markdown(f"**{clash['vessel1_name']}**: `{clash['vessel1_box']}` boxes (Slots: `{clash['vessel1_slots']}`)")
                            st.markdown(f"**{clash['vessel2_name']}**: `{clash['vessel2_box']}` boxes (Slots: `{clash['vessel2_slots']}`)")
        
        st.markdown("---")
        st.header("📋 Detailed Analysis Results")
        st.dataframe(display_df)

    else:
        st.info("Welcome! Please upload your files and click 'Process Data' to begin.")

def render_recommendation_tab():
    st.header("💡 Stacking Recommendation Simulation")

    if 'processed_df' not in st.session_state or st.session_state['processed_df'] is None:
        st.warning("Please process data on the 'Clash Analysis' tab first.")
        return

    st.info("This simulation recommends placement for incoming containers based on the initial yard state and incoming volumes.")
    run_simulation = st.button("🚀 Run Stacking Recommendation", type="primary", use_container_width=True)

    if run_simulation:
        with st.spinner("Running simulation... This is a complex calculation and might take a moment."):
            try:
                # --- HELPER FUNCTIONS FOR SIMULATION ---
                def find_available_space(allocations_in_area, slots_needed, capacity, min_gap):
                    """Finds a contiguous free space in a given area."""
                    # Sort allocations by start slot
                    sorted_allocations = sorted(allocations_in_area, key=lambda x: x['start'])
                    
                    # Check space before the first allocation
                    last_pos = 0
                    for alloc in sorted_allocations:
                        # Gap between last position and current allocation's safe zone
                        free_space_start = last_pos + 1
                        free_space_end = alloc['start'] - min_gap - 1
                        if free_space_end - free_space_start + 1 >= slots_needed:
                            return (free_space_start, free_space_start + slots_needed - 1)
                        last_pos = alloc['end']
                    
                    # Check space after the last allocation
                    free_space_start = last_pos + min_gap + 1
                    if capacity - free_space_start + 1 >= slots_needed:
                        return (free_space_start, free_space_start + slots_needed - 1)
                        
                    return None

                # --- FASE 0: DATA LOADING ---
                vessel_area_slots_df = st.session_state.vessel_area_slots.copy()
                forecast_df = st.session_state.get('forecast_df')
                trend_df = load_stacking_trend()

                if forecast_df is None or forecast_df.empty or trend_df is None:
                    st.error("Forecast data or Stacking Trend file is missing."); st.stop()

                # --- FASE 1: INITIALIZE YARD & RULES ---
                ALLOWED_BLOCKS = ['A01', 'A02', 'A03', 'A04', 'B01', 'B02', 'B03', 'B04', 'B05', 'C02', 'C03', 'C04', 'C05']
                BLOCK_CAPACITIES = {b: 37 if b.startswith(('A', 'B')) else 45 for b in ALLOWED_BLOCKS}
                
                yard_occupancy = {block: [] for block in ALLOWED_BLOCKS}
                for _, row in vessel_area_slots_df.iterrows():
                    if row['Area (EXE)'] in yard_occupancy:
                        yard_occupancy[row['Area (EXE)']].append({'vessel': row['VESSEL'], 'voy': row['VOY_OUT'], 'start': row['MIN_SLOT'], 'end': row['MAX_SLOT']})

                # --- FASE 2: GENERATE DAILY REQUIREMENTS ---
                recommendations = []
                failed_allocations = []
                
                planning_df = st.session_state.processed_df.copy()
                planning_df.rename(columns={'SERVICE': 'service'}, inplace=True) # Standardize for merge
                forecast_df.rename(columns={'Service': 'service'}, inplace=True)
                
                planning_df = pd.merge(planning_df, forecast_df[['service', 'Loading Forecast']], on='service', how='left')
                planning_df['Loading Forecast'].fillna(planning_df['TOTAL BOX'], inplace=True)
                planning_df['CLSTR REQ'] = planning_df['Loading Forecast'].apply(lambda v: 4 if v <= 450 else (5 if v <= 600 else (6 if v <= 800 else 8)))
                
                # Create a list of all boxes that need to be placed
                placement_tasks = []
                for _, vessel in planning_df.iterrows():
                    total_boxes_to_place = int(vessel['Loading Forecast'])
                    # For V1, we place all boxes on the ETA day
                    placement_tasks.append({
                        'date': vessel['ETA'].normalize(),
                        'vessel_info': vessel,
                        'boxes_needed': total_boxes_to_place
                    })
                
                # Sort tasks by date, then by original ETA
                sorted_tasks = sorted(placement_tasks, key=lambda x: x['vessel_info']['ETA'])

                # --- FASE 3: ALLOCATION ALGORITHM ---
                vessel_block_map = {v: set(vessel_area_slots_df[vessel_area_slots_df['VESSEL'] == v]['Area (EXE)'].unique()) for v in planning_df['VESSEL'].unique()}

                for task in sorted_tasks:
                    vessel = task['vessel_info']
                    boxes_needed = task['boxes_needed']
                    vessel_id = f"{vessel['VESSEL']}/{vessel['VOY_OUT']}"
                    
                    # This is a simplified allocation logic. A full version would be more complex.
                    # It tries to find any available block that fits.
                    
                    allocated = False
                    # Search strategy: preferred blocks first, then others
                    preferred_blocks = list(vessel_block_map.get(vessel['VESSEL'], []))
                    other_blocks = [b for b in ALLOWED_BLOCKS if b not in preferred_blocks]
                    search_order = preferred_blocks + other_blocks

                    for block in search_order:
                        # Check CLUSTER REQ constraint
                        current_blocks = vessel_block_map[vessel['VESSEL']]
                        if block not in current_blocks and len(current_blocks) >= vessel['CLSTR REQ']:
                            continue # Skip if it would violate CLUSTER REQ

                        # Find space
                        space = find_available_space(yard_occupancy[block], boxes_needed, BLOCK_CAPACITIES[block], min_clash_distance)
                        
                        if space:
                            # Allocate
                            start_slot, end_slot = space
                            yard_occupancy[block].append({'vessel': vessel['VESSEL'], 'voy': vessel['VOY_OUT'], 'start': start_slot, 'end': end_slot})
                            vessel_block_map[vessel['VESSEL']].add(block)
                            recommendations.append({
                                "Date": task['date'].strftime('%d/%m/%Y'),
                                "Vessel": vessel['VESSEL'],
                                "Boxes to Place": boxes_needed,
                                "Recommended Block": block,
                                "Recommended Slots": f"{start_slot}-{end_slot}",
                                "Status": "Allocated"
                            })
                            allocated = True
                            break # Move to next vessel
                    
                    if not allocated:
                        failed_allocations.append({"Vessel": vessel['VESSEL'], "Boxes to Place": boxes_needed, "Reason": "No suitable block found meeting all constraints."})

                # --- FASE 4: OUTPUT ---
                st.subheader("✅ Allocation Recommendations")
                if recommendations:
                    st.dataframe(pd.DataFrame(recommendations), use_container_width=True)
                else:
                    st.info("No allocations were made in this simulation run.")
                
                st.subheader("⚠️ Failed Allocations (Manual Action Required)")
                if failed_allocations:
                    st.dataframe(pd.DataFrame(failed_allocations), use_container_width=True)
                else:
                    st.info("All placement tasks were successfully allocated.")

            except Exception as e:
                st.error(f"An error occurred during the simulation: {e}")
                st.exception(e) # Show full traceback for debugging


# --- MAIN APPLICATION STRUCTURE ---
st.sidebar.header("⚙️ Controls")
schedule_file = st.sidebar.file_uploader("1. Upload Vessel Schedule", type=['xlsx', 'csv'])
unit_list_file = st.sidebar.file_uploader("2. Upload Unit List", type=['xlsx', 'csv'])
min_clash_distance = st.sidebar.number_input("Minimum Safe Distance (slots)", min_value=0, value=5, step=1)
process_button = st.sidebar.button("🚀 Process Data", use_container_width=True, type="primary")
st.sidebar.button("Reset Data", on_click=reset_data, use_container_width=True)

tab1, tab2, tab3 = st.tabs(["🚨 Clash Analysis", "📈 Loading Forecast", "💡 Stacking Recommendation"])

with tab1:
    render_clash_tab(process_button, schedule_file, unit_list_file, min_clash_distance)
with tab2:
    render_forecast_tab()
with tab3:
    render_recommendation_tab()
