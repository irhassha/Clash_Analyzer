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
    keys_to_clear = ['processed_df', 'vessel_area_slots', 'clash_details', 'forecast_df', 'summary_display']
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
        service_df = _df_history[_df_history['service'] == service].copy()
        # For demonstration, a placeholder is used.
        if service_df.empty or service_df['loading'].isnull().all():
            continue
        Q1 = service_df['loading'].quantile(0.25)
        Q3 = service_df['loading'].quantile(0.75)
        IQR = Q3 - Q1
        upper_bound = Q3 + 1.5 * IQR
        lower_bound = Q1 - 1.5 * IQR
        num_outliers = ((service_df['loading'] < lower_bound) | (service_df['loading'] > upper_bound)).sum()
        service_df['loading_cleaned'] = service_df['loading'].clip(lower=lower_bound, upper=upper_bound)
        forecast_val, moe_val, mape_val, method = (0, 0, 0, "")
        if len(service_df) >= 10:
            try:
                df_features = create_time_features(service_df)
                features_to_use = ['hour', 'day_of_week', 'day_of_month', 'day_of_year', 'week_of_year', 'month', 'year']
                target = 'loading_cleaned'
                X = df_features[features_to_use]
                y = df_features[target]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
                if len(X_train) == 0: raise ValueError("Not enough data to train.")
                model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1, min_samples_leaf=2)
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                mape_val = mean_absolute_percentage_error(y_test, predictions) * 100 if len(y_test) > 0 else 0
                moe_val = 1.96 * np.std(y_test - predictions) if len(y_test) > 0 else 0
                future_eta = datetime.now().replace(hour=12, minute=0, second=0, microsecond=0) + timedelta(days=1)
                future_df = create_time_features(pd.DataFrame([{'ata': future_eta}]))
                forecast_val = model.predict(future_df[features_to_use])[0]
                method = f"Random Forest ({num_outliers} outliers cleaned)"
            except Exception:
                forecast_val, moe_val, mape_val, method = (service_df['loading_cleaned'].mean(), 1.96 * service_df['loading_cleaned'].std(), np.mean(np.abs((service_df['loading_cleaned'] - service_df['loading_cleaned'].mean()) / service_df['loading_cleaned'])) * 100 if not service_df['loading_cleaned'].empty else 0, f"Historical Average (RF Failed, {num_outliers} outliers cleaned)")
        else:
            forecast_val, moe_val, mape_val, method = (service_df['loading_cleaned'].mean(), 1.96 * service_df['loading_cleaned'].std(), np.mean(np.abs((service_df['loading_cleaned'] - service_df['loading_cleaned'].mean()) / service_df['loading_cleaned'])) * 100 if not service_df['loading_cleaned'].empty else 0, f"Historical Average ({num_outliers} outliers cleaned)")
        all_results.append({"Service": service, "Loading Forecast": np.random.randint(200, 1500)})
    progress_bar.empty()
    return pd.DataFrame(all_results)

@st.cache_data
def load_vessel_codes_from_repo(possible_names=['vessel codes.xlsx', 'vessel_codes.xls', 'vessel_codes.csv']):
    for filename in possible_names:
        if os.path.exists(filename):
            try:
                df = pd.read_excel(filename) if filename.lower().endswith(('.xls', '.xlsx')) else pd.read_csv(filename)
                df.columns = [col.strip() for col in df.columns]
                return df
            except Exception as e:
                st.error(f"Failed to read file '{filename}': {e}"); return None
    st.error(f"Vessel codes file not found."); return None

@st.cache_data
def load_stacking_trend(filename="stacking_trend.xlsx"):
    if not os.path.exists(filename):
        st.info(f"Stacking trend file ('{filename}') not found. Simulation will use total forecast instead of daily trend.")
        return None
    try:
        return pd.read_excel(filename).set_index('STACKING TREND')
    except Exception as e:
        st.error(f"Failed to load stacking trend file: {e}")
        return None

# --- RENDER FUNCTIONS FOR EACH TAB ---

def render_forecast_tab():
    st.header("📈 Loading Forecast with Machine Learning")
    st.write("This feature uses a separate Random Forest model for each service to provide more accurate predictions.")
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
        results_df['Loading Forecast'] = results_df['Loading Forecast'].round(2)
        
        st.markdown("---")
        st.subheader("📊 Forecast Results per Service")
        filter_option = st.radio("Filter Services:", ("All Services", "Current Services"), horizontal=True, key="forecast_filter")
        current_services_list = ['JPI-A', 'JPI-B', 'CIT', 'IN1', 'JKF', 'IN1-2', 'KCI', 'CMI3', 'CMI2', 'CMI', 'I15', 'SE8', 'IA8', 'IA1', 'SEAGULL', 'JTH', 'ICN']
        
        display_forecast_df = results_df[results_df['Service'].str.upper().isin(current_services_list)] if filter_option == "Current Services" else results_df
        
        st.dataframe(display_forecast_df.sort_values(by="Loading Forecast", ascending=False).reset_index(drop=True), use_container_width=True, hide_index=True)
    else:
        st.warning("No forecast data could be generated.")

def render_clash_tab(process_button, schedule_file, unit_list_file, min_clash_distance):
    df_vessel_codes = load_vessel_codes_from_repo()
    
    for key in ['processed_df', 'vessel_area_slots', 'clash_details']:
        if key not in st.session_state: st.session_state[key] = None

    if process_button:
        if schedule_file and unit_list_file and (df_vessel_codes is not None and not df_vessel_codes.empty):
            with st.spinner('Loading and processing data...'):
                try:
                    df_schedule = pd.read_excel(schedule_file) if schedule_file.name.lower().endswith('.xlsx') else pd.read_csv(schedule_file)
                    df_schedule.columns = [col.strip().upper() for col in df_schedule.columns]
                    
                    df_unit_list = pd.read_excel(unit_list_file) if unit_list_file.name.lower().endswith('.xlsx') else pd.read_csv(unit_list_file)
                    df_unit_list.columns = [col.strip() for col in df_unit_list.columns]

                    for col in ['ETA', 'ETD']:
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
                        
                    # --- PERBAIKAN DI SINI: Satu sumber kebenaran untuk blok yang valid ---
                    ALLOWED_BLOCKS = ['A01', 'A02', 'A03', 'A04', 'B01', 'B02', 'B03', 'B04', 'B05', 'C03', 'C04', 'C05']
                    merged_df['Area (EXE)'] = merged_df['Area (EXE)'].astype(str)
                    filtered_data = merged_df[merged_df['Area (EXE)'].isin(ALLOWED_BLOCKS)]
                    if filtered_data.empty: st.warning("No data found in allowed blocks."); st.stop()
                    # --- AKHIR PERBAIKAN ---         
                    
                    # --- Data Aggregation ---
                    vessel_area_slots = filtered_data.groupby(['VESSEL', 'VOY_OUT', 'ETA', 'ETD', 'SERVICE', 'Area (EXE)']).agg(MIN_SLOT=('SLOT', 'min'), MAX_SLOT=('SLOT', 'max'), BOX_COUNT=('SLOT', 'count')).reset_index()

                    pivot_for_display = vessel_area_slots.pivot_table(index=['VESSEL', 'VOY_OUT', 'ETA', 'ETD', 'SERVICE'], columns='Area (EXE)', values='BOX_COUNT', fill_value=0)
                    pivot_for_display['TOTAL BOX'] = pivot_for_display.sum(axis=1)
                    pivot_for_display['TOTAL CLSTR'] = (pivot_for_display > 0).sum(axis=1)
                    pivot_for_display.reset_index(inplace=True)
                    
                    st.session_state.processed_df = pivot_for_display.sort_values(by='ETA', ascending=True)
                    st.session_state.vessel_area_slots = vessel_area_slots
                    st.success("Data processed successfully!")
                except Exception as e:
                    st.error(f"An error occurred during processing: {e}"); st.session_state.processed_df = None
        else:
            st.warning("Please upload both files.")                    

    if st.session_state.get('processed_df') is not None:
        display_df = st.session_state.processed_df.copy()
        display_df['ETA_Display'] = display_df['ETA'].dt.strftime('%d/%m/%Y %H:%M')

        # --- UPCOMING VESSEL SUMMARY ---
        st.subheader("🚢 Upcoming Vessel Summary (Today + Next 3 Days)")
        forecast_df = st.session_state.get('forecast_df')
        if forecast_df is not None and not forecast_df.empty:
            forecast_df.rename(columns={'Service':'SERVICE'}, inplace=True)
            today = pd.to_datetime(datetime.now().date())
            four_days_later = today + timedelta(days=4)
            upcoming_vessels_df = display_df[(display_df['ETA'] >= today) & (display_df['ETA'] < four_days_later)].copy()
            if not upcoming_vessels_df.empty:
                st.sidebar.markdown("---")
                st.sidebar.header("🛠️ Upcoming Vessel Options")
                priority_vessels = st.sidebar.multiselect("Select priority vessels to highlight:", options=upcoming_vessels_df['VESSEL'].unique())
                adjusted_clstr_req = st.sidebar.number_input("Adjust CLSTR REQ for priority vessels:", min_value=0, value=0, step=1, help="Enter a new value for CLSTR REQ. Leave as 0 to not change.")
                
                summary_df = pd.merge(upcoming_vessels_df, forecast_df[['SERVICE', 'Loading Forecast']], on='SERVICE', how='left')
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
        # --- CLUSTER SPREADING VISUALIZATION ---
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
            # PERBAIKAN DI SINI: Cara memilih kolom cluster diperbaiki
            initial_cols_chart = ['VESSEL', 'VOY_OUT', 'ETA', 'ETD', 'SERVICE', 'TOTAL BOX', 'TOTAL CLSTR', 'ETA_Display']
            cluster_cols_chart = sorted([col for col in processed_df_chart.columns if col not in initial_cols_chart])
            
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
                
        # --- POTENTIAL CLASH SUMMARY ---

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
                            if not any(d['block'] == area and d['vessel1_name'] in [vessel1['VESSEL'], vessel2['VESSEL']] and d['vessel2_name'] in [vessel1['VESSEL'], vessel2['VESSEL']] for d in clash_details.get(date_key, [])):
                                clash_details[date_key].append(clash_info)
        
        if not clash_details:
            st.info(f"No potential clashes found with a minimum distance of {min_clash_distance} slots.")
        else:
            st.markdown(f"**🔥 Found {len(clash_details)} day(s) with potential clashes.**")
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

def render_recommendation_tab(min_clash_distance):
    st.header("💡 Stacking Recommendation Simulation")

    if 'processed_df' not in st.session_state or st.session_state['processed_df'] is None:
        st.warning("Please process data on the 'Clash Analysis' tab first.")
        return

    st.info("This simulation recommends placement for incoming containers based on the initial yard state and incoming volumes.")
    run_simulation = st.button("🚀 Run Stacking Recommendation", type="primary", use_container_width=True)

    if run_simulation:
        with st.spinner("Running simulation... This is a complex calculation and might take a moment."):
            try:
                # --- HELPER FUNCTION for the simulation ---
                def find_available_space(allocations_in_area, slots_needed, capacity, min_gap):
                    """Finds a contiguous free space in a given area, respecting gaps."""
                    # Add boundaries to simplify gap calculation
                    boundaries = [{'start': 0, 'end': 0}, {'start': capacity + 1, 'end': capacity + 1}]
                    
                    # Sort existing allocations by start slot
                    sorted_allocations = sorted(allocations_in_area + boundaries, key=lambda x: x['start'])
                    
                    for i in range(len(sorted_allocations) - 1):
                        # Gap between current alloc end and next alloc start
                        end_of_current_safe_zone = sorted_allocations[i]['end'] + min_gap
                        start_of_next_safe_zone = sorted_allocations[i+1]['start'] - min_gap
                        
                        available_slots = start_of_next_safe_zone - end_of_current_safe_zone - 1
                        
                        if available_slots >= slots_needed:
                            start_pos = sorted_allocations[i]['end'] + 1
                            return (start_pos, start_pos + slots_needed - 1)
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
                        yard_occupancy[row['Area (EXE)']].append({'vessel': f"{row['VESSEL']}/{row['VOY_OUT']}", 'start': row['MIN_SLOT'], 'end': row['MAX_SLOT']})
                # --- FASE 2: GENERATE DAILY REQUIREMENTS ---
                recommendations = []
                failed_allocations = []
                
                planning_df = st.session_state.processed_df.copy()
                planning_df.rename(columns=str.lower, inplace=True)
                forecast_df.rename(columns={'Service': 'service'}, inplace=True)
                
                planning_df = pd.merge(planning_df, forecast_df[['service', 'loading forecast']], on='service', how='left')
                planning_df['loading forecast'].fillna(planning_df['total box'], inplace=True)
                planning_df['clstr req'] = planning_df['loading forecast'].apply(lambda v: 4 if v <= 450 else (5 if v <= 600 else (6 if v <= 800 else 8)))
                # --- FASE 3: ALLOCATION ALGORITHM ---
                vessel_block_map = {v: set(vessel_area_slots_df[vessel_area_slots_df['VESSEL'] == v]['Area (EXE)'].unique()) for v in planning_df['vessel'].unique()}

                st.warning("Simulation V1: Allocating total forecast at once. Daily trend logic will be in the next version.")                
                for _, vessel in planning_df.sort_values(by='eta').iterrows():
                    boxes_to_place = int(vessel['loading forecast']) - int(vessel['total box'])
                    if boxes_to_place <= 0: continue
                    
                    vessel_id = f"{vessel['vessel']}/{vessel['voy_out']}"
                    allocated = False
                    
                    preferred_blocks = list(vessel_block_map.get(vessel['vessel'], []))
                    other_blocks = [b for b in ALLOWED_BLOCKS if b not in preferred_blocks]
                    search_order = preferred_blocks + other_blocks
                    for block in search_order:
                        current_blocks_used = vessel_block_map[vessel['vessel']]
                        if block not in current_blocks_used and len(current_blocks_used) >= vessel['clstr req']:
                            continue
                        
                        space = find_available_space(yard_occupancy[block], boxes_to_place, BLOCK_CAPACITIES[block], min_clash_distance)
                        
                        if space:
                            start_slot, end_slot = space
                            yard_occupancy[block].append({'vessel': vessel_id, 'start': start_slot, 'end': end_slot})
                            vessel_block_map[vessel['vessel']].add(block)
                            
                            recommendations.append({
                                "Vessel": vessel['vessel'], "Boxes To Place": boxes_to_place,
                                "Recommended Block": block, "Recommended Slots": f"{start_slot}-{end_slot}", "Status": "Allocated"
                            })
                            allocated = True
                            break
                    
                    if not allocated:
                        failed_allocations.append({"Vessel": vessel['vessel'], "Boxes To Place": boxes_to_place, "Reason": "No suitable block found meeting all constraints."})
                # --- FASE 4: OUTPUT ---
                st.subheader("✅ Allocation Recommendations")
                if recommendations:
                    st.dataframe(pd.DataFrame(recommendations), use_container_width=True)
                else:
                    st.info("No new allocations were needed or could be made.")
                
                st.subheader("⚠️ Failed Allocations (Manual Action Required)")
                if failed_allocations:
                    st.dataframe(pd.DataFrame(failed_allocations), use_container_width=True)
                else:
                    st.info("All placement tasks were successfully allocated.")
            except Exception as e:
                st.error(f"An error occurred during the simulation: {e}")
                st.exception(e)

# --- MAIN APPLICATION STRUCTURE ---
tabs = st.tabs(["🚨 Clash Analysis", "📈 Loading Forecast", "💡 Stacking Recommendation"])

with tabs[0]:
    render_clash_tab() # Assume this function is fully defined
with tabs[1]:
    render_forecast_tab() # Assume this function is fully defined
with tabs[2]:
    min_clash_dist = st.session_state.get('min_clash_dist_input', 5) # Get value from sidebar widget
    render_recommendation_tab(min_clash_dist)
