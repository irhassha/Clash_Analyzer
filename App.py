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

# --- FUNCTIONS FOR FORECASTING ---
@st.cache_data
def load_history_data(filename="History Loading.xlsx"):
    """Finds and loads the historical data file for forecasting."""
    if os.path.exists(filename):
        try:
            if filename.lower().endswith('.csv'):
                df = pd.read_csv(filename)
            else:
                df = pd.read_excel(filename)
            
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
    """Creates time-based features from the 'ata' column."""
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
    """Runs the outlier cleaning and forecasting process for each service."""
    all_results = []
    unique_services = _df_history['service'].unique()
    progress_bar = st.progress(0, text="Analyzing services...")
    total_services = len(unique_services)
    for i, service in enumerate(unique_services):
        progress_text = f"Analyzing service: {service} ({i+1}/{total_services})"
        progress_bar.progress((i + 1) / total_services, text=progress_text)
        service_df = _df_history[_df_history['service'] == service].copy()
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
                if len(X_train) == 0:
                    raise ValueError("Not enough data to train the model.")
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
                forecast_val = service_df['loading_cleaned'].mean()
                moe_val = 1.96 * service_df['loading_cleaned'].std()
                actuals = service_df['loading_cleaned']
                mape_val = np.mean(np.abs((actuals - forecast_val) / actuals)) * 100 if not actuals.empty else 0
                method = f"Historical Average (RF Failed, {num_outliers} outliers cleaned)"
        else:
            forecast_val = service_df['loading_cleaned'].mean()
            moe_val = 1.96 * service_df['loading_cleaned'].std()
            actuals = service_df['loading_cleaned']
            mape_val = np.mean(np.abs((actuals - forecast_val) / actuals)) * 100 if not actuals.empty else 0
            method = f"Historical Average ({num_outliers} outliers cleaned)"
        all_results.append({
            "Service": service, "Loading Forecast": max(0, forecast_val),
            "Margin of Error (± box)": moe_val, "MAPE (%)": mape_val, "Method": method
        })
    progress_bar.empty()
    return pd.DataFrame(all_results)

def render_forecast_tab():
    """Function to display the entire content of the forecasting tab."""
    st.header("📈 Loading Forecast with Machine Learning")
    # ... (code for this tab remains unchanged)
    
@st.cache_data
def load_vessel_codes_from_repo(possible_names=['vessel codes.xlsx', 'vessel_codes.xls', 'vessel_codes.csv']):
    """Finds and loads the vessel codes file."""
    for filename in possible_names:
        if os.path.exists(filename):
            try:
                if filename.lower().endswith('.csv'): df = pd.read_csv(filename)
                else: df = pd.read_excel(filename)
                df.columns = df.columns.str.strip()
                return df
            except Exception as e:
                st.error(f"Failed to read file '{filename}': {e}"); return None
    st.error(f"Vessel codes file not found."); return None

def render_clash_tab():
    """Function to display the entire content of the Clash Analysis tab."""
    st.sidebar.header("⚙️ Upload Your Files")
    schedule_file = st.sidebar.file_uploader("1. Upload Vessel Schedule", type=['xlsx', 'csv'])
    unit_list_file = st.sidebar.file_uploader("2. Upload Unit List", type=['xlsx', 'csv'])
    min_clash_distance = st.sidebar.number_input("Minimum Safe Distance (slots)", min_value=0, value=5, step=1, help="Clash is detected if the distance between vessel allocations is less than or equal to this value.")
    
    process_button = st.sidebar.button("🚀 Process Data", use_container_width=True, type="primary")
    st.sidebar.button("Reset Data", on_click=reset_data, use_container_width=True, help="Clear all processed data and caches to start fresh.")

    for key in ['processed_df', 'summary_display', 'vessel_area_slots']:
        if key not in st.session_state:
            st.session_state[key] = None

    df_vessel_codes = load_vessel_codes_from_repo()
    if process_button:
        if schedule_file and unit_list_file and (df_vessel_codes is not None and not df_vessel_codes.empty):
            with st.spinner('Loading and processing data...'):
                try:
                    # --- Data Loading and Parsing ---
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

                    # --- Merging and Filtering ---
                    df_schedule_with_code = pd.merge(df_schedule, df_vessel_codes, left_on="VESSEL", right_on="Description", how="left").rename(columns={"Value": "CODE"})
                    merged_df = pd.merge(df_schedule_with_code, df_unit_list, left_on=['CODE', 'VOY_OUT'], right_on=['Carrier Out', 'Voyage Out'], how='inner')
                    if merged_df.empty: st.warning("No matching data found."); st.session_state.processed_df = None; st.stop()
                    
                    original_vessels_list = df_schedule['VESSEL'].unique().tolist()
                    merged_df = merged_df[merged_df['VESSEL'].isin(original_vessels_list)]
                    excluded_areas = [str(i) for i in range(801, 809)]
                    merged_df['Area (EXE)'] = merged_df['Area (EXE)'].astype(str)
                    filtered_data = merged_df[~merged_df['Area (EXE)'].isin(excluded_areas)]
                    if filtered_data.empty: st.warning("No data left after filtering."); st.session_state.processed_df = None; st.stop()
                    
                    # --- Data Aggregation for Display and Clash Logic ---
                    vessel_area_slots = filtered_data.groupby(['VESSEL', 'VOY_OUT', 'ETA', 'ETD', 'SERVICE', 'Area (EXE)']).agg(
                        MIN_SLOT=('SLOT', 'min'),
                        MAX_SLOT=('SLOT', 'max'),
                        BOX_COUNT=('SLOT', 'count')
                    ).reset_index()

                    pivot_for_display = vessel_area_slots.pivot_table(
                        index=['VESSEL', 'VOY_OUT', 'ETA', 'ETD', 'SERVICE'],
                        columns='Area (EXE)',
                        values='BOX_COUNT',
                        fill_value=0
                    )
                    
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
        
        # ... (Upcoming Vessel Summary section placeholder for brevity) ...

        # ... (Cluster Spreading Visualization section placeholder for brevity) ...

        # --- Clash Detection & Display Logic ---
        st.markdown("---")
        st.header("💥 Potential Clash Summary")
        
        clash_details = {}
        active_vessels = vessel_area_slots_df[['VESSEL', 'VOY_OUT', 'ETA', 'ETD', 'SERVICE']].drop_duplicates()
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
                        # --- Menentukan tanggal representatif untuk grouping ---
                        clash_date = max(vessel1['ETA'], vessel2['ETA']).normalize()
                        date_key = clash_date.strftime('%d/%m/%Y')
                        
                        if date_key not in clash_details:
                            clash_details[date_key] = []
                        
                        clash_info = {
                            "block": area,
                            "vessel1_name": vessel1['VESSEL'],
                            "vessel1_slots": f"{range1[0]}-{range1[1]}",
                            "vessel1_box": row['BOX_COUNT_v1'],
                            "vessel1_period": f"{vessel1['ETA'].strftime('%d/%m %H:%M')} - {vessel1['ETD'].strftime('%d/%m %H:%M')}",
                            "vessel2_name": vessel2['VESSEL'],
                            "vessel2_slots": f"{range2[0]}-{range2[1]}",
                            "vessel2_box": row['BOX_COUNT_v2'],
                            "vessel2_period": f"{vessel2['ETA'].strftime('%d/%m %H:%M')} - {vessel2['ETD'].strftime('%d/%m %H:%M')}",
                            "gap": gap
                        }

                        # Cek duplikasi sebelum menambahkan
                        is_duplicate = False
                        for existing_clash in clash_details[date_key]:
                            if (existing_clash['block'] == area and 
                                existing_clash['vessel1_name'] in [vessel1['VESSEL'], vessel2['VESSEL']] and
                                existing_clash['vessel2_name'] in [vessel1['VESSEL'], vessel2['VESSEL']]):
                                is_duplicate = True
                                break
                        if not is_duplicate:
                            clash_details[date_key].append(clash_info)
        
        # --- PERUBAHAN TAMPILAN CLASH SUMMARY MENJADI CARDS ---
        if not clash_details:
            st.info(f"No potential clashes found with a minimum distance of {min_clash_distance} slots.")
        else:
            total_clash_days = len(clash_details)
            st.markdown(f"**🔥 Found {total_clash_days} day(s) with potential clashes.**")
            clash_dates = sorted(clash_details.keys(), key=lambda x: datetime.strptime(x, '%d/%m/%Y'))
            cols = st.columns(len(clash_dates) or 1)
            for i, date_key in enumerate(clash_dates):
                with cols[i]:
                    clashes_on_day = clash_details.get(date_key, [])
                    summary_html = f"""
                    <div style="background-color: #F8F9FA; border: 1px solid #E9ECEF; border-radius: 10px; padding: 15px; margin-top: 1rem; height: 100%;">
                        <strong style='font-size: 1.2em;'>Potential Clash on: {date_key}</strong>
                        <hr style='margin: 10px 0;'>
                        <div style='line-height: 1.7;'>
                    """
                    for clash in clashes_on_day:
                        summary_html += f"""
                        <strong style="font-size: 1.1em;">Block {clash['block']}</strong> (Gap: {clash['gap']} slots)<br>
                        <ul>
                          <li style="margin-bottom: 8px;">
                            <strong>{clash['vessel1_name']}</strong> (<span style='color:#E67E22; font-weight:bold;'>{clash['vessel1_box']} boxes</span>)
                            <br><small><i>Slots: {clash['vessel1_slots']} | Period: {clash['vessel1_period']}</i></small>
                          </li>
                          <li>
                            <strong>{clash['vessel2_name']}</strong> (<span style='color:#E67E22; font-weight:bold;'>{clash['vessel2_box']} boxes</span>)
                            <br><small><i>Slots: {clash['vessel2_slots']} | Period: {clash['vessel2_period']}</i></small>
                          </li>
                        </ul>
                        """
                    summary_html += "</div></div>"
                    st.markdown(summary_html, unsafe_allow_html=True)
        # --- AKHIR PERUBAHAN TAMPILAN ---

        st.markdown("---")
        st.header("📋 Detailed Analysis Results")
        st.dataframe(display_df) # Placeholder display
    else:
        st.info("Welcome! Please upload your files and click 'Process Data' to begin.")

# --- MAIN STRUCTURE WITH TABS ---
tab1, tab2 = st.tabs(["🚨 Clash Analysis", "📈 Loading Forecast"])
with tab1:
    render_clash_tab()
with tab2:
    render_forecast_tab()
