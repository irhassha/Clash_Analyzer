import streamlit as st
import pandas as pd
import os
from datetime import datetime, timedelta
import json
import io
import warnings
import numpy as np
import plotly.express as px

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

# --- FUNCTIONS FOR FORECASTING (NEW MODEL: PER-SERVICE RF) ---
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
    st.write("This feature uses a separate **Random Forest** model for each service. The model learns from historical time patterns to provide more accurate predictions, complete with anomaly data cleaning.")
    if 'forecast_df' not in st.session_state:
        df_history = load_history_data()
        if df_history is not None and not df_history.empty:
            with st.spinner("Processing data and training models for each service..."):
                forecast_df = run_per_service_rf_forecast(df_history)
                st.session_state.forecast_df = forecast_df
        else:
            st.session_state.forecast_df = pd.DataFrame()
            if df_history is None:
                st.error("Could not load historical data. Process canceled.")
    
    if 'forecast_df' in st.session_state and not st.session_state.forecast_df.empty:
        results_df = st.session_state.forecast_df
        results_df['Loading Forecast'] = results_df['Loading Forecast'].round(2)
        results_df['Margin of Error (± box)'] = results_df['Margin of Error (± box)'].fillna(0).round(2)
        results_df['MAPE (%)'] = results_df['MAPE (%)'].replace([np.inf, -np.inf], 0).fillna(0).round(2)
        st.markdown("---")
        st.subheader("📊 Forecast Results per Service")
        st.dataframe(
            results_df.sort_values(by="Loading Forecast", ascending=False).reset_index(drop=True),
            use_container_width=True, hide_index=True,
            column_config={"MAPE (%)": st.column_config.NumberColumn(format="%.2f%%")}
        )
        st.markdown("---")
        st.subheader("💡 How to Read These Results")
        st.markdown("- **Loading Forecast**: The estimated number of boxes for the next vessel arrival of that service.\n- **Margin of Error (± box)**: The level of uncertainty in the prediction. A prediction of **300** with a MoE of **±50** means the actual value is likely between **250** and **350**.\n- **MAPE (%)**: The average percentage error of the model when tested on its historical data. **The smaller the value, the more accurate the model has been in the past.**\n- **Method**: The technique used for the forecast and the number of outliers handled.")
    else:
        st.warning("No forecast data could be generated. The history file might be empty or contain no valid service data.")

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
    process_button = st.sidebar.button("🚀 Process Clash Data", type="primary")

    if 'processed_df' not in st.session_state:
        st.session_state.processed_df = None
    if 'clash_summary_df' not in st.session_state:
        st.session_state.clash_summary_df = None
    if 'summary_display' not in st.session_state:
        st.session_state.summary_display = None

    df_vessel_codes = load_vessel_codes_from_repo()
    if process_button:
        if schedule_file and unit_list_file and (df_vessel_codes is not None and not df_vessel_codes.empty):
            with st.spinner('Loading and processing data...'):
                try:
                    if schedule_file.name.lower().endswith(('.xls', '.xlsx')): df_schedule = pd.read_excel(schedule_file)
                    else: df_schedule = pd.read_csv(schedule_file)
                    df_schedule.columns = [col.strip().upper() for col in df_schedule.columns]
                    if unit_list_file.name.lower().endswith(('.xls', '.xlsx')): df_unit_list = pd.read_excel(unit_list_file)
                    else: df_unit_list = pd.read_csv(unit_list_file)
                    df_unit_list.columns = [col.strip() for col in df_unit_list.columns]
                    original_vessels_list = df_schedule['VESSEL'].unique().tolist()
                    df_schedule['ETA'] = pd.to_datetime(df_schedule['ETA'], errors='coerce')
                    df_schedule['CLOSING PHYSIC'] = pd.to_datetime(df_schedule['CLOSING PHYSIC'], errors='coerce')
                    df_schedule_with_code = pd.merge(df_schedule, df_vessel_codes, left_on="VESSEL", right_on="Description", how="left").rename(columns={"Value": "CODE"})
                    merged_df = pd.merge(df_schedule_with_code, df_unit_list, left_on=['CODE', 'VOY_OUT'], right_on=['Carrier Out', 'Voyage Out'], how='inner')
                    if merged_df.empty: st.warning("No matching data found."); st.session_state.processed_df = None; st.stop()
                    merged_df = merged_df[merged_df['VESSEL'].isin(original_vessels_list)]
                    excluded_areas = [str(i) for i in range(801, 809)]
                    merged_df['Area (EXE)'] = merged_df['Area (EXE)'].astype(str)
                    filtered_data = merged_df[~merged_df['Area (EXE)'].isin(excluded_areas)]

                    if filtered_data.empty: st.warning("No data left after filtering."); st.session_state.processed_df = None; st.stop()

                    grouping_cols = ['VESSEL', 'CODE', 'SERVICE', 'VOY_OUT', 'ETA', 'CLOSING PHYSIC']
                    pivot_df = filtered_data.pivot_table(index=grouping_cols, columns='Area (EXE)', aggfunc='size', fill_value=0)
                    cluster_cols_for_calc = pivot_df.columns.tolist()
                    pivot_df['TOTAL BOX'] = pivot_df[cluster_cols_for_calc].sum(axis=1)
                    exclude_for_clstr = ['D01', 'C01', 'C02', 'OOG', 'UNKNOWN', 'BR9', 'RC9']
                    clstr_calculation_cols = [col for col in cluster_cols_for_calc if col not in exclude_for_clstr]
                    pivot_df['TOTAL CLSTR'] = (pivot_df[clstr_calculation_cols] > 0).sum(axis=1)
                    pivot_df = pivot_df.reset_index()
                    two_days_ago = pd.Timestamp.now() - timedelta(days=2)
                    condition_to_hide = (pivot_df['ETA'] < two_days_ago) & (pivot_df['TOTAL BOX'] < 50)
                    pivot_df = pivot_df[~condition_to_hide]
                    if pivot_df.empty: st.warning("No data left after ETA & Total filter."); st.session_state.processed_df = None; st.stop()

                    initial_cols = ['VESSEL', 'CODE', 'SERVICE', 'VOY_OUT', 'ETA', 'CLOSING PHYSIC', 'TOTAL BOX', 'TOTAL CLSTR']
                    final_cluster_cols = [col for col in pivot_df.columns if col not in initial_cols]
                    final_display_cols = initial_cols + sorted(final_cluster_cols)
                    pivot_df = pivot_df[final_display_cols]
                    pivot_df['ETA_str'] = pd.to_datetime(pivot_df['ETA']).dt.strftime('%Y-%m-%d %H:%M')
                    pivot_df['CLOSING_PHYSIC_str'] = pd.to_datetime(pivot_df['CLOSING PHYSIC']).dt.strftime('%Y-%m-%d %H:%M')

                    pivot_df = pivot_df.sort_values(by='ETA', ascending=True).reset_index(drop=True)
                    st.session_state.processed_df = pivot_df
                    st.success("Data processed successfully!")
                except Exception as e:
                    st.error(f"An error occurred during processing: {e}")
                    st.session_state.processed_df = None
        else:
            st.warning("Please upload both files.")

    if st.session_state.get('processed_df') is not None:
        display_df = st.session_state.processed_df
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
                forecast_lookup = forecast_df[['Service', 'Loading Forecast']].copy()
                summary_df = pd.merge(upcoming_vessels_df, forecast_lookup, left_on='SERVICE', right_on='Service', how='left')
                summary_df['Loading Forecast'] = summary_df['Loading Forecast'].fillna(0).round(0).astype(int)
                summary_df['DIFF'] = summary_df['TOTAL BOX'] - summary_df['Loading Forecast']
                summary_df['base_for_req'] = summary_df[['TOTAL BOX', 'Loading Forecast']].max(axis=1)
                def get_clstr_requirement(value):
                    if value <= 450: return 4
                    elif 451 <= value <= 600: return 5
                    elif 601 <= value <= 800: return 6
                    else: return 8
                summary_df['CLSTR REQ'] = summary_df['base_for_req'].apply(get_clstr_requirement)
                if priority_vessels and adjusted_clstr_req > 0:
                    summary_df.loc[summary_df['VESSEL'].isin(priority_vessels), 'CLSTR REQ'] = adjusted_clstr_req

                summary_display_cols = ['VESSEL', 'SERVICE', 'ETA', 'CLOSING_PHYSIC_str', 'TOTAL BOX', 'Loading Forecast', 'DIFF', 'TOTAL CLSTR', 'CLSTR REQ']
                visible_cols = summary_display_cols
                summary_display = summary_df[visible_cols].rename(columns={
                    'ETA': 'ETA',
                    'CLOSING_PHYSIC_str': 'CLOSING TIME',
                    'TOTAL BOX': 'BOX STACKED',
                    'Loading Forecast': 'LOADING FORECAST'
                })
                st.session_state.summary_display = summary_display

                def style_diff(v):
                    color = '#4CAF50' if v > 0 else '#F44336' if v < 0 else '#757575'
                    return f'color: {color}; font-weight: bold;'

                def highlight_rows(row):
                    if row['TOTAL CLSTR'] < row['CLSTR REQ']:
                        return ['background-color: #FFCDD2'] * len(row)
                    if row['VESSEL'] in priority_vessels:
                        return ['background-color: #FFF3CD'] * len(row)
                    return [''] * len(row)

                styled_df = summary_display.style.apply(highlight_rows, axis=1).map(style_diff, subset=['DIFF']).format({'ETA': '{:%Y-%m-%d %H:%M}'})
                st.dataframe(styled_df, use_container_width=False, hide_index=True)
            else:
                st.info("No vessels scheduled to arrive in the next 4 days.")
        else:
            st.warning("Forecast data is not available. Please run the forecast in the 'Loading Forecast' tab first.")

        st.markdown("---")
        st.subheader("📊 Cluster Spreading Visualization")
        st.write("This chart shows the box distribution across various clusters for each vessel. Hover on bars for details and click on the legend to toggle clusters.")

        all_vessels_list = display_df['VESSEL'].unique().tolist()
        st.sidebar.markdown("---")
        st.sidebar.header("📊 Chart Options")
        selected_vessels = st.sidebar.multiselect(
            "Filter Vessels on Chart:",
            options=all_vessels_list,
            default=all_vessels_list
        )
        
        font_size = st.sidebar.slider("Adjust Chart Font Size", min_value=6, max_value=20, value=10, step=1)

        if not selected_vessels:
            st.warning("Please select at least one vessel from the sidebar to display the chart.")
        elif st.session_state.get('processed_df') is not None and not st.session_state.processed_df.empty:
            processed_df = display_df[display_df['VESSEL'].isin(selected_vessels)]
            
            initial_cols = ['VESSEL', 'CODE', 'SERVICE', 'VOY_OUT', 'ETA', 'CLOSING PHYSIC', 'TOTAL BOX', 'TOTAL CLSTR', 'ETA_str', 'CLOSING_PHYSIC_str']
            exclude_from_chart = ['BR9', 'RC9', 'D01', 'C01', 'C02']
            cluster_cols = sorted([
                col for col in processed_df.columns 
                if col not in initial_cols and col not in exclude_from_chart
            ])

            chart_data_long = pd.melt(processed_df, id_vars=['VESSEL'], value_vars=cluster_cols, var_name='Cluster', value_name='Box Count')
            chart_data_long = chart_data_long[chart_data_long['Box Count'] > 0]

            if chart_data_long.empty:
                st.info("No cluster data to visualize for the selected vessels (after exclusions).")
            else:
                # --- PERUBAHAN DI SINI: Buat kolom teks gabungan ---
                chart_data_long['combined_text'] = chart_data_long['Cluster'] + ' / ' + chart_data_long['Box Count'].astype(str)
                # --- AKHIR PERUBAHAN ---
                
                cluster_color_map = {
                    'A01': '#5409DA', 'A02': '#4E71FF', 'A03': '#8DD8FF', 'A04': '#BBFBFF', 'A05': '#8DBCC7',
                    'B01': '#328E6E', 'B02': '#67AE6E', 'B03': '#90C67C', 'B04': '#E1EEBC', 'B05': '#E7EFC7',
                    'C03': '#B33791', 'C04': '#C562AF', 'C05': '#DB8DD0',
                    'E11': '#8D493A', 'E12': '#D0B8A8', 'E13': '#DFD3C3', 'E14': '#F8EDE3',
                    'EA09':'#EECEB9', 'OOG': 'black'
                }
                
                vessel_order_by_eta = processed_df['VESSEL'].tolist()
                
                fig = px.bar(
                    chart_data_long,
                    x='Box Count',
                    y='VESSEL',
                    color='Cluster',
                    color_discrete_map=cluster_color_map,
                    orientation='h',
                    title='Box Distribution per Cluster for Each Vessel',
                    # --- PERUBAHAN DI SINI: Gunakan kolom combined_text ---
                    text='combined_text',
                    hover_data={'VESSEL': False, 'Cluster': True, 'Box Count': True}
                )

                fig.update_layout(
                    xaxis_title=None,
                    yaxis_title=None,
                    height=len(processed_df['VESSEL'].unique()) * 35 + 150,
                    legend_title_text='Cluster Area',
                    title_x=0
                )
                fig.update_yaxes(categoryorder='array', categoryarray=vessel_order_by_eta[::-1])
                fig.update_traces(textposition='inside', textfont_size=font_size, textangle=0)
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Process data first to see the visualization.")

        st.markdown("---")
        st.header("📋 Detailed Analysis Results")
        df_for_grid = display_df.copy()
        df_for_grid['ETA_Date'] = pd.to_datetime(df_for_grid['ETA']).dt.strftime('%Y-%m-%d')
        df_for_grid['ETA'] = df_for_grid['ETA_str']
        unique_dates = df_for_grid['ETA_Date'].unique()
        date_color_map = {date: ['#F8F0E5', '#DAC0A3'][i % 2] for i, date in enumerate(unique_dates)}
        clash_map = {}
        cluster_cols_aggrid = [col for col in df_for_grid.columns if col not in ['VESSEL', 'CODE', 'SERVICE', 'VOY_OUT', 'ETA', 'CLOSING PHYSIC', 'TOTAL BOX', 'TOTAL CLSTR', 'ETA_Date', 'ETA_str', 'CLOSING_PHYSIC_str']]
        for date, group in df_for_grid.groupby('ETA_Date'):
            clash_areas_for_date = []
            for col in cluster_cols_aggrid:
                if (group[col] > 0).sum() > 1:
                    clash_areas_for_date.append(col)
            if clash_areas_for_date:
                clash_map[date] = clash_areas_for_date
        
        if clash_map:
            summary_data = []
            with st.expander("Show Clash Summary", expanded=True):
                total_clash_days = len(clash_map)
                total_conflicting_blocks = sum(len(areas) for areas in clash_map.values())
                st.markdown(f"**🔥 Found {total_clash_days} clash days with a total of {total_conflicting_blocks} conflicting blocks.**")
                clash_dates = sorted(clash_map.keys())
                cols = st.columns(len(clash_dates) or 1)
                for i, date in enumerate(clash_dates):
                    with cols[i]:
                        areas = clash_map.get(date, [])
                        summary_exclude_blocks = ['BR9', 'RC9', 'C01', 'D01', 'OOG']
                        filtered_areas = [area for area in areas if area not in summary_exclude_blocks]
                        if not filtered_areas:
                            continue
                        summary_html = f"""<div style="background-color: #F8F9FA; border: 1px solid #E9ECEF; border-radius: 10px; padding: 15px; margin-top: 1rem; height: 100%;"><strong style='font-size: 1.2em;'>Clash on: {date}</strong><hr style='margin: 10px 0;'><div style='line-height: 1.7;'>"""
                        for area in sorted(filtered_areas):
                            clashing_rows = df_for_grid[(df_for_grid['ETA_Date'] == date) & (df_for_grid[area] > 0)]
                            clashing_vessels = clashing_rows['VESSEL'].tolist()
                            total_clash_boxes = clashing_rows[area].sum()
                            vessel_list_str = ", ".join(clashing_vessels)
                            summary_html += f"<b>Block {area}</b> (<span style='color:#E67E22; font-weight:bold;'>{total_clash_boxes} boxes</span>):<br><small>{vessel_list_str}</small><br>"
                            summary_data.append({"Clash Date": date, "Block": area, "Total Boxes": total_clash_boxes, "Vessel(s)": vessel_list_str, "Notes": ""})
                        summary_html += "</div></div>"
                        st.markdown(summary_html, unsafe_allow_html=True)
                st.session_state.clash_summary_df = pd.DataFrame(summary_data)
        
        hide_zero_jscode = JsCode("""function(params) { if (params.value == 0 || params.value === null) { return ''; } return params.value; }""")
        clash_cell_style_jscode = JsCode(f"""function(params) {{ const clashMap = {json.dumps(clash_map)}; const date = params.data.ETA_Date; const colId = params.colDef.field; const isClash = clashMap[date] ? clashMap[date].includes(colId) : false; if (isClash) {{ return {{'backgroundColor': '#FFAA33', 'color': 'black'}}; }} return null; }}""")
        zebra_row_style_jscode = JsCode(f"""function(params) {{ const dateColorMap = {json.dumps(date_color_map)}; const date = params.data.ETA_Date; const color = dateColorMap[date]; return {{ 'background-color': color }}; }}""")
        default_col_def = {"suppressMenu": True, "sortable": True, "resizable": True, "editable": False, "minWidth": 40}
        column_defs = []
        pinned_cols = ['VESSEL', 'CODE', 'SERVICE', 'VOY_OUT', 'ETA', 'TOTAL BOX', 'TOTAL CLSTR']
        for col in pinned_cols:
            width = 110 if col == 'VESSEL' else 80
            if col == 'ETA': width = 120
            if col == 'SERVICE': width = 90
            col_def = {"field": col, "headerName": col, "pinned": "left", "width": width}
            if col in ["TOTAL BOX", "TOTAL CLSTR"]: col_def["cellRenderer"] = hide_zero_jscode
            column_defs.append(col_def)
        for col in cluster_cols_aggrid:
            column_defs.append({"field": col, "headerName": col, "width": 60, "cellRenderer": hide_zero_jscode, "cellStyle": clash_cell_style_jscode})
        column_defs.append({"field": "ETA_Date", "hide": True})
        gridOptions = {"defaultColDef": default_col_def, "columnDefs": column_defs, "getRowStyle": zebra_row_style_jscode}
        AgGrid(df_for_grid.drop(columns=['ETA_str', 'CLOSING_PHYSIC_str', 'CLOSING PHYSIC']), gridOptions=gridOptions, height=600, width='100%', theme='streamlit', allow_unsafe_jscode=True)

        st.markdown("---")
        st.subheader("📥 Download Center")
        
        try:
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                workbook = writer.book
                center_format = workbook.add_format({'align': 'center', 'valign': 'vcenter'})

                def auto_adjust_and_format_sheet(df, sheet_name, writer_obj):
                    if df is not None and not df.empty:
                        df_to_write = df.copy()
                        if 'ETA' in df_to_write.columns and pd.api.types.is_datetime64_any_dtype(df_to_write['ETA']):
                            df_to_write['ETA'] = pd.to_datetime(df_to_write['ETA']).dt.strftime('%Y-%m-%d %H:%M')
                        df_to_write.to_excel(writer_obj, sheet_name=sheet_name, index=False)
                        worksheet = writer_obj.sheets[sheet_name]
                        for idx, col_name in enumerate(df_to_write.columns):
                            series = df_to_write[col_name].dropna()
                            max_len = max(
                                ([len(str(s)) for s in series] if not series.empty else [0]) + [len(str(col_name))]
                            ) + 5
                            max_len = min(max_len, 50)
                            worksheet.set_column(idx, idx, max_len, center_format)
                
                auto_adjust_and_format_sheet(st.session_state.get('processed_df'), 'Detailed Analysis', writer)
                auto_adjust_and_format_sheet(st.session_state.get('summary_display'), 'Upcoming Vessel Summary', writer)
                
                if st.session_state.get('clash_summary_df') is not None and not st.session_state.clash_summary_df.empty:
                    summary_df_for_export = st.session_state.clash_summary_df
                    final_summary_export = []
                    last_date = None
                    for index, row in summary_df_for_export.iterrows():
                        current_date = row['Clash Date']
                        if last_date is not None and current_date != last_date:
                            final_summary_export.append({})
                        final_summary_export.append(row.to_dict())
                        last_date = current_date
                    final_summary_df = pd.DataFrame(final_summary_export)
                    auto_adjust_and_format_sheet(final_summary_df, 'Clash Summary', writer)

            if output.tell() > 0:
                st.download_button(
                    label="📥 Download All Analysis Tables (Excel)",
                    data=output.getvalue(),
                    file_name="clash_analysis_report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            else:
                st.info("No data available to download. Please process the data first.")
        except Exception as e:
            st.error(f"Failed to create download file: {e}")

# --- MAIN STRUCTURE WITH TABS ---
tab1, tab2 = st.tabs(["🚨 Clash Analysis", "📈 Loading Forecast"])
with tab1:
    render_clash_tab()
with tab2:
    render_forecast_tab()
