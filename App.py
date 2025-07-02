import streamlit as st
import pandas as pd
import os
from datetime import datetime, timedelta
import json
import io
import warnings
import numpy as np
import matplotlib.pyplot as plt

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
def run_per_service_rf_forecast(df_history):
    """Runs the outlier cleaning and forecasting process for each service."""
    all_results = []
    unique_services = df_history['service'].unique()

    progress_bar = st.progress(0, text="Analyzing services...")
    total_services = len(unique_services)

    for i, service in enumerate(unique_services):
        progress_text = f"Analyzing service: {service} ({i+1}/{total_services})"
        progress_bar.progress((i + 1) / total_services, text=progress_text)

        service_df = df_history[df_history['service'] == service].copy()
        
        if service_df.empty or service_df['loading'].isnull().all():
            continue

        # --- 1. Outlier Cleaning ---
        Q1 = service_df['loading'].quantile(0.25)
        Q3 = service_df['loading'].quantile(0.75)
        IQR = Q3 - Q1
        upper_bound = Q3 + 1.5 * IQR
        lower_bound = Q1 - 1.5 * IQR
        num_outliers = ((service_df['loading'] < lower_bound) | (service_df['loading'] > upper_bound)).sum()
        service_df['loading_cleaned'] = service_df['loading'].clip(lower=lower_bound, upper=upper_bound)

        # --- 2. Select Model & Forecast ---
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
            "Service": service,
            "Loading Forecast": max(0, forecast_val),
            "Margin of Error (± box)": moe_val,
            "MAPE (%)": mape_val,
            "Method": method
        })
        
    progress_bar.empty()
    return pd.DataFrame(all_results)

def render_forecast_tab():
    """Function to display the entire content of the forecasting tab."""
    st.header("📈 Loading Forecast with Machine Learning")
    st.write("""
    This feature uses a separate **Random Forest** model for each service. 
    The model learns from historical time patterns to provide more accurate predictions, complete with anomaly data cleaning.
    """)

    if 'forecast_df' not in st.session_state:
        df_history = load_history_data()
        if df_history is not None:
            with st.spinner("Processing data and training models for each service..."):
                forecast_df = run_per_service_rf_forecast(df_history)
                st.session_state.forecast_df = forecast_df
        else:
            st.session_state.forecast_df = pd.DataFrame()
            st.error("Could not load historical data. Process canceled.")
    
    if 'forecast_df' in st.session_state:
        results_df = st.session_state.forecast_df
        
        if not results_df.empty:
            results_df['Loading Forecast'] = results_df['Loading Forecast'].round(2)
            results_df['Margin of Error (± box)'] = results_df['Margin of Error (± box)'].fillna(0).round(2)
            results_df['MAPE (%)'] = results_df['MAPE (%)'].replace([np.inf, -np.inf], 0).fillna(0).round(2)

            st.markdown("---")
            st.subheader("📊 Forecast Results per Service")
            st.dataframe(
                results_df.sort_values(by="Loading Forecast", ascending=False).reset_index(drop=True),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "MAPE (%)": st.column_config.NumberColumn(format="%.2f%%")
                }
            )
            
            st.markdown("---")
            st.subheader("💡 How to Read These Results")
            st.markdown("""
            - **Loading Forecast**: The estimated number of boxes for the next vessel arrival of that service.
            - **Margin of Error (± box)**: The level of uncertainty in the prediction. A prediction of **300** with a MoE of **±50** means the actual value is likely between **250** and **350**.
            - **MAPE (%)**: The average percentage error of the model when tested on its historical data. **The smaller the value, the more accurate the model has been in the past.**
            - **Method**: The technique used for the forecast and the number of outliers handled.
            """)
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

    df_vessel_codes = load_vessel_codes_from_repo()

    if process_button:
        if schedule_file and unit_list_file and (df_vessel_codes is not None and not df_vessel_codes.empty):
            with st.spinner('Loading and processing data...'):
                try:
                    # 1. Loading & Cleaning
                    if schedule_file.name.lower().endswith(('.xls', '.xlsx')): df_schedule = pd.read_excel(schedule_file)
                    else: df_schedule = pd.read_csv(schedule_file)
                    df_schedule.columns = [col.strip().upper() for col in df_schedule.columns] # Consistent to uppercase
                    
                    if unit_list_file.name.lower().endswith(('.xls', '.xlsx')): df_unit_list = pd.read_excel(unit_list_file)
                    else: df_unit_list = pd.read_csv(unit_list_file)
                    df_unit_list.columns = [col.strip() for col in df_unit_list.columns]
                    
                    # 2. Main Processing
                    original_vessels_list = df_schedule['VESSEL'].unique().tolist()
                    df_schedule['ETA'] = pd.to_datetime(df_schedule['ETA'], errors='coerce')
                    df_schedule_with_code = pd.merge(df_schedule, df_vessel_codes, left_on="VESSEL", right_on="Description", how="left").rename(columns={"Value": "CODE"})
                    merged_df = pd.merge(df_schedule_with_code, df_unit_list, left_on=['CODE', 'VOY_OUT'], right_on=['Carrier Out', 'Voyage Out'], how='inner')
                    if merged_df.empty: st.warning("No matching data found."); st.session_state.processed_df = None; st.stop()
                    
                    # 3. Filtering
                    merged_df = merged_df[merged_df['VESSEL'].isin(original_vessels_list)]
                    excluded_areas = [str(i) for i in range(801, 809)]
                    merged_df['Area (EXE)'] = merged_df['Area (EXE)'].astype(str)
                    filtered_data = merged_df[~merged_df['Area (EXE)'].isin(excluded_areas)]
                    if filtered_data.empty: st.warning("No data left after filtering."); st.session_state.processed_df = None; st.stop()

                    # 4. Pivoting
                    grouping_cols = ['VESSEL', 'CODE', 'SERVICE', 'VOY_OUT', 'ETA']
                    pivot_df = filtered_data.pivot_table(index=grouping_cols, columns='Area (EXE)', aggfunc='size', fill_value=0)
                    
                    cluster_cols_for_calc = pivot_df.columns.tolist()
                    pivot_df['TOTAL BOX'] = pivot_df[cluster_cols_for_calc].sum(axis=1)
                    
                    exclude_for_clstr = ['D01', 'C01', 'C02', 'OOG', 'UNKNOWN', 'BR9', 'RC9']
                    clstr_calculation_cols = [col for col in cluster_cols_for_calc if col not in exclude_for_clstr]
                    pivot_df['TOTAL CLSTR'] = (pivot_df[clstr_calculation_cols] > 0).sum(axis=1)
                    
                    pivot_df = pivot_df.reset_index()
                    
                    # 5. Conditional Filtering
                    two_days_ago = pd.Timestamp.now() - timedelta(days=2)
                    condition_to_hide = (pivot_df['ETA'] < two_days_ago) & (pivot_df['TOTAL BOX'] < 50)
                    pivot_df = pivot_df[~condition_to_hide]
                    if pivot_df.empty: st.warning("No data left after ETA & Total filter."); st.session_state.processed_df = None; st.stop()

                    # 6. Sorting and Ordering
                    initial_cols = ['VESSEL', 'CODE', 'SERVICE', 'VOY_OUT', 'ETA', 'TOTAL BOX', 'TOTAL CLSTR']
                    final_cluster_cols = [col for col in pivot_df.columns if col not in initial_cols]
                    final_display_cols = initial_cols + sorted(final_cluster_cols)
                    pivot_df = pivot_df[final_display_cols]
                    
                    pivot_df['ETA_str'] = pd.to_datetime(pivot_df['ETA']).dt.strftime('%Y-%m-%d %H:%M')
                    
                    pivot_df = pivot_df.sort_values(by='ETA', ascending=True).reset_index(drop=True)
                    
                    st.session_state.processed_df = pivot_df
                    st.success("Data processed successfully!")

                except Exception as e:
                    st.error(f"An error occurred during processing: {e}")
                    st.session_state.processed_df = None
        else:
            st.warning("Please upload both files.")

    # --- Display Area ---
    if st.session_state.get('processed_df') is not None:
        display_df = st.session_state.processed_df
        
        # --- NEW SUMMARY: UPCOMING VESSELS & FORECAST ---
        st.subheader("🚢 Upcoming Vessel Summary (Today + Next 3 Days)")
        
        forecast_df = st.session_state.get('forecast_df')
        if forecast_df is not None and not forecast_df.empty:
            today = pd.to_datetime(datetime.now().date())
            four_days_later = today + timedelta(days=4)
            
            upcoming_vessels_df = display_df[
                (display_df['ETA'] >= today) & 
                (display_df['ETA'] < four_days_later)
            ].copy()

            if not upcoming_vessels_df.empty:
                # --- NEW SIDEBAR OPTIONS ---
                st.sidebar.markdown("---")
                st.sidebar.header("🛠️ Upcoming Vessel Options")
                
                all_summary_cols = ['VESSEL', 'SERVICE', 'ETA', 'TOTAL BOX', 'Loading Forecast', 'DIFF', 'TOTAL CLSTR', 'CLSTR REQ']
                
                cols_to_hide = st.sidebar.multiselect(
                    "Hide columns from summary:",
                    options=all_summary_cols,
                    default=[]
                )
                
                priority_vessels = st.sidebar.multiselect(
                    "Select priority vessels to highlight:",
                    options=upcoming_vessels_df['VESSEL'].unique()
                )
                
                adjusted_clstr_req = st.sidebar.number_input(
                    "Adjust CLSTR REQ for priority vessels:",
                    min_value=0,
                    value=0,
                    step=1,
                    help="Enter a new value for CLSTR REQ. Leave as 0 to not change."
                )

                forecast_lookup = forecast_df[['Service', 'Loading Forecast']].copy()
                
                summary_df = pd.merge(
                    upcoming_vessels_df,
                    forecast_lookup,
                    left_on='SERVICE',
                    right_on='Service',
                    how='left'
                )
                
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
                
                summary_display_cols = [
                    'VESSEL', 'SERVICE', 'ETA_str', 'TOTAL BOX', 
                    'Loading Forecast', 'DIFF', 'TOTAL CLSTR', 'CLSTR REQ'
                ]
                
                visible_cols = [col for col in summary_display_cols if col not in cols_to_hide]
                summary_display = summary_df[visible_cols].rename(columns={'ETA_str': 'ETA'})
                
                # --- START: STYLING LOGIC ---
                def style_DIFF(v):
                    color = '#4CAF50' if v > 0 else '#F44336' if v < 0 else '#757575' # Green, Red, Gray
                    return f'color: {color}; font-weight: bold;'

                def highlight_rows(row):
                    # Priority 1: Cluster issue (light red)
                    if row['TOTAL CLSTR'] < row['CLSTR REQ']:
                        return ['background-color: #FFCDD2'] * len(row)
                    # Priority 2: User-selected priority vessel (light yellow)
                    if row.VESSEL in priority_vessels:
                        return ['background-color: #FFF3CD'] * len(row)
                    # Default: No highlight
                    return [''] * len(row)

                # Apply styles
                styled_df = summary_display.style \
                    .apply(highlight_rows, axis=1) \
                    .map(style_DIFF, subset=['DIFF'])

                # Display the styled DataFrame (without bar chart column_config)
                st.dataframe(
                    styled_df,
                    use_container_width=False,
                    hide_index=True,
                )
                # --- END: STYLING LOGIC ---
                
            else:
                st.info("No vessels scheduled to arrive in the next 4 days.")
        else:
            st.warning("Forecast data is not available. Please run the forecast in the 'Loading Forecast' tab first.")

        st.markdown("---")
        st.header("📋 Detailed Analysis Results")

        # --- Preparation for AG Grid Styling and Summary ---
        df_for_grid = display_df.copy()
        df_for_grid['ETA_Date'] = pd.to_datetime(df_for_grid['ETA']).dt.strftime('%Y-%m-%d')
        df_for_grid['ETA'] = df_for_grid['ETA_str']
        
        unique_dates = df_for_grid['ETA_Date'].unique()
        zebra_colors = ['#F8F0E5', '#DAC0A3'] 
        date_color_map = {date: zebra_colors[i % 2] for i, date in enumerate(unique_dates)}

        clash_map = {}
        cluster_cols = [col for col in df_for_grid.columns if col not in ['VESSEL', 'CODE', 'SERVICE', 'VOY_OUT', 'ETA', 'TOTAL BOX', 'TOTAL CLSTR', 'ETA_Date', 'ETA_str']]
        for date, group in df_for_grid.groupby('ETA_Date'):
            clash_areas_for_date = []
            for col in cluster_cols:
                if (group[col] > 0).sum() > 1:
                    clash_areas_for_date.append(col)
            if clash_areas_for_date:
                clash_map[date] = clash_areas_for_date

        # --- CLASH SUMMARY DISPLAY WITH CARDS ---
        summary_data = []
        if clash_map:
            summary_exclude_blocks = ['BR9', 'RC9', 'C01', 'D01', 'OOG']

            with st.expander("Show Clash Summary", expanded=True):
                total_clash_days = len(clash_map)
                total_conflicting_blocks = sum(len(areas) for areas in clash_map.values())
                st.markdown(f"**🔥 Found {total_clash_days} clash days with a total of {total_conflicting_blocks} conflicting blocks.**")
                
                clash_dates = sorted(clash_map.keys())
                cols = st.columns(len(clash_dates) or 1)

                for i, date in enumerate(clash_dates):
                    with cols[i]:
                        areas = clash_map[date]
                        filtered_areas = [area for area in areas if area not in summary_exclude_blocks]
                        if not filtered_areas:
                            continue
                        
                        summary_html = f"""
                        <div style="background-color: #F8F9FA; border: 1px solid #E9ECEF; border-radius: 10px; padding: 15px; height: 100%;">
                            <strong style='font-size: 1.2em;'>Clash on: {date}</strong>
                            <hr style='margin: 10px 0;'>
                            <div style='line-height: 1.7;'>
                        """
                        for area in sorted(filtered_areas):
                            clashing_rows = df_for_grid[(df_for_grid['ETA_Date'] == date) & (df_for_grid[area] > 0)]
                            clashing_vessels = clashing_rows['VESSEL'].tolist()
                            total_clash_boxes = clashing_rows[area].sum()
                            vessel_list_str = ", ".join(clashing_vessels)
                            
                            summary_html += f"<b>Block {area}</b> (<span style='color:#E67E22; font-weight:bold;'>{total_clash_boxes} boxes</span>):<br><small>{vessel_list_str}</small><br>"
                            
                            summary_data.append({
                                "Clash Date": date,
                                "Block": area,
                                "Total Boxes": total_clash_boxes,
                                "Vessel(s)": vessel_list_str,
                                "Notes": ""
                            })
                        summary_html += "</div></div>"
                        st.markdown(summary_html, unsafe_allow_html=True)
                
                st.session_state.clash_summary_df = pd.DataFrame(summary_data)

        st.markdown("---")

        # --- USING AG-GRID ---
        hide_zero_jscode = JsCode("""function(params) { if (params.value == 0 || params.value === null) { return ''; } return params.value; }""")
        clash_cell_style_jscode = JsCode(f"""
            function(params) {{
                const clashMap = {json.dumps(clash_map)};
                const date = params.data.ETA_Date;
                const colId = params.colDef.field;
                const isClash = clashMap[date] ? clashMap[date].includes(colId) : false;
                if (isClash) {{ return {{'backgroundColor': '#FFAA33', 'color': 'black'}}; }}
                return null;
            }}
        """)
        zebra_row_style_jscode = JsCode(f"""
            function(params) {{
                const dateColorMap = {json.dumps(date_color_map)};
                const date = params.data.ETA_Date;
                const color = dateColorMap[date];
                return {{ 'background-color': color }};
            }}
        """)

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
        for col in cluster_cols:
            column_defs.append({"field": col, "headerName": col, "width": 60, "cellRenderer": hide_zero_jscode, "cellStyle": clash_cell_style_jscode})
        column_defs.append({"field": "ETA_Date", "hide": True})
        gridOptions = {"defaultColDef": default_col_def, "columnDefs": column_defs, "getRowStyle": zebra_row_style_jscode}

        AgGrid(df_for_grid.drop(columns=['ETA_str']), gridOptions=gridOptions, height=600, width='100%', theme='streamlit', allow_unsafe_jscode=True)
        
        # --- EXCEL DOWNLOAD BUTTON LOGIC ---
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            if 'clash_summary_df' in st.session_state and st.session_state.clash_summary_df is not None:
                summary_df_for_export = st.session_state.clash_summary_df
                
                final_summary_export = []
                last_date = None
                if not summary_df_for_export.empty:
                    for index, row in summary_df_for_export.iterrows():
                        current_date = row['Clash Date']
                        if last_date is not None and current_date != last_date:
                            final_summary_export.append({}) 
                        final_summary_export.append(row.to_dict())
                        last_date = current_date
                
                final_summary_df = pd.DataFrame(final_summary_export)

                final_summary_df.to_excel(writer, sheet_name='Clash Summary', index=False)
                
                workbook = writer.book
                summary_sheet = writer.sheets['Clash Summary']
                center_format = workbook.add_format({'align': 'center', 'valign': 'vcenter'})

                for idx, col in enumerate(final_summary_df.columns):
                    series = final_summary_df[col].dropna()
                    if not series.empty:
                        max_len = max([len(str(s)) for s in series] + [len(col)]) + 4
                        summary_sheet.set_column(idx, idx, max_len, center_format)
                    else:
                        summary_sheet.set_column(idx, idx, len(col) + 4, center_format)

        st.download_button(
            label="📥 Download Clash Summary (Excel)",
            data=output.getvalue(),
            file_name="clash_summary_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

# --- MAIN STRUCTURE WITH TABS ---
tab1, tab2 = st.tabs(["🚨 Clash Analysis", "📈 Loading Forecast"])

with tab1:
    render_clash_tab()

with tab2:
    render_forecast_tab()
