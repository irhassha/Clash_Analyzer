import streamlit as st
import pandas as pd
import os
from datetime import datetime, timedelta
import json
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode

# --- Page Setup ---
st.set_page_config(page_title="Clash Analyzer", layout="wide")
st.title("🚨 Yard Clash Monitoring")

@st.cache_data
def load_vessel_codes_from_repo(possible_names=['vessel codes.xlsx', 'vessel_codes.xls', 'vessel_codes.csv']):
    for filename in possible_names:
        if os.path.exists(filename):
            try:
                if filename.lower().endswith('.csv'):
                    df = pd.read_csv(filename)
                else:
                    df = pd.read_excel(filename)
                df.columns = df.columns.str.strip()
                return df
            except Exception as e:
                st.error(f"Failed to read file '{filename}': {e}")
                return None
    st.error("Vessel code file not found.")
    return None

# --- Sidebar ---
st.sidebar.header("⚙️ Your File Uploads")
schedule_file = st.sidebar.file_uploader("1. Upload Vessel Schedule", type=['xlsx', 'csv'])
unit_list_file = st.sidebar.file_uploader("2. Upload Unit List", type=['xlsx', 'csv'])
process_button = st.sidebar.button("🚀 Process Data", type="primary")

if 'processed_df' not in st.session_state:
    st.session_state.processed_df = None

df_vessel_codes = load_vessel_codes_from_repo()

if process_button:
    if schedule_file and unit_list_file and (df_vessel_codes is not None and not df_vessel_codes.empty):
        with st.spinner('Loading and processing data...'):
            try:
                if schedule_file.name.lower().endswith(('.xls', '.xlsx')):
                    df_schedule = pd.read_excel(schedule_file)
                else:
                    df_schedule = pd.read_csv(schedule_file)
                df_schedule.columns = df_schedule.columns.str.strip()

                if unit_list_file.name.lower().endswith(('.xls', '.xlsx')):
                    df_unit_list = pd.read_excel(unit_list_file)
                else:
                    df_unit_list = pd.read_csv(unit_list_file)
                df_unit_list.columns = df_unit_list.columns.str.strip()

                original_vessels_list = df_schedule['VESSEL'].unique().tolist()
                df_schedule['ETA'] = pd.to_datetime(df_schedule['ETA'], errors='coerce')
                df_schedule_with_code = pd.merge(
                    df_schedule, df_vessel_codes,
                    left_on="VESSEL", right_on="Description", how="left"
                ).rename(columns={"Value": "CODE"})

                merged_df = pd.merge(
                    df_schedule_with_code, df_unit_list,
                    left_on=['CODE', 'VOY_OUT'], right_on=['Carrier Out', 'Voyage Out'],
                    how='inner'
                )

                if merged_df.empty:
                    st.warning("No matching data found.")
                    st.session_state.processed_df = None
                    st.stop()

                merged_df = merged_df[merged_df['VESSEL'].isin(original_vessels_list)]
                excluded_areas = [str(i) for i in range(801, 809)]
                merged_df['Area (EXE)'] = merged_df['Area (EXE)'].astype(str)
                filtered_data = merged_df[~merged_df['Area (EXE)'].isin(excluded_areas)]

                if filtered_data.empty:
                    st.warning("No data remaining after filtering.")
                    st.session_state.processed_df = None
                    st.stop()

                grouping_cols = ['VESSEL', 'CODE', 'VOY_OUT', 'ETA']
                pivot_df = filtered_data.pivot_table(index=grouping_cols, columns='Area (EXE)', aggfunc='size', fill_value=0)

                cluster_cols = pivot_df.columns.tolist()
                pivot_df['Total Box'] = pivot_df[cluster_cols].sum(axis=1)
                pivot_df['Total cluster'] = (pivot_df[cluster_cols] > 0).sum(axis=1)
                pivot_df = pivot_df.reset_index()

                two_days_ago = pd.Timestamp.now() - timedelta(days=2)
                condition_to_hide = (pivot_df['ETA'] < two_days_ago) & (pivot_df['Total Box'] < 50)
                pivot_df = pivot_df[~condition_to_hide]

                cols_awal = ['VESSEL', 'CODE', 'VOY_OUT', 'ETA', 'Total Box', 'Total cluster']
                final_cluster_cols = [col for col in pivot_df.columns if col not in cols_awal]
                final_display_cols = cols_awal + sorted(final_cluster_cols)
                pivot_df = pivot_df[final_display_cols]

                pivot_df['ETA'] = pd.to_datetime(pivot_df['ETA']).dt.strftime('%Y-%m-%d %H:%M:%S')
                pivot_df = pivot_df.sort_values(by='ETA', ascending=True).reset_index(drop=True)

                st.session_state.processed_df = pivot_df
                st.success("Data processed successfully!")

            except Exception as e:
                st.error(f"An error occurred during processing: {e}")
                st.session_state.processed_df = None
    else:
        st.warning("Please upload both files.")

# --- Display Area ---
if st.session_state.processed_df is not None:
    display_df = st.session_state.processed_df.copy()
    st.header("✅ Analysis Result")

    display_df['ETA_Date'] = pd.to_datetime(display_df['ETA']).dt.strftime('%Y-%m-%d')

    # Clash logic
    clash_map = {}
    cluster_cols = [col for col in display_df.columns if col not in ['VESSEL', 'CODE', 'VOY_OUT', 'ETA', 'Total Box', 'Total cluster', 'ETA_Date']]
    for date, group in display_df.groupby('ETA_Date'):
        clash_areas = [col for col in cluster_cols if (group[col] > 0).sum() > 1]
        if clash_areas:
            clash_map[date] = clash_areas

    unique_dates = sorted(display_df['ETA_Date'].unique())
    selected_dates = st.multiselect("**Focus on Date(s):**", options=unique_dates)

    faded_dates = [d for d in unique_dates if d not in selected_dates] if selected_dates else []

    # JS Functions
    hide_zero_jscode = JsCode("""
        function(params) {
            if (params.value == 0) {
                return '';
            }
            return params.value;
        }
    """)

    cell_style_jscode = JsCode(f"""
        function(params) {{
            const clashMap = {json.dumps(clash_map)};
            const fadedDates = {json.dumps(faded_dates)};
            const date = params.data.ETA_Date;
            const colId = params.colDef.field;

            const isFaded = fadedDates.includes(date);
            const isClash = clashMap[date] ? clashMap[date].includes(colId) : false;

            if (isClash && isFaded) {{
                return {{'backgroundColor': '#FFE8D6', 'color': '#BDBDBD'}};
            }}
            if (isClash && !isFaded) {{
                return {{'backgroundColor': '#FFAA33', 'color': 'black'}};
            }}
            if (isFaded) {{
                return {{'color': '#E0E0E0'}};
            }}
            return null;
        }}
    """)

    # Grid Configuration
    gb = GridOptionsBuilder.from_dataframe(display_df)

    # Freeze columns
    frozen = ['VESSEL', 'CODE', 'VOY_OUT', 'ETA', 'Total Box', 'Total cluster']
    for col in frozen:
        gb.configure_column(col, pinned="left", width=120)

    # Configure all cluster cols
    for col in cluster_cols:
        gb.configure_column(col, cellStyle=cell_style_jscode, cellRenderer=hide_zero_jscode, width=90)

    # Non-cluster fields with zero-hider
    gb.configure_column("Total Box", cellRenderer=hide_zero_jscode)
    gb.configure_column("Total cluster", cellRenderer=hide_zero_jscode)

    # Disable filter and sort globally
    gb.configure_default_column(resizable=True, filterable=False, sortable=False, editable=False)

    gridOptions = gb.build()

    custom_css = {
        # HILANGKAN ICON FILTER & SORT
        ".ag-theme-streamlit .ag-header-cell-menu-button": {
            "display": "none !important"
        },
        # RATAKAN HEADER TEXT KE TENGAH
        ".ag-theme-streamlit .ag-header-cell-label": {
            "justify-content": "center !important"
        },
        # GARIS-GARIS TABEL
        ".ag-theme-streamlit .ag-cell": {
            "border-right": "2px solid #DCDCDC !important",
            "border-bottom": "2px solid #DCDCDC !important"
        },
        ".ag-theme-streamlit .ag-header-cell": {
            "border-bottom": "2px solid #CCCCCC !important"
        }
    }

    st.markdown("---")
    AgGrid(
        display_df,
        gridOptions=gridOptions,
        height=600,
        width='100%',
        theme='streamlit',
        custom_css=custom_css,
        allow_unsafe_jscode=True,
        column_defs=[{"field": "ETA_Date", "hide": True}]
    )

    # Download Button
    csv_export = display_df.drop(columns=["ETA_Date"]).to_csv(index=False).encode('utf-8')
    st.download_button(label="📥 Download Result as CSV", data=csv_export, file_name='analysis_result.csv', mime='text/csv')
