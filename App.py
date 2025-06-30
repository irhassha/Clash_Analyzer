import streamlit as st
import pandas as pd
import os
from datetime import datetime, timedelta
import json
import io

# Import pustaka yang diperlukan
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode

# --- Konfigurasi Halaman & Judul ---
st.set_page_config(page_title="Clash Analyzer", layout="wide")
st.title("🚨 Yard Clash Monitoring")

# --- Fungsi-fungsi Inti ---
@st.cache_data
def load_vessel_codes_from_repo(possible_names=['vessel codes.xlsx', 'vessel_codes.xls', 'vessel_codes.csv']):
    """Mencari dan memuat file kode kapal."""
    for filename in possible_names:
        if os.path.exists(filename):
            try:
                if filename.lower().endswith('.csv'): df = pd.read_csv(filename)
                else: df = pd.read_excel(filename)
                df.columns = df.columns.str.strip()
                return df
            except Exception as e:
                st.error(f"Failed to read file '{filename}': {e}"); return None
    st.error(f"Vessel code file not found."); return None

# --- Sidebar & Proses Utama ---
st.sidebar.header("⚙️ Your File Uploads")
schedule_file = st.sidebar.file_uploader("1. Upload Vessel Schedule", type=['xlsx', 'csv'])
unit_list_file = st.sidebar.file_uploader("2. Upload Unit List", type=['xlsx', 'csv'])
process_button = st.sidebar.button("🚀 Process Data", type="primary")

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
                df_schedule.columns = df_schedule.columns.str.strip()
                if unit_list_file.name.lower().endswith(('.xls', '.xlsx')): df_unit_list = pd.read_excel(unit_list_file)
                else: df_unit_list = pd.read_csv(unit_list_file)
                df_unit_list.columns = df_unit_list.columns.str.strip()
                
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
                if filtered_data.empty: st.warning("No data remaining after filtering."); st.session_state.processed_df = None; st.stop()

                # 4. Pivoting
                grouping_cols = ['VESSEL', 'CODE', 'VOY_OUT', 'ETA']
                pivot_df = filtered_data.pivot_table(index=grouping_cols, columns='Area (EXE)', aggfunc='size', fill_value=0)
                
                cluster_cols_for_calc = pivot_df.columns.tolist()
                pivot_df['TTL BOX'] = pivot_df[cluster_cols_for_calc].sum(axis=1)
                pivot_df['TTL CLSTR'] = (pivot_df[cluster_cols_for_calc] > 0).sum(axis=1)
                pivot_df = pivot_df.reset_index()
                
                # 5. Conditional Filtering
                two_days_ago = pd.Timestamp.now() - timedelta(days=2)
                condition_to_hide = (pivot_df['ETA'] < two_days_ago) & (pivot_df['TTL BOX'] < 50)
                pivot_df = pivot_df[~condition_to_hide]
                if pivot_df.empty: st.warning("No data remaining after ETA & Total filter."); st.session_state.processed_df = None; st.stop()

                # 6. Sorting and Ordering
                cols_awal = ['VESSEL', 'CODE', 'VOY_OUT', 'ETA', 'TTL BOX', 'TTL CLSTR']
                final_cluster_cols = [col for col in pivot_df.columns if col not in cols_awal]
                final_display_cols = cols_awal + sorted(final_cluster_cols)
                pivot_df = pivot_df[final_display_cols]
                
                pivot_df['ETA'] = pd.to_datetime(pivot_df['ETA']).dt.strftime('%Y-%m-%d %H:%M')
                
                pivot_df = pivot_df.sort_values(by='ETA', ascending=True).reset_index(drop=True)
                
                st.session_state.processed_df = pivot_df
                st.success("Data processed successfully!")

            except Exception as e:
                st.error(f"An error occurred during processing: {e}")
                st.session_state.processed_df = None
    else:
        st.warning("Please upload both files.")

# --- Area Tampilan ---
if st.session_state.processed_df is not None:
    display_df = st.session_state.processed_df
    
    st.header("✅ Analysis Result")

    # --- Persiapan untuk Styling AG Grid dan Summary ---
    
    df_for_grid = display_df.copy()
    df_for_grid['ETA_Date'] = pd.to_datetime(df_for_grid['ETA']).dt.strftime('%Y-%m-%d')
    
    unique_dates = df_for_grid['ETA_Date'].unique()
    zebra_colors = ['#F8F0E5', '#DAC0A3'] 
    date_color_map = {date: zebra_colors[i % 2] for i, date in enumerate(unique_dates)}

    clash_map = {}
    cluster_cols = [col for col in df_for_grid.columns if col not in ['VESSEL', 'CODE', 'VOY_OUT', 'ETA', 'TTL BOX', 'TTL CLSTR', 'ETA_Date']]
    for date, group in df_for_grid.groupby('ETA_Date'):
        clash_areas_for_date = []
        for col in cluster_cols:
            if (group[col] > 0).sum() > 1:
                clash_areas_for_date.append(col)
        if clash_areas_for_date:
            clash_map[date] = clash_areas_for_date

    # --- TAMPILAN RINGKASAN CLASH DENGAN KARTU ---
    summary_data = []
    if clash_map:
        summary_exclude_blocks = ['BR9', 'RC9', 'C01', 'D01', 'OOG']

        with st.expander("Show Clash Summary", expanded=True):
            total_clash_days = len(clash_map)
            total_conflicting_blocks = sum(len(areas) for areas in clash_map.values())
            st.markdown(f"**🔥 Found {total_clash_days} clash day(s) with a total of {total_conflicting_blocks} conflicting blocks.**")
            st.markdown("---")
            
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
                            "Vessels": vessel_list_str,
                            "Remark": ""
                        })
                    summary_html += "</div></div>"
                    st.markdown(summary_html, unsafe_allow_html=True)
        
        st.session_state.clash_summary_df = pd.DataFrame(summary_data)

    st.markdown("---")


    # --- PENGGUNAAN AG-GRID ---
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
    pinned_cols = ['VESSEL', 'CODE', 'VOY_OUT', 'ETA', 'TTL BOX', 'TTL CLSTR']
    for col in pinned_cols:
        width = 110 if col == 'VESSEL' else 80
        if col == 'ETA': width = 120 
        col_def = {"field": col, "headerName": col, "pinned": "left", "width": width}
        if col in ["TTL BOX", "TTL CLSTR"]: col_def["cellRenderer"] = hide_zero_jscode
        column_defs.append(col_def)
    for col in cluster_cols:
        column_defs.append({"field": col, "headerName": col, "width": 60, "cellRenderer": hide_zero_jscode, "cellStyle": clash_cell_style_jscode})
    column_defs.append({"field": "ETA_Date", "hide": True})
    gridOptions = {"defaultColDef": default_col_def, "columnDefs": column_defs, "getRowStyle": zebra_row_style_jscode}

    AgGrid(df_for_grid, gridOptions=gridOptions, height=600, width='100%', theme='streamlit', allow_unsafe_jscode=True)
    
    # --- LOGIKA TOMBOL DOWNLOAD EXCEL ---
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        display_df.to_excel(writer, index=False, sheet_name='Analysis Result')
        
        # Jika ada summary, format dan tulis ke sheet kedua
        if 'clash_summary_df' in st.session_state and st.session_state.clash_summary_df is not None:
            summary_df_for_export = st.session_state.clash_summary_df
            
            # --- LOGIKA BARU UNTUK FORMATTING EXCEL ---
            workbook = writer.book
            summary_sheet = writer.sheets['Clash Summary']
            
            # Buat format rata tengah
            center_format = workbook.add_format({'align': 'center', 'valign': 'vcenter'})

            # Sisipkan baris kosong antar tanggal
            last_date = None
            row_to_write = 0
            for index, row in summary_df_for_export.iterrows():
                current_date = row['Clash Date']
                if last_date is not None and current_date != last_date:
                    # Tulis baris kosong sebagai pemisah
                    writer.sheets['Clash Summary'].write_string(row_to_write, 0, '')
                    row_to_write += 1
                
                # Tulis data asli
                row.to_frame().T.to_excel(writer, sheet_name='Clash Summary', startrow=row_to_write, header=False, index=False)
                last_date = current_date
                row_to_write += 1
            
            # Tulis header di awal
            summary_df_for_export.iloc[0:0].to_excel(writer, sheet_name='Clash Summary', index=False)
            
            # Terapkan format rata tengah dan auto-fit ke semua kolom di sheet summary
            for idx, col in enumerate(summary_df_for_export):
                series = summary_df_for_export[col]
                max_len = max((series.astype(str).map(len).max(), len(str(series.name)))) + 4
                summary_sheet.set_column(idx, idx, max_len, center_format)

    st.download_button(
        label="📥 Download Excel Report",
        data=output.getvalue(),
        file_name="analysis_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

