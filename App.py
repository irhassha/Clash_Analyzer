import streamlit as st
import pandas as pd
import os
from datetime import datetime, timedelta
import json

# Import pustaka baru
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
                pivot_df['Total Box'] = pivot_df[cluster_cols_for_calc].sum(axis=1)
                pivot_df['Total cluster'] = (pivot_df[cluster_cols_for_calc] > 0).sum(axis=1)
                pivot_df = pivot_df.reset_index()
                
                # 5. Conditional Filtering
                two_days_ago = pd.Timestamp.now() - timedelta(days=2)
                condition_to_hide = (pivot_df['ETA'] < two_days_ago) & (pivot_df['Total Box'] < 50)
                pivot_df = pivot_df[~condition_to_hide]
                if pivot_df.empty: st.warning("No data remaining after ETA & Total filter."); st.session_state.processed_df = None; st.stop()

                # 6. Sorting and Ordering
                cols_awal = ['VESSEL', 'CODE', 'VOY_OUT', 'ETA', 'Total Box', 'Total cluster']
                final_cluster_cols = [col for col in pivot_df.columns if col not in cols_awal]
                final_display_cols = cols_awal + sorted(final_cluster_cols)
                pivot_df = pivot_df[final_display_cols]
                
                # Format ETA untuk display
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

    # --- Persiapan untuk Styling AG Grid ---
    
    df_for_grid = display_df.copy()
    df_for_grid['ETA_Date'] = pd.to_datetime(df_for_grid['ETA']).dt.strftime('%Y-%m-%d')
    
    # Widget untuk memilih tanggal yang akan difokuskan
    unique_dates_in_data = sorted(df_for_grid['ETA_Date'].unique())
    selected_dates_str = st.multiselect(
        "**Focus on Date(s):**",
        options=unique_dates_in_data
    )
    
    # Tentukan tanggal mana yang akan dibuat pudar
    faded_dates_str = []
    if selected_dates_str:
        faded_dates_str = [d for d in unique_dates_in_data if d not in selected_dates_str]

    # Buat Peta Warna untuk Zebra Pattern
    # =======================================================
    # GANTI WARNA ZEBRA PATTERN DI SINI
    zebra_color_1 = '#FFFFFF' # Putih
    zebra_color_2 = '#F0F2F6' # Abu-abu muda
    # =======================================================
    zebra_colors = [zebra_color_1, zebra_color_2]
    date_color_map = {date: zebra_colors[i % 2] for i, date in enumerate(unique_dates_in_data)}
    
    # Tentukan sel mana saja yang bentrok
    clash_map = {}
    cluster_cols = [col for col in df_for_grid.columns if col not in ['VESSEL', 'CODE', 'VOY_OUT', 'ETA', 'Total Box', 'Total cluster', 'ETA_Date']]
    for date, group in df_for_grid.groupby('ETA_Date'):
        clash_areas_for_date = []
        for col in cluster_cols:
            if (group[col] > 0).sum() > 1:
                clash_areas_for_date.append(col)
        if clash_areas_for_date:
            clash_map[date] = clash_areas_for_date

    # --- PENGGUNAAN AG-GRID DENGAN SEMUA FITUR ---
    
    hide_zero_jscode = JsCode("""function(params) { if (params.value == 0 || params.value === null) { return ''; } return params.value; }""")
    
    # JsCode untuk styling sel (Zebra + Fade + Highlight)
    cell_style_jscode = JsCode(f"""
        function(params) {{
            // Lokasi untuk mengubah warna secara manual
            // =======================================================
            const FADED_TEXT_COLOR = '#E0E0E0';
            const BRIGHT_ORANGE = '#FFAA33';
            const FADED_ORANGE_BG = '#FFE8D6';
            const FADED_ORANGE_TEXT = '#BDBDBD';
            // =======================================================

            // Data yang disuntikkan dari Python
            const dateColorMap = {json.dumps(date_color_map)};
            const clashMap = {json.dumps(clash_map)};
            const fadedDates = {json.dumps(faded_dates_str)};
            
            // Variabel untuk baris dan sel saat ini
            const date = params.data.ETA_Date;
            const colId = params.colDef.field;
            const isFaded = fadedDates.includes(date);
            const isClash = clashMap[date] ? clashMap[date].includes(colId) : false;

            // Logika styling
            let style = {{}};
            
            // 1. Terapkan warna dasar Zebra Pattern
            style.backgroundColor = dateColorMap[date];
            
            // 2. Terapkan efek fade jika perlu (menimpa warna teks)
            if (isFaded) {{
                style.color = FADED_TEXT_COLOR;
            }}

            // 3. Terapkan highlight bentrokan (menimpa semua style sebelumnya jika bentrok)
            if (isClash) {{
                if (isFaded) {{
                    style.backgroundColor = FADED_ORANGE_BG;
                    style.color = FADED_ORANGE_TEXT;
                }} else {{
                    style.backgroundColor = BRIGHT_ORANGE;
                    style.color = 'black';
                }}
            }}
            
            return style;
        }}
    """)
    
    # --- KONFIGURASI GRID SECARA MANUAL ---
    gb = GridOptionsBuilder.from_dataframe(df_for_grid)
    
    gb.configure_default_column(
        resizable=True, 
        sortable=True, 
        editable=False, 
        suppressMenu=True,
        minWidth=50
    )

    pinned_cols = ['VESSEL', 'CODE', 'VOY_OUT', 'ETA', 'Total Box', 'Total cluster']
    for col in pinned_cols:
        width = 115 if col == 'VESSEL' else 85
        if col == 'ETA': width = 125 
        gb.configure_column(col, pinned="left", width=width, cellStyle=cell_style_jscode)
        if col in ["Total Box", "Total cluster"]:
            gb.configure_column(col, cellRenderer=hide_zero_jscode)

    for col in cluster_cols:
        gb.configure_column(col, width=65, cellRenderer=hide_zero_jscode, cellStyle=cell_style_jscode)
    
    gridOptions = gb.build()
    
    gridOptions['columnDefs'].append({"field": "ETA_Date", "hide": True})

    # Tampilkan tabel
    st.markdown("---")
    AgGrid(
        df_for_grid,
        gridOptions=gridOptions,
        height=600,
        width='100%',
        theme='streamlit',
        allow_unsafe_jscode=True,
    )
    
    csv_export = display_df.to_csv(index=False).encode('utf-8')
    st.download_button(label="📥 Download Result as CSV", data=csv_export, file_name='analysis_result.csv', mime='text/csv')
