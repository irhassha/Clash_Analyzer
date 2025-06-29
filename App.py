import streamlit as st
import pandas as pd
import os
from datetime import datetime, timedelta

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

# --- FUNGSI STYLING BARU (METODE ROW-WISE) ---
def style_row_wise(row, is_faded_map, clash_cells):
    """Menerapkan style baris per baris."""
    # Tentukan style dasar untuk seluruh baris (pudar atau normal)
    is_faded = is_faded_map.get(row.name, False)
    base_style = 'color: #E0E0E0;' if is_faded else ''
    styles = [base_style for _ in row.index]

    # Timpa style untuk sel yang bentrok
    for i, col_name in enumerate(row.index):
        if (row.name, col_name) in clash_cells:
            if is_faded:
                styles[i] = 'background-color: #FFE8D6; color: #BDBDBD;' # Oranye Pudar
            else:
                styles[i] = 'background-color: #FFAA33; color: black;'    # Oranye Terang
    return styles

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
                # ... (semua proses loading hingga pivot tidak berubah) ...
                if schedule_file.name.lower().endswith(('.xls', '.xlsx')): df_schedule = pd.read_excel(schedule_file)
                else: df_schedule = pd.read_csv(schedule_file)
                df_schedule.columns = df_schedule.columns.str.strip()
                if unit_list_file.name.lower().endswith(('.xls', '.xlsx')): df_unit_list = pd.read_excel(unit_list_file)
                else: df_unit_list = pd.read_csv(unit_list_file)
                df_unit_list.columns = df_unit_list.columns.str.strip()
                original_vessels_list = df_schedule['VESSEL'].unique().tolist()
                df_schedule['ETA'] = pd.to_datetime(df_schedule['ETA'], errors='coerce')
                df_schedule_with_code = pd.merge(df_schedule, df_vessel_codes, left_on="VESSEL", right_on="Description", how="left").rename(columns={"Value": "CODE"})
                merged_df = pd.merge(df_schedule_with_code, df_unit_list, left_on=['CODE', 'VOY_OUT'], right_on=['Carrier Out', 'Voyage Out'], how='inner')
                if merged_df.empty: st.warning("No matching data found."); st.session_state.processed_df = None; st.stop()
                merged_df = merged_df[merged_df['VESSEL'].isin(original_vessels_list)]
                excluded_areas = [str(i) for i in range(801, 809)]
                merged_df['Area (EXE)'] = merged_df['Area (EXE)'].astype(str)
                filtered_data = merged_df[~merged_df['Area (EXE)'].isin(excluded_areas)]
                if filtered_data.empty: st.warning("No data remaining after filtering."); st.session_state.processed_df = None; st.stop()
                grouping_cols = ['VESSEL', 'CODE', 'VOY_OUT', 'ETA']
                pivot_df = filtered_data.pivot_table(index=grouping_cols, columns='Area (EXE)', aggfunc='size', fill_value=0)
                cluster_cols_for_calc = pivot_df.columns.tolist()
                pivot_df['Total Box'] = pivot_df[cluster_cols_for_calc].sum(axis=1)
                pivot_df['Total cluster'] = (pivot_df[cluster_cols_for_calc] > 0).sum(axis=1)
                pivot_df = pivot_df.reset_index()
                two_days_ago = pd.Timestamp.now() - timedelta(days=2)
                condition_to_hide = (pivot_df['ETA'] < two_days_ago) & (pivot_df['Total Box'] < 50)
                pivot_df = pivot_df[~condition_to_hide]
                if pivot_df.empty: st.warning("No data remaining after ETA & Total filter."); st.session_state.processed_df = None; st.stop()
                cols_awal = ['VESSEL', 'CODE', 'VOY_OUT', 'ETA', 'Total Box', 'Total cluster']
                final_cluster_cols = [col for col in pivot_df.columns if col not in cols_awal]
                final_display_cols = cols_awal + sorted(final_cluster_cols)
                pivot_df = pivot_df[final_display_cols]
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
    display_df = st.session_state.processed_df
    
    st.header("✅ Analysis Result")
    st.markdown("---")

    col1, col2 = st.columns([1, 2])
    with col1:
        display_df_copy = display_df.copy()
        display_df_copy['ETA_Date_Only'] = pd.to_datetime(display_df_copy['ETA']).dt.date
        unique_dates_in_data = sorted(display_df_copy['ETA_Date_Only'].unique())
        selected_dates = st.multiselect(
            "**Focus on Date(s):**",
            options=unique_dates_in_data,
            format_func=lambda date: date.strftime('%Y-%m-%d')
        )
    
    # --- PERSIAPAN SEBELUM STYLING ---
    df_to_style = display_df.copy()
    df_to_style['ETA'] = pd.to_datetime(df_to_style['ETA'])
    df_to_style['ETA_Date'] = df_to_style['ETA'].dt.date

    # 1. Tentukan baris mana saja yang pudar
    is_faded_map = pd.Series(False, index=df_to_style.index)
    if selected_dates and len(selected_dates) < len(df_to_style['ETA_Date'].unique()):
        is_faded_map[~df_to_style['ETA_Date'].isin(selected_dates)] = True

    # 2. Tentukan sel mana saja yang bentrok
    clash_cells = set()
    cluster_cols = [col for col in df_to_style.columns if col not in ['VESSEL', 'CODE', 'VOY_OUT', 'ETA', 'Total Box', 'Total cluster', 'ETA_Date']]
    for col in cluster_cols:
        numeric_col = pd.to_numeric(df_to_style[col], errors='coerce').fillna(0)
        subset_df = df_to_style[numeric_col > 0]
        clash_dates = subset_df[subset_df.duplicated(subset='ETA_Date', keep=False)]['ETA_Date'].unique()
        for date in clash_dates:
            clashing_indices = subset_df[subset_df['ETA_Date'] == date].index
            for idx in clashing_indices:
                clash_cells.add((idx, col))

    # Hapus kolom sementara sebelum styling
    df_to_style = df_to_style.drop(columns=['ETA_Date'])

    # 3. Definisikan format
    header_style = {'selector': 'th', 'props': [('font-weight', 'bold')]}
    numeric_cols_format = [col for col in df_to_style.columns if df_to_style[col].dtype in ['int64', 'float64']]
    formatter = {col: lambda x: '' if x == 0 else f'{x:.0f}' for col in numeric_cols_format}
    formatter['ETA'] = '{:%Y-%m-%d %H:%M:%S}'

    # 4. Terapkan styling dan format
    styled_df = (
        df_to_style.style
        .apply(style_row_wise, axis=1, is_faded_map=is_faded_map, clash_cells=clash_cells)
        .format(formatter)
        .set_table_styles([header_style])
    )
    
    st.dataframe(styled_df, use_container_width=True)
    
    csv_export = display_df.to_csv(index=False).encode('utf-8')
    st.download_button(label="📥 Download Result as CSV", data=csv_export, file_name='analysis_result.csv', mime='text/csv')






