import streamlit as st
import pandas as pd
import os
from datetime import datetime, timedelta

# --- FUNGSI BARU UNTUK MEMUAT FILE CSS EKSTERNAL ---
def load_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"File CSS tidak ditemukan: {file_name}. Pastikan file tersebut ada di repository.")

# --- Panggil fungsi untuk memuat style.css di awal ---
load_css("style.css")


# --- Konfigurasi Halaman & Judul ---
# st.set_page_config() tidak diperlukan lagi karena tema diatur oleh CSS
st.title("🚢 Aplikasi Pencocokan Jadwal Kapal & Unit List")

# ... sisa skrip Anda dari sini ke bawah tidak perlu diubah sama sekali ...
@st.cache_data
def load_vessel_codes_from_repo(possible_names=['vessel_codes.xlsx', 'vessel_codes.xls', 'vessel_codes.csv']):
    for filename in possible_names:
        if os.path.exists(filename):
            try:
                if filename.lower().endswith('.csv'): df = pd.read_csv(filename)
                else: df = pd.read_excel(filename)
                df.columns = df.columns.str.strip()
                return df
            except Exception as e:
                st.error(f"Gagal membaca file '{filename}': {e}"); return None
    st.error(f"File kode kapal tidak ditemukan."); return None

def apply_all_styles(df, selected_dates):
    styler = pd.DataFrame('', index=df.index, columns=df.columns)
    df_copy = df.copy()
    df_copy['ETA'] = pd.to_datetime(df_copy['ETA'])
    df_copy['ETA_Date'] = df_copy['ETA'].dt.date
    is_faded_row = pd.Series(False, index=df.index)
    if selected_dates and len(selected_dates) < len(df_copy['ETA_Date'].unique()):
        is_faded_row[~df_copy['ETA_Date'].isin(selected_dates)] = True
    clash_mask = pd.DataFrame(False, index=df.index, columns=df.columns)
    cluster_cols = [col for col in df.columns if col not in ['VESSEL', 'CODE', 'VOY_OUT', 'ETA', 'TOTAL']]
    for col in cluster_cols:
        numeric_col = pd.to_numeric(df_copy[col], errors='coerce').fillna(0)
        subset_df = df_copy[numeric_col > 0]
        clash_dates = subset_df[subset_df.duplicated(subset='ETA_Date', keep=False)]['ETA_Date'].unique()
        for date in clash_dates:
            clashing_indices = subset_df[subset_df['ETA_Date'] == date].index
            clash_mask.loc[clashing_indices, col] = True
    for r_idx in df.index:
        for c_name in df.columns:
            is_faded = is_faded_row[r_idx]
            is_clash = clash_mask.loc[r_idx, c_name]
            if is_clash and is_faded:
                styler.loc[r_idx, c_name] = 'background-color: #FFE8D6; color: #BDBDBD;'
            elif is_clash and not is_faded:
                styler.loc[r_idx, c_name] = 'background-color: #FFAA33; color: black;'
            elif not is_clash and is_faded:
                styler.loc[r_idx, c_name] = 'color: #E0E0E0;'
    return styler

st.sidebar.header("⚙️ Unggah File Anda")
schedule_file = st.sidebar.file_uploader("1. Unggah File Jadwal Kapal", type=['xlsx', 'csv'])
unit_list_file = st.sidebar.file_uploader("2. Unggah File Daftar Unit", type=['xlsx', 'csv'])
process_button = st.sidebar.button("🚀 Proses Data", type="primary")

if 'processed_df' not in st.session_state:
    st.session_state.processed_df = None

df_vessel_codes = load_vessel_codes_from_repo()

if process_button:
    if schedule_file and unit_list_file and (df_vessel_codes is not None and not df_vessel_codes.empty):
        with st.spinner('Memuat dan memproses data...'):
            try:
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
                if merged_df.empty: st.warning("Tidak ditemukan data yang cocok."); st.session_state.processed_df = None; st.stop()
                merged_df = merged_df[merged_df['VESSEL'].isin(original_vessels_list)]
                excluded_areas = [str(i) for i in range(801, 809)]
                merged_df['Area (EXE)'] = merged_df['Area (EXE)'].astype(str)
                filtered_data = merged_df[~merged_df['Area (EXE)'].isin(excluded_areas)]
                if filtered_data.empty: st.warning("Setelah filter, tidak ada data tersisa."); st.session_state.processed_df = None; st.stop()
                grouping_cols = ['VESSEL', 'CODE', 'VOY_OUT', 'ETA']
                pivot_df = filtered_data.pivot_table(index=grouping_cols, columns='Area (EXE)', aggfunc='size', fill_value=0)
                pivot_df['TOTAL'] = pivot_df.sum(axis=1)
                pivot_df = pivot_df.reset_index()
                two_days_ago = pd.Timestamp.now() - timedelta(days=2)
                condition_to_hide = (pivot_df['ETA'] < two_days_ago) & (pivot_df['TOTAL'] < 50)
                pivot_df = pivot_df[~condition_to_hide]
                if pivot_df.empty: st.warning("Setelah filter ETA & Total, tidak ada data tersisa."); st.session_state.processed_df = None; st.stop()
                cols_awal = ['VESSEL', 'CODE', 'VOY_OUT', 'ETA', 'TOTAL']
                cols_clusters = [col for col in pivot_df.columns if col not in cols_awal]
                final_display_cols = cols_awal + sorted(cols_clusters)
                pivot_df = pivot_df[final_display_cols]
                pivot_df = pivot_df.sort_values(by='ETA', ascending=True).reset_index(drop=True)
                st.session_state.processed_df = pivot_df
                st.success("Data berhasil diproses! Anda sekarang bisa menggunakan filter tanggal di sidebar.")

            except Exception as e:
                st.error(f"Terjadi kesalahan saat memproses file: {e}")
                st.session_state.processed_df = None
    else:
        st.warning("Mohon unggah kedua file dan pastikan file kode kapal ada di repository sebelum memproses.")

if st.session_state.processed_df is not None:
    display_df = st.session_state.processed_df
    st.sidebar.header("Highlight Tanggal")
    display_df_copy = display_df.copy()
    display_df_copy['ETA_Date_Only'] = pd.to_datetime(display_df_copy['ETA']).dt.date
    unique_dates_in_data = sorted(display_df_copy['ETA_Date_Only'].unique())
    selected_dates = st.sidebar.multiselect("Pilih tanggal untuk difokuskan:", options=unique_dates_in_data, format_func=lambda date: date.strftime('%Y-%m-%d'))
    st.header("✅ Hasil Akhir")
    df_to_style = display_df.copy()
    numeric_cols = [col for col in df_to_style.columns if df_to_style[col].dtype in ['int64', 'float64']]
    formatter = {col: lambda x: '' if x == 0 else f'{x:.0f}' for col in numeric_cols}
    formatter['ETA'] = '{:%Y-%m-%d %H:%M:%S}'
    styled_df = (
        df_to_style.style
        .apply(apply_all_styles, axis=None, selected_dates=selected_dates)
        .format(formatter)
    )
    st.dataframe(styled_df, use_container_width=True)
    csv_export = display_df.to_csv(index=False).encode('utf-8')
    st.download_button(label="📥 Unduh Hasil sebagai CSV", data=csv_export, file_name='hasil_pivot_clusters.csv', mime='text/csv')
