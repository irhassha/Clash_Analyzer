import streamlit as st
import pandas as pd
import os
from datetime import datetime, timedelta

st.set_page_config(page_title="App Pencocokan Vessel", layout="wide")
st.title("🚢 Aplikasi Pencocokan Jadwal Kapal & Unit List")

st.info("Aplikasi ini secara otomatis membersihkan nama kolom dari spasi tambahan untuk mencegah error.")

@st.cache_data
def load_vessel_codes_from_repo(possible_names=['vessel codes.xlsx', 'vessel_codes.xls', 'vessel_codes.csv']):
    for filename in possible_names:
        if os.path.exists(filename):
            try:
                if filename.lower().endswith('.csv'): df = pd.read_csv(filename)
                else: df = pd.read_excel(filename)
                df.columns = df.columns.str.strip() # <-- PEMBERSIHAN NAMA KOLOM
                return df
            except Exception as e:
                st.error(f"Gagal membaca file '{filename}': {e}"); return None
    st.error(f"File kode kapal tidak ditemukan."); return None

def style_final_table(df):
    styler = pd.DataFrame('', index=df.index, columns=df.columns)
    df_copy = df.copy()
    df_copy['ETA_Date'] = pd.to_datetime(df_copy['ETA']).dt.date
    cluster_cols = [col for col in df.columns if col not in ['VESSEL', 'CODE', 'VOY_OUT', 'ETA', 'TOTAL']]
    for col in cluster_cols:
        subset_df = df_copy[df_copy[col] > 0]
        clash_dates = subset_df[subset_df.duplicated(subset='ETA_Date', keep=False)]['ETA_Date'].unique()
        for date in clash_dates:
            clashing_indices = subset_df[subset_df['ETA_Date'] == date].index
            styler.loc[clashing_indices, col] = 'background-color: #FFAA33'
    date_change_indices = df_copy.index[df_copy['ETA_Date'].diff() > pd.Timedelta(0)]
    for idx in date_change_indices:
        for col_name in styler.columns:
            current_style = styler.loc[idx, col_name]
            separator_style = 'border-top: 2px solid #555555;'
            styler.loc[idx, col_name] = f"{current_style}; {separator_style}"
    return styler

st.sidebar.header("⚙️ Unggah File Anda")
schedule_file = st.sidebar.file_uploader("1. Unggah File Jadwal Kapal", type=['xlsx', 'csv'])
unit_list_file = st.sidebar.file_uploader("2. Unggah File Daftar Unit", type=['xlsx', 'csv'])
process_button = st.sidebar.button("🚀 Proses Data", type="primary")

df_vessel_codes = load_vessel_codes_from_repo()

if process_button:
    if schedule_file and unit_list_file and (df_vessel_codes is not None and not df_vessel_codes.empty):
        with st.spinner('Memuat dan memproses data...'):
            try:
                # 1. Memuat file dan langsung membersihkan nama kolom
                if schedule_file.name.lower().endswith(('.xls', '.xlsx')): df_schedule = pd.read_excel(schedule_file)
                else: df_schedule = pd.read_csv(schedule_file)
                df_schedule.columns = df_schedule.columns.str.strip() # <-- PEMBERSIHAN NAMA KOLOM

                if unit_list_file.name.lower().endswith(('.xls', '.xlsx')): df_unit_list = pd.read_excel(unit_list_file)
                else: df_unit_list = pd.read_csv(unit_list_file)
                df_unit_list.columns = df_unit_list.columns.str.strip() # <-- PEMBERSIHAN NAMA KOLOM

                # ---- Mulai proses utama (sekarang nama kolom sudah bersih) ----
                original_vessels_list = df_schedule['VESSEL'].unique().tolist()
                df_schedule['ETA'] = pd.to_datetime(df_schedule['ETA'], errors='coerce')

                # ... sisa skrip sama persis karena sekarang dijamin tidak akan error ...
                df_schedule_with_code = pd.merge(df_schedule, df_vessel_codes, left_on="VESSEL", right_on="Description", how="left").rename(columns={"Value": "CODE"})
                merged_df = pd.merge(df_schedule_with_code, df_unit_list, left_on=['CODE', 'VOY_OUT'], right_on=['Carrier Out', 'Voyage Out'], how='inner')

                if not merged_df.empty:
                    merged_df = merged_df[merged_df['VESSEL'].isin(original_vessels_list)]
                    excluded_areas = [str(i) for i in range(801, 809)] 
                    merged_df['Area (EXE)'] = merged_df['Area (EXE)'].astype(str)
                    filtered_data = merged_df[~merged_df['Area (EXE)'].isin(excluded_areas)]
                    if filtered_data.empty: st.warning("Setelah filter, tidak ada data tersisa."); st.stop()

                    grouping_cols = ['VESSEL', 'CODE', 'VOY_OUT', 'ETA']
                    pivot_df = filtered_data.pivot_table(index=grouping_cols, columns='Area (EXE)', aggfunc='size', fill_value=0)
                    pivot_df['TOTAL'] = pivot_df.sum(axis=1)
                    pivot_df = pivot_df.reset_index()
                    
                    two_days_ago = pd.Timestamp.now() - timedelta(days=2)
                    condition_to_hide = (pivot_df['ETA'] < two_days_ago) & (pivot_df['TOTAL'] < 50)
                    pivot_df = pivot_df[~condition_to_hide]
                    if pivot_df.empty: st.warning("Setelah filter ETA & Total, tidak ada data tersisa."); st.stop()

                    cols_awal = ['VESSEL', 'CODE', 'VOY_OUT', 'ETA', 'TOTAL']
                    cols_clusters = [col for col in pivot_df.columns if col not in cols_awal]
                    final_display_cols = cols_awal + sorted(cols_clusters)
                    pivot_df = pivot_df[final_display_cols]
                    pivot_df = pivot_df.sort_values(by='ETA', ascending=True).reset_index(drop=True)

                    st.header("✅ Hasil Akhir")
                    st.success(f"Berhasil memproses data untuk {len(pivot_df)} jadwal kapal.")
                    
                    eta_for_styling = pivot_df['ETA']
                    styled_df = pivot_df.style.apply(style_final_table, axis=None, df=eta_for_styling)
                    styled_df = styled_df.format({'ETA': lambda x: x.strftime('%Y-%m-%d %H:%M:%S')})
                    
                    st.dataframe(styled_df, height=(len(pivot_df) + 1) * 35 + 3)
                    
                    csv_export = pivot_df.to_csv(index=False).encode('utf-8')
                    st.download_button(label="📥 Unduh Hasil sebagai CSV", data=csv_export, file_name='hasil_pivot_clusters.csv', mime='text/csv')
                else:
                    st.warning("Tidak ditemukan data yang cocok antara file Jadwal Kapal dan Daftar Unit.")
            except KeyError as e:
                st.error(f"Error: Nama kolom tidak ditemukan: {e}. Pastikan file Excel/CSV Anda memiliki nama kolom yang benar (misal: 'VESSEL', 'CODE', 'VOY_OUT', dll).")
            except Exception as e:
                st.error(f"Terjadi kesalahan saat memproses file: {e}"); st.error(f"Detail error: {str(e)}")
    else:
        st.warning("Mohon unggah kedua file dan pastikan file kode kapal ada di repository sebelum memproses.")
