import streamlit as st
import pandas as pd
import os
from datetime import datetime, timedelta

# --- Konfigurasi Halaman & Judul ---
st.set_page_config(page_title="App Pencocokan Vessel", layout="wide")
st.title("🚢 Aplikasi Pencocokan Jadwal Kapal & Unit List")

st.info(
    "Aplikasi ini dikonfigurasi untuk secara otomatis membaca file `vessel_codes.xlsx` "
    "(atau `.csv`) yang ada di dalam repository GitHub Anda."
)

# --- Fungsi untuk memuat file kode kapal dari repository ---
@st.cache_data
def load_vessel_codes_from_repo(possible_names=['vessel codes.xlsx', 'vessel_codes.xls', 'vessel_codes.csv']):
    for filename in possible_names:
        if os.path.exists(filename):
            try:
                if filename.lower().endswith('.csv'): return pd.read_csv(filename)
                else: return pd.read_excel(filename)
            except Exception as e:
                st.error(f"Gagal membaca file '{filename}': {e}")
                return None
    st.error(f"File kode kapal tidak ditemukan. Pastikan ada file dengan nama vessel_codes.xlsx atau .csv di repository Anda.")
    return None

# --- Fungsi untuk Styling ---
def highlight_clashes(df):
    """Fungsi untuk membuat DataFrame style yang akan mewarnai sel bentrokan."""
    # Buat DataFrame kosong dengan struktur yang sama untuk menyimpan style CSS
    styler = pd.DataFrame('', index=df.index, columns=df.columns)
    
    # Ambil hanya tanggal dari ETA untuk perbandingan
    df_copy = df.copy()
    df_copy['ETA_Date'] = pd.to_datetime(df_copy['ETA']).dt.date
    
    # Tentukan kolom mana saja yang merupakan cluster
    cluster_cols = [col for col in df.columns if col not in ['VESSEL', 'CODE', 'VOY_OUT', 'ETA', 'TOTAL']]
    
    # Loop melalui setiap kolom cluster untuk mencari bentrokan
    for col in cluster_cols:
        # Ambil data di mana ada muatan di cluster ini (nilai > 0)
        subset_df = df_copy[df_copy[col] > 0]
        
        # Cari tanggal ETA yang duplikat (lebih dari 1 kapal di hari yang sama)
        clash_dates = subset_df[subset_df.duplicated(subset='ETA_Date', keep=False)]['ETA_Date'].unique()
        
        # Tandai sel yang bentrok di DataFrame styler
        for date in clash_dates:
            # Dapatkan baris/index dari kapal yang bentrok pada tanggal ini
            clashing_indices = subset_df[subset_df['ETA_Date'] == date].index
            # Beri warna pada sel yang bentrok
            styler.loc[clashing_indices, col] = 'background-color: #FFC700' # Warna kuning/gold
            
    return styler


# --- Sidebar untuk Input Pengguna ---
st.sidebar.header("⚙️ Unggah File Anda")
schedule_file = st.sidebar.file_uploader("1. Unggah File Jadwal Kapal", type=['xlsx', 'csv'])
unit_list_file = st.sidebar.file_uploader("2. Unggah File Daftar Unit", type=['xlsx', 'csv'])
process_button = st.sidebar.button("🚀 Proses Data", type="primary")

# --- Area Proses Utama ---
df_vessel_codes = load_vessel_codes_from_repo()

if process_button:
    if schedule_file and unit_list_file and (df_vessel_codes is not None and not df_vessel_codes.empty):
        with st.spinner('Memuat, memproses, dan mewarnai data...'):
            try:
                # 1. Memuat & Pra-proses
                df_schedule = pd.read_excel(schedule_file) if schedule_file.name.lower().endswith(('.xls', '.xlsx')) else pd.read_csv(schedule_file)
                df_unit_list = pd.read_excel(unit_list_file) if unit_list_file.name.lower().endswith(('.xls', '.xlsx')) else pd.read_csv(unit_list_file)
                original_vessels_list = df_schedule['VESSEL'].unique().tolist()
                df_schedule['ETA'] = pd.to_datetime(df_schedule['ETA'], errors='coerce')

                # 2. Penggabungan Data
                df_schedule_with_code = pd.merge(df_schedule, df_vessel_codes, left_on="VESSEL", right_on="Description", how="left").rename(columns={"Value": "CODE"})
                merged_df = pd.merge(df_schedule_with_code, df_unit_list, left_on=['CODE', 'VOY_OUT'], right_on=['Carrier Out', 'Voyage Out'], how='inner')

                # 3. Filter
                if not merged_df.empty:
                    merged_df = merged_df[merged_df['VESSEL'].isin(original_vessels_list)]
                    excluded_areas = [str(i) for i in range(801, 809)] 
                    merged_df['Area (EXE)'] = merged_df['Area (EXE)'].astype(str)
                    filtered_data = merged_df[~merged_df['Area (EXE)'].isin(excluded_areas)]

                    if filtered_data.empty:
                         st.warning("Setelah filter, tidak ada data yang tersisa untuk ditampilkan."); st.stop()

                    # 4. Transformasi ke Pivot
                    grouping_cols = ['VESSEL', 'CODE', 'VOY_OUT', 'ETA']
                    pivot_df = filtered_data.pivot_table(index=grouping_cols, columns='Area (EXE)', aggfunc='size', fill_value=0)
                    pivot_df['TOTAL'] = pivot_df.sum(axis=1)
                    pivot_df = pivot_df.reset_index()
                    
                    # 5. Filter Gabungan ETA & Total
                    two_days_ago = pd.Timestamp.now() - timedelta(days=2)
                    condition_to_hide = (pivot_df['ETA'] < two_days_ago) & (pivot_df['TOTAL'] < 50)
                    pivot_df = pivot_df[~condition_to_hide]
                    
                    if pivot_df.empty:
                         st.warning("Setelah filter ETA dan Total, tidak ada data tersisa."); st.stop()

                    # 6. Menata & Mengurutkan
                    cols_awal = ['VESSEL', 'CODE', 'VOY_OUT', 'ETA', 'TOTAL']
                    cols_clusters = [col for col in pivot_df.columns if col not in cols_awal]
                    final_display_cols = cols_awal + sorted(cols_clusters)
                    pivot_df = pivot_df[final_display_cols]
                    pivot_df = pivot_df.sort_values(by='ETA', ascending=True).reset_index(drop=True)

                    # 7. Menampilkan Hasil dengan Highlight
                    st.header("✅ Hasil Akhir (dengan Highlight Bentrokan Jadwal)")
                    st.success(f"Berhasil memproses dan mengelompokkan data untuk {len(pivot_df)} kapal unik (sesuai semua filter).")
                    
                    # Buat DataFrame Styler dengan menerapkan fungsi highlight
                    styled_df = pivot_df.style.apply(highlight_clashes, axis=None)
                    
                    # Atur format tanggal dan angka untuk ditampilkan
                    styled_df = styled_df.format({'ETA': lambda x: x.strftime('%Y-%m-%d %H:%M:%S')})
                    
                    st.dataframe(styled_df)
                    
                    csv_export = pivot_df.to_csv(index=False).encode('utf-8')
                    st.download_button(label="📥 Unduh Hasil sebagai CSV", data=csv_export, file_name='hasil_pivot_clusters.csv', mime='text/csv')
                else:
                    st.warning("Tidak ditemukan data yang cocok antara file Jadwal Kapal dan Daftar Unit.")
            except Exception as e:
                st.error(f"Terjadi kesalahan saat memproses file: {e}"); st.error(f"Detail error: {str(e)}")
    else:
        st.warning("Mohon unggah kedua file dan pastikan file kode kapal ada di repository sebelum memproses.")
