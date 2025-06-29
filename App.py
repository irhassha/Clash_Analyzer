import streamlit as st
import pandas as pd
import os

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
    """Mencari dan memuat file kode kapal dari daftar nama file yang mungkin."""
    for filename in possible_names:
        if os.path.exists(filename):
            try:
                if filename.lower().endswith('.csv'):
                    return pd.read_csv(filename)
                else:
                    return pd.read_excel(filename)
            except Exception as e:
                st.error(f"Gagal membaca file '{filename}': {e}")
                return None
    
    st.error(f"File kode kapal tidak ditemukan. Pastikan ada file dengan nama vessel_codes.xlsx atau .csv di repository Anda.")
    return None

# --- Sidebar untuk Input Pengguna ---
st.sidebar.header("⚙️ Unggah File Anda")
schedule_file = st.sidebar.file_uploader("1. Unggah File Jadwal Kapal", type=['xlsx', 'csv'])
unit_list_file = st.sidebar.file_uploader("2. Unggah File Daftar Unit", type=['xlsx', 'csv'])

process_button = st.sidebar.button("🚀 Proses Data", type="primary")


# --- Area Proses Utama ---
df_vessel_codes = load_vessel_codes_from_repo()

if process_button:
    if schedule_file and unit_list_file and (df_vessel_codes is not None and not df_vessel_codes.empty):
        with st.spinner('Memuat dan memproses file...'):
            try:
                # 1. Memuat file
                df_schedule = pd.read_excel(schedule_file) if schedule_file.name.lower().endswith(('.xls', '.xlsx')) else pd.read_csv(schedule_file)
                df_unit_list = pd.read_excel(unit_list_file) if unit_list_file.name.lower().endswith(('.xls', '.xlsx')) else pd.read_csv(unit_list_file)

                # 2. Proses Penggabungan Data
                # Langkah A: Gabungkan Jadwal dengan Kode
                df_schedule_with_code = pd.merge(
                    df_schedule, df_vessel_codes,
                    left_on="VESSEL", right_on="Description",
                    how="left"
                ).rename(columns={"Value": "CODE"})

                # Langkah B: Gabungkan dengan Daftar Unit
                final_df = pd.merge(
                    df_schedule_with_code, df_unit_list,
                    left_on=['CODE', 'VOY_OUT'],
                    right_on=['Carrier Out', 'Voyage Out'],
                    how='inner'
                )

                # 3. Transformasi Data ke Format Pivot (Perubahan Kunci Ada di Sini)
                st.header("✅ Hasil Akhir (Format Pivot)")
                if not final_df.empty:
                    # Tentukan kolom-kolom utama untuk pengelompokan
                    grouping_cols = ['VESSEL', 'CODE', 'VOY_OUT', 'ETA']
                    
                    # Buat tabel pivot untuk menghitung jumlah tiap Area (EXE)
                    pivot_df = final_df.pivot_table(
                        index=grouping_cols,
                        columns='Area (EXE)',
                        aggfunc='size', # Menghitung jumlah kemunculan
                        fill_value=0
                    )
                    
                    # Hitung kolom TOTAL
                    pivot_df['TOTAL'] = pivot_df.sum(axis=1)
                    
                    # Reset index agar kolom grouping kembali menjadi kolom biasa
                    pivot_df = pivot_df.reset_index()

                    # Susun ulang urutan kolom sesuai permintaan
                    cols_awal = ['VESSEL', 'CODE', 'VOY_OUT', 'ETA', 'TOTAL']
                    # Ambil nama-nama kolom cluster (semua kolom kecuali yang awal)
                    cols_clusters = [col for col in pivot_df.columns if col not in cols_awal]
                    
                    # Gabungkan urutan kolom
                    final_display_cols = cols_awal + sorted(cols_clusters)
                    pivot_df = pivot_df[final_display_cols]

                    st.success(f"Berhasil memproses dan mengelompokkan data untuk {len(pivot_df)} kapal unik.")
                    
                    st.dataframe(pivot_df)
                    
                    csv_export = pivot_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                       label="📥 Unduh Hasil sebagai CSV",
                       data=csv_export,
                       file_name='hasil_pivot_clusters.csv',
                       mime='text/csv',
                    )
                else:
                    st.warning("Tidak ditemukan data yang cocok antara file Jadwal Kapal dan Daftar Unit.")

            except Exception as e:
                st.error(f"Terjadi kesalahan saat memproses file: {e}")
                # Tambahkan detail error untuk debugging
                st.error(f"Detail error: {str(e)}")
    else:
        st.warning("Mohon unggah kedua file dan pastikan file kode kapal ada di repository sebelum memproses.")
