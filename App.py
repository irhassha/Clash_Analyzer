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
@st.cache_data # Cache data agar tidak perlu dibaca ulang setiap ada interaksi
def load_vessel_codes_from_repo(possible_names=['vessel codes.xlsx', 'vessel_codes.xls', 'vessel_codes.csv']):
    """Mencari dan memuat file kode kapal dari daftar nama file yang mungkin."""
    for filename in possible_names:
        if os.path.exists(filename):
            try:
                if filename.lower().endswith('.csv'):
                    return pd.read_csv(filename)
                else:
                    # Menggunakan openpyxl secara implisit
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

# Langsung muat data kode kapal saat aplikasi dimulai
df_vessel_codes = load_vessel_codes_from_repo()

if process_button:
    # Validasi input
    if schedule_file and unit_list_file and (df_vessel_codes is not None and not df_vessel_codes.empty):
        with st.spinner('Memuat dan memproses file...'):
            try:
                # 1. Memuat file yang diunggah
                df_schedule = pd.read_excel(schedule_file) if schedule_file.name.lower().endswith(('.xls', '.xlsx')) else pd.read_csv(schedule_file)
                df_unit_list = pd.read_excel(unit_list_file) if unit_list_file.name.lower().endswith(('.xls', '.xlsx')) else pd.read_csv(unit_list_file)

                with st.expander("🔬 Lihat Data Awal yang Dimuat"):
                    st.subheader("Data Jadwal Kapal (terunggah)")
                    st.dataframe(df_schedule.head())
                    st.subheader("Data Daftar Unit (terunggah)")
                    st.dataframe(df_unit_list.head())
                    st.subheader("Data Kode Kapal (dari repository)")
                    st.dataframe(df_vessel_codes.head())

                # 2. Proses Penggabungan Data (Logika Inti)
                st.header("🔄 Proses Penggabungan")
                
                # Langkah A: Gabungkan Jadwal Kapal dengan Kode Kapal untuk mendapatkan kolom 'CODE'
                df_schedule_with_code = pd.merge(
                    df_schedule,
                    df_vessel_codes,
                    left_on="VESSEL",
                    right_on="Description",
                    how="left"
                ).rename(columns={"Value": "CODE"})

                # Langkah B: Gabungkan hasil dengan Daftar Unit
                # --- PERUBAHAN LOGIKA KUNCI JOIN ADA DI SINI ---
                final_df = pd.merge(
                    df_schedule_with_code,
                    df_unit_list,
                    left_on=['CODE', 'VOY_OUT'], # Diubah dari ['LINE', 'VOY_OUT'] sesuai revisi
                    right_on=['Carrier Out', 'Voyage Out'],
                    how='inner'
                )

                # 3. Menampilkan Hasil
                st.header("✅ Hasil Akhir")
                if not final_df.empty:
                    st.success(f"Berhasil menemukan {len(final_df)} baris data yang cocok!")
                    st.subheader("Area (EXE) Unik yang Ditemukan:")
                    unique_areas = final_df['Area (EXE)'].unique()
                    st.write(list(unique_areas))
                    
                    st.subheader("Tabel Data Gabungan:")
                    st.dataframe(final_df)
                    
                    csv_export = final_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                       label="📥 Unduh Hasil sebagai CSV",
                       data=csv_export,
                       file_name='hasil_gabungan.csv',
                       mime='text/csv',
                    )
                else:
                    st.warning("Tidak ditemukan data yang cocok antara file Jadwal Kapal dan Daftar Unit menggunakan 'CODE' dan 'VOY_OUT'.")

            except Exception as e:
                st.error(f"Terjadi kesalahan saat memproses file: {e}")
    else:
        st.warning("Mohon unggah kedua file (Jadwal Kapal dan Daftar Unit) dan pastikan file kode kapal ada di repository sebelum memproses.")
