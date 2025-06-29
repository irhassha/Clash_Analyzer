import streamlit as st
import pandas as pd
import requests
import io

# --- Konfigurasi Halaman & Judul ---
st.set_page_config(page_title="App Pencocokan Vessel", layout="wide")
st.title("🚢 Aplikasi Pencocokan Jadwal Kapal & Unit List")
st.write("""
Aplikasi ini mengotomatiskan proses pencocokan data dari tiga sumber:
1.  **File Jadwal Kapal (Vessel Schedule):** Berisi jadwal kedatangan kapal.
2.  **File Daftar Unit (Unit List):** Berisi detail muatan per kapal.
3.  **File Kode Kapal (Vessel Codes):** File pemetaan dari nama kapal ke kode unik (diambil dari GitHub).
""")

# --- Fungsi untuk mengambil data dari GitHub ---
@st.cache_data
def load_data_from_url(url):
    """Mengambil dan memuat file CSV dari URL mentah GitHub ke dalam DataFrame."""
    try:
        # Memastikan URL adalah URL raw GitHub
        if "github.com" in url and "blob" in url:
            url = url.replace("blob/", "raw/")
            
        response = requests.get(url)
        response.raise_for_status()  # Cek jika ada error HTTP
        return pd.read_csv(io.StringIO(response.text))
    except requests.exceptions.RequestException as e:
        st.error(f"Error saat mengambil data dari URL: {e}")
    except Exception as e:
        st.error(f"Gagal memproses file dari URL. Pastikan file adalah CSV yang valid. Error: {e}")
    return None

# --- Sidebar untuk Input dari Pengguna ---
st.sidebar.header("⚙️ 1. Masukkan Data")

# Input URL untuk file Vessel Codes
github_url = st.sidebar.text_input(
    "URL file 'Vessel Codes' di GitHub (raw)",
    "https://raw.githubusercontent.com/USERNAME/REPOSITORY/main/vessel_codes.csv" # Ganti dengan URL Anda
)

# File Uploader
schedule_file = st.sidebar.file_uploader(
    "Unggah File Jadwal Kapal (Excel/CSV)", 
    type=['xlsx', 'csv']
)
unit_list_file = st.sidebar.file_uploader(
    "Unggah File Daftar Unit (Excel/CSV)", 
    type=['xlsx', 'csv']
)

# Tombol untuk memulai proses
process_button = st.sidebar.button("🚀 Proses Data", type="primary")


# --- Area Proses Utama ---
if process_button:
    # Validasi semua input sudah ada
    if schedule_file and unit_list_file and github_url:
        st.header("⏳ Sedang memproses...")

        # 1. Memuat semua data ke DataFrame
        try:
            # Baca file yang diunggah
            df_schedule = pd.read_excel(schedule_file) if schedule_file.name.endswith('.xlsx') else pd.read_csv(schedule_file)
            df_unit_list = pd.read_excel(unit_list_file) if unit_list_file.name.endswith('.xlsx') else pd.read_csv(unit_list_file)
            
            # Ambil data kode kapal dari GitHub
            df_vessel_codes = load_data_from_url(github_url)
            
            if df_vessel_codes is None:
                st.stop() # Hentikan eksekusi jika gagal mengambil data dari GitHub

            with st.expander("🔬 Lihat Data Awal yang Dimuat"):
                st.subheader("Data Jadwal Kapal")
                st.dataframe(df_schedule.head())
                st.subheader("Data Daftar Unit")
                st.dataframe(df_unit_list.head())
                st.subheader("Data Kode Kapal (dari GitHub)")
                st.dataframe(df_vessel_codes.head())

            # 2. Proses Penggabungan Data (sesuai logika yang sudah diperbaiki)
            st.header("🔄 Proses Penggabungan")

            # Langkah A: Gabungkan Jadwal Kapal dengan Kode Kapal untuk mendapatkan 'CODE'
            # Kita tidak menggunakan 'CODE' ini untuk join akhir, tapi bisa ditampilkan untuk referensi
            df_schedule_with_code = pd.merge(
                df_schedule,
                df_vessel_codes,
                left_on="VESSEL",
                right_on="Description",
                how="left"
            ).rename(columns={"Value": "CODE"})

            # Langkah B: Gabungkan hasil dengan Daftar Unit menggunakan 'LINE' dan 'VOY_OUT'
            # Ini adalah logika kunci yang kita perbaiki sebelumnya
            final_df = pd.merge(
                df_schedule_with_code,
                df_unit_list,
                left_on=['LINE', 'VOY_OUT'],
                right_on=['Carrier Out', 'Voyage Out'],
                how='inner' # 'inner' join hanya akan mengambil baris yang cocok di kedua file
            )

            # 3. Menampilkan Hasil
            st.header("✅ Hasil Akhir")

            if not final_df.empty:
                st.success(f"Berhasil menemukan {len(final_df)} baris data yang cocok!")

                # Menampilkan Area (EXE) unik
                st.subheader("Area (EXE) Unik yang Ditemukan:")
                unique_areas = final_df['Area (EXE)'].unique()
                st.write(list(unique_areas))

                # Menampilkan DataFrame hasil akhir
                st.subheader("Tabel Data Gabungan:")
                st.dataframe(final_df)

                # Tombol Download
                csv = final_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                   label="📥 Unduh Hasil sebagai CSV",
                   data=csv,
                   file_name='hasil_gabungan.csv',
                   mime='text/csv',
                )
            else:
                st.warning("Tidak ditemukan data yang cocok antara file Jadwal Kapal dan Daftar Unit berdasarkan 'LINE' dan 'VOY_OUT'.")

        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses file: {e}")
    else:
        st.warning("Mohon unggah kedua file dan pastikan URL GitHub sudah terisi untuk memulai proses.")
