import streamlit as st
import pandas as pd
import requests
import io

# --- Konfigurasi Halaman & Judul ---
st.set_page_config(page_title="App Pencocokan Vessel", layout="wide")
st.title("🚢 Aplikasi Pencocokan Jadwal Kapal & Unit List")
st.write("""
Aplikasi ini mengotomatiskan proses pencocokan data dari tiga sumber:
1.  **File Jadwal Kapal (Vessel Schedule)**
2.  **File Daftar Unit (Unit List)**
3.  **File Kode Kapal (Vessel Codes)**
""")

# --- Fungsi untuk mengambil data dari URL (bisa CSV/Excel) ---
@st.cache_data # Cache data agar tidak perlu download ulang setiap ada interaksi
def load_data_from_url(url):
    """Mengambil dan memuat file CSV atau Excel dari URL."""
    try:
        if not url or not url.startswith('http'):
            st.error("URL tidak valid. Harap masukkan URL yang benar.")
            return None
            
        # Memastikan URL adalah URL raw GitHub jika dari GitHub
        if "github.com" in url and "blob" in url:
            url = url.replace("blob/", "raw/")
            
        response = requests.get(url)
        response.raise_for_status()  # Cek jika ada error HTTP

        # Cek tipe file dari URL untuk pembacaan yang benar
        if url.lower().endswith('.csv'):
            return pd.read_csv(io.StringIO(response.text))
        elif url.lower().endswith(('.xls', '.xlsx')):
            return pd.read_excel(io.BytesIO(response.content))
        else:
            # Jika ekstensi tidak jelas, coba baca sebagai Excel dulu, lalu CSV
            try:
                return pd.read_excel(io.BytesIO(response.content))
            except Exception:
                return pd.read_csv(io.StringIO(response.text))

    except requests.exceptions.RequestException as e:
        st.error(f"Error saat mengambil data dari URL: {e}")
    except Exception as e:
        st.error(f"Gagal memproses file dari URL. Pastikan file dan URL valid. Error: {e}")
    return None

# --- Sidebar untuk Input dari Pengguna ---
st.sidebar.header("⚙️ 1. Unggah File Utama")
schedule_file = st.sidebar.file_uploader("Unggah Jadwal Kapal (Excel/CSV)", type=['xlsx', 'csv'])
unit_list_file = st.sidebar.file_uploader("Unggah Daftar Unit (Excel/CSV)", type=['xlsx', 'csv'])

st.sidebar.header("⚙️ 2. Masukkan Data Kode Kapal")
source_option = st.sidebar.radio(
    "Pilih sumber file 'Vessel Codes':",
    ('Unggah File Langsung', 'Ambil dari URL')
)

# Variabel untuk menampung DataFrame kode kapal
df_vessel_codes = None

if source_option == 'Unggah File Langsung':
    vessel_code_file = st.sidebar.file_uploader(
        "Unggah File Vessel Code (Excel/CSV)",
        type=['xlsx', 'xls', 'csv']
    )
    if vessel_code_file:
        try:
            # Baca file sesuai ekstensinya
            if vessel_code_file.name.lower().endswith('.csv'):
                df_vessel_codes = pd.read_csv(vessel_code_file)
            else:
                df_vessel_codes = pd.read_excel(vessel_code_file)
        except Exception as e:
            st.error(f"Gagal membaca file Vessel Code yang diunggah: {e}")

else: # Opsi 'Ambil dari URL'
    github_url = st.sidebar.text_input(
        "URL file 'Vessel Codes' (raw GitHub atau link langsung)",
        ""
    )
    if github_url:
        df_vessel_codes = load_data_from_url(github_url)

# Tombol untuk memulai proses
process_button = st.sidebar.button("🚀 Proses Data", type="primary")

# --- Area Proses Utama ---
if process_button:
    # Validasi semua input sudah ada
    if schedule_file and unit_list_file and (df_vessel_codes is not None and not df_vessel_codes.empty):
        with st.spinner('Memuat dan memproses file...'):
            try:
                # 1. Memuat file utama
                df_schedule = pd.read_excel(schedule_file) if schedule_file.name.lower().endswith(('.xls', '.xlsx')) else pd.read_csv(schedule_file)
                df_unit_list = pd.read_excel(unit_list_file) if unit_list_file.name.lower().endswith(('.xls', '.xlsx')) else pd.read_csv(unit_list_file)

                # Tampilkan data awal yang berhasil dimuat
                with st.expander("🔬 Lihat Data Awal yang Dimuat"):
                    st.subheader("Data Jadwal Kapal")
                    st.dataframe(df_schedule.head())
                    st.subheader("Data Daftar Unit")
                    st.dataframe(df_unit_list.head())
                    st.subheader("Data Kode Kapal")
                    st.dataframe(df_vessel_codes.head())

                # 2. Proses Penggabungan Data
                st.header("🔄 Proses Penggabungan")

                # Langkah A: Gabungkan Jadwal Kapal dengan Kode Kapal 
                df_schedule_with_code = pd.merge(
                    df_schedule,
                    df_vessel_codes,
                    left_on="VESSEL",
                    right_on="Description",
                    how="left"
                ).rename(columns={"Value": "CODE"})

                # Langkah B: Gabungkan hasil dengan Daftar Unit menggunakan 'LINE' dan 'VOY_OUT'
                final_df = pd.merge(
                    df_schedule_with_code,
                    df_unit_list,
                    left_on=['LINE', 'VOY_OUT'],
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
        st.warning("Mohon pastikan kedua file utama telah diunggah dan data Kode Kapal sudah tersedia (baik dari unggahan maupun URL) sebelum memproses.")
