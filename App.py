import streamlit as st
import pandas as pd
import requests
from datetime import datetime

# URL API yang menyediakan data jadwal dalam format JSON
API_URL = "https://www.npct1.co.id/api/vessel-schedules"

@st.cache_data(ttl=300) # Cache data selama 5 menit untuk performa
def scrape_vessel_schedule():
    """
    Fungsi ini mengambil data jadwal kapal langsung dari API NPCT1
    dan mengembalikannya sebagai sebuah Pandas DataFrame.
    """
    try:
        # Mengatur User-Agent untuk meniru browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Mengirim permintaan HTTP GET ke URL API
        response = requests.get(API_URL, headers=headers, timeout=15)
        response.raise_for_status()  # Cek jika ada error HTTP

        # Mengurai respons JSON menjadi list dari dictionary
        data = response.json()

        # Membuat DataFrame pandas langsung dari data JSON
        if not data:
            st.warning("API tidak mengembalikan data jadwal saat ini.")
            return pd.DataFrame()

        df = pd.DataFrame(data)
        
        # --- Pembersihan dan Pemformatan Data ---
        # Memilih kolom yang paling relevan untuk ditampilkan
        relevant_columns = {
            'vessel_name': 'Vessel Name',
            'voyage_no': 'Voyage No',
            'service_name': 'Service',
            'eta': 'ETA',
            'etd': 'ETD',
            'berthing_time': 'Berthing Time',
            'closing_time': 'Closing Time'
        }
        
        # Filter DataFrame agar hanya berisi kolom yang kita inginkan
        df_filtered = df[list(relevant_columns.keys())]
        
        # Mengganti nama kolom agar lebih mudah dibaca
        df_renamed = df_filtered.rename(columns=relevant_columns)

        # Mengonversi kolom tanggal ke format yang lebih rapi (contoh: 28 Jun 2024, 15:30)
        for col in ['ETA', 'ETD', 'Berthing Time', 'Closing Time']:
            # Menggunakan errors='coerce' akan mengubah nilai yang tidak valid menjadi NaT (Not a Time)
            df_renamed[col] = pd.to_datetime(df_renamed[col], errors='coerce').dt.strftime('%d %b %Y, %H:%M')

        return df_renamed

    except requests.exceptions.RequestException as e:
        st.error(f"Gagal terhubung ke API NPCT1: {e}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses data: {e}")
        return pd.DataFrame()

# --- Konfigurasi Tampilan Aplikasi Streamlit ---

# Mengatur judul tab browser dan layout halaman
st.set_page_config(page_title="Jadwal Kapal NPCT1", layout="wide")

# Judul utama aplikasi
st.title("🚢 Scraper Jadwal Kapal NPCT1")
st.markdown(f"Menampilkan data langsung dari API NPCT1.")

# Tombol untuk menyegarkan data secara manual
if st.button("🔄 Segarkan Data"):
    # Membersihkan cache agar fungsi scrape_vessel_schedule dijalankan kembali
    st.cache_data.clear()
    st.toast("Data sedang diperbarui dari API...")

# Memanggil fungsi untuk mendapatkan data
schedule_df = scrape_vessel_schedule()

# Menampilkan data jika berhasil di-scrape
if not schedule_df.empty:
    st.success("Data jadwal kapal berhasil dimuat.")
    # Menampilkan DataFrame di aplikasi
    st.dataframe(schedule_df, use_container_width=True, hide_index=True)
else:
    # Menampilkan pesan jika tidak ada data atau terjadi error
    st.warning("Tidak ada data untuk ditampilkan. Coba segarkan halaman atau periksa kembali nanti.")

st.markdown("---")
st.write("Dibuat dengan Streamlit dan Python.")
