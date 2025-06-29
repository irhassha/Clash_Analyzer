import streamlit as st
import pandas as pd
import requests
from datetime import datetime

# URL API BARU yang menyediakan data jadwal dalam format JSON
# Endpoint lama (vessel-schedules) tidak lagi valid (404 Not Found).
API_URL = "https://www.npct1.co.id/api/vessel/schedule"

@st.cache_data(ttl=300) # Cache data selama 5 menit untuk performa
def get_vessel_schedule():
    """
    Fungsi ini mengambil data jadwal kapal langsung dari API NPCT1 yang baru
    dan mengembalikannya sebagai sebuah Pandas DataFrame.
    """
    try:
        # Mengatur User-Agent untuk meniru browser agar tidak diblokir
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/json'
        }
        
        # Mengirim permintaan HTTP GET ke URL API
        response = requests.get(API_URL, headers=headers, timeout=15)
        response.raise_for_status()  # Cek jika ada error HTTP (spt 404, 500)

        # Mengurai respons JSON
        data = response.json()

        # Membuat DataFrame pandas langsung dari data JSON
        if not data:
            st.warning("API tidak mengembalikan data jadwal saat ini.")
            return pd.DataFrame()

        df = pd.DataFrame(data)
        
        # --- Pembersihan dan Pemformatan Data ---
        # Memilih kolom yang paling relevan untuk ditampilkan
        # Kunci di kiri adalah nama kolom dari API, nilai di kanan adalah header tabel yang akan ditampilkan
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
        # Cek kolom mana saja dari 'relevant_columns' yang benar-benar ada di DataFrame
        existing_columns = {k: v for k, v in relevant_columns.items() if k in df.columns}
        
        if not existing_columns:
            st.error("Struktur data dari API telah berubah dan kolom yang diharapkan tidak ditemukan.")
            return pd.DataFrame()
            
        df_filtered = df[list(existing_columns.keys())]
        
        # Mengganti nama kolom agar lebih mudah dibaca
        df_renamed = df_filtered.rename(columns=existing_columns)

        # Mengonversi kolom tanggal ke format yang lebih rapi (contoh: 28 Jun 2024, 15:30)
        date_cols = ['ETA', 'ETD', 'Berthing Time', 'Closing Time']
        for col in date_cols:
            if col in df_renamed.columns:
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

st.set_page_config(page_title="Jadwal Kapal NPCT1", layout="wide")

st.title("🚢 Scraper Jadwal Kapal NPCT1")
st.markdown("Menampilkan data langsung dari API NPCT1.")

if st.button("🔄 Segarkan Data"):
    st.cache_data.clear()
    st.toast("Data sedang diperbarui dari API...")

# Memanggil fungsi untuk mendapatkan data
schedule_df = get_vessel_schedule()

# Menampilkan data jika berhasil di-scrape
if schedule_df is not None and not schedule_df.empty:
    st.success("Data jadwal kapal berhasil dimuat.")
    st.dataframe(schedule_df, use_container_width=True, hide_index=True)
else:
    st.info("Tidak ada data untuk ditampilkan atau terjadi kesalahan saat mengambil data.")

st.markdown("---")
st.write("Dibuat dengan Streamlit dan Python.")

