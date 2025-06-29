import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup

# URL dari situs web target
URL = "https://www.npct1.co.id/vessel-schedule#schedule-tab"

@st.cache_data(ttl=600) # Menambahkan cache untuk performa, data akan di-refresh setiap 10 menit
def scrape_vessel_schedule():
    """
    Fungsi ini melakukan scraping data tabel dari URL NPCT1
    dan mengembalikannya sebagai sebuah Pandas DataFrame.
    """
    try:
        # Mengatur User-Agent untuk meniru browser agar tidak diblokir
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Mengirim permintaan HTTP GET ke URL
        response = requests.get(URL, headers=headers, timeout=15)
        response.raise_for_status()  # Cek jika ada error HTTP (spt 404, 500)

        # Parsing konten HTML dengan BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')

        # Mencari elemen tabel berdasarkan ID 'schedule'
        table = soup.find('table', id='schedule')

        # Jika tabel tidak ditemukan, kembalikan DataFrame kosong
        if not table:
            st.error("Tabel jadwal tidak dapat ditemukan di halaman web. Mungkin struktur web telah berubah.")
            return pd.DataFrame()

        # Mengekstrak header dari tabel (tag <th>)
        table_headers = [th.text.strip() for th in table.find('thead').find_all('th')]

        # Mengekstrak semua baris data dari body tabel (tag <tr> di dalam <tbody>)
        table_rows = []
        for row in table.find('tbody').find_all('tr'):
            # Mengekstrak semua sel data (tag <td>) dalam satu baris
            cols = [td.text.strip() for td in row.find_all('td')]
            table_rows.append(cols)

        # Membuat DataFrame pandas dari header dan data baris
        df = pd.DataFrame(table_rows, columns=table_headers)
        return df

    except requests.exceptions.RequestException as e:
        st.error(f"Gagal terhubung ke URL: {e}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses data: {e}")
        return pd.DataFrame()

# --- Konfigurasi Tampilan Aplikasi Streamlit ---

# Mengatur judul tab browser dan layout halaman
st.set_page_config(page_title="Jadwal Kapal NPCT1", layout="wide")

# Judul utama aplikasi
st.title("🚢 Scraper Jadwal Kapal NPCT1")
st.markdown(f"Menampilkan data langsung dari sumber: [npct1.co.id]({URL})")

# Tombol untuk menyegarkan data secara manual
if st.button("🔄 Segarkan Data"):
    # Membersihkan cache agar fungsi scrape_vessel_schedule dijalankan kembali
    st.cache_data.clear()
    st.toast("Data sedang diperbarui...")

# Memanggil fungsi untuk mendapatkan data
schedule_df = scrape_vessel_schedule()

# Menampilkan data jika berhasil di-scrape
if not schedule_df.empty:
    st.success("Data jadwal kapal berhasil dimuat.")
    # Menampilkan DataFrame di aplikasi
    st.dataframe(schedule_df, use_container_width=True)
else:
    # Menampilkan pesan jika tidak ada data atau terjadi error
    st.warning("Tidak ada data untuk ditampilkan. Coba segarkan halaman atau periksa kembali nanti.")

st.markdown("---")
st.write("Dibuat dengan Streamlit dan Python.")

