import streamlit as st
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options

# URL Halaman Web (bukan lagi API)
PAGE_URL = "https://www.npct1.co.id/vessel-schedule"

@st.cache_data(ttl=600) # Cache data selama 10 menit
def get_schedule_with_selenium():
    """
    Mengambil jadwal kapal dengan mengotomatiskan browser Chrome menggunakan Selenium
    untuk melewati proteksi anti-scraping.
    """
    st.info("Menginisialisasi browser virtual untuk scraping...")
    
    # Mengatur opsi untuk menjalankan Chrome dalam mode 'headless' (tanpa UI)
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("window-size=1920x1080")

    # Menginstal dan mengelola ChromeDriver secara otomatis
    service = ChromeService(ChromeDriverManager().install())
    
    driver = None # Inisialisasi driver
    try:
        # Memulai driver browser Chrome
        driver = webdriver.Chrome(service=service, options=chrome_options)
        
        # Membuka halaman web
        st.write("Mengunjungi halaman jadwal kapal...")
        driver.get(PAGE_URL)

        # Menunggu hingga tabel dengan ID 'schedule' muncul dan terlihat
        # Ini adalah langkah krusial, menunggu maksimal 30 detik
        st.write("Menunggu data tabel dimuat oleh JavaScript...")
        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.ID, "schedule"))
        )
        st.write("Tabel ditemukan! Mengekstrak data...")

        # Setelah tabel dimuat, ambil source HTML halaman
        html_source = driver.page_source
        
        # Parsing HTML menggunakan BeautifulSoup
        soup = BeautifulSoup(html_source, 'html.parser')
        table = soup.find('table', id='schedule')

        if not table:
            st.error("Gagal menemukan tabel bahkan setelah menunggu.")
            return pd.DataFrame()

        # Mengekstrak header dan baris data (sama seperti metode awal)
        headers = [th.text.strip() for th in table.find('thead').find_all('th')]
        rows = []
        for row in table.find('tbody').find_all('tr'):
            cols = [td.text.strip() for td in row.find_all('td')]
            rows.append(cols)

        return pd.DataFrame(rows, columns=headers)

    except Exception as e:
        st.error(f"Terjadi kesalahan saat proses scraping dengan Selenium: {e}")
        return pd.DataFrame()
    finally:
        # Pastikan browser ditutup untuk membebaskan sumber daya
        if driver:
            driver.quit()
        st.info("Proses scraping selesai. Browser virtual telah ditutup.")


# --- Konfigurasi Tampilan Aplikasi Streamlit ---

st.set_page_config(page_title="Jadwal Kapal NPCT1", layout="wide")

st.title("🚢 Scraper Jadwal Kapal NPCT1 (Metode Selenium)")
st.markdown("Menggunakan otomasi browser untuk mengambil data dari situs yang dilindungi.")

if st.button("🔄 Ambil Data Terbaru"):
    st.cache_data.clear()
    st.toast("Memulai proses scraping baru...")

# Memanggil fungsi untuk mendapatkan data
schedule_df = get_schedule_with_selenium()

# Menampilkan data jika berhasil di-scrape
if schedule_df is not None and not schedule_df.empty:
    st.success("Data jadwal kapal berhasil dimuat!")
    st.dataframe(schedule_df, use_container_width=True, hide_index=True)
else:
    st.warning("Tidak ada data untuk ditampilkan atau terjadi kesalahan selama proses.")

st.markdown("---")
st.write("Dibuat dengan Streamlit, Selenium, dan Python.")
