import streamlit as st
import pandas as pd
import os
from datetime import datetime, timedelta
import json
import io
import warnings
import numpy as np
import plotly.express as px
from itertools import combinations

# --- Library untuk Machine Learning ---
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

# --- Library untuk Tabel Interaktif ---
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode

# Mengabaikan peringatan yang tidak kritikal
warnings.filterwarnings("ignore", category=UserWarning)

# --- Konfigurasi Halaman & Judul ---
st.set_page_config(page_title="Yard Cluster Monitoring", layout="wide")
st.title("Yard Cluster Monitoring")

# --- PERBAIKAN: CSS Kustom untuk Merapatkan Spasi ---
st.markdown("""
    <style>
        /* Mengurangi margin atas dan bawah untuk garis pemisah */
        hr {
            margin-top: 10px;
            margin-bottom: 10px;
        }
        /* Mengurangi margin atas dan bawah untuk subheader */
        h2, h3 {
            margin-top: 20px;
            margin-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)


# --- Fungsi untuk mereset data di memori ---
def reset_data():
    """Membersihkan data relevan dari session state dan cache."""
    keys_to_clear = ['processed_df', 'clash_summary_df', 'summary_display', 'vessel_area_slots', 'forecast_df']
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    st.cache_data.clear()
    st.cache_resource.clear()
    st.success("Data berhasil direset!")


# --- FUNGSI UNTUK FORECASTING ---
@st.cache_data
def load_history_data(filename="History Loading.xlsx"):
    """Mencari dan memuat file data historis untuk peramalan."""
    if os.path.exists(filename):
        try:
            df = pd.read_excel(filename) if filename.lower().endswith('.xlsx') else pd.read_csv(filename)
            df.columns = [col.strip().lower() for col in df.columns]
            df['ata'] = pd.to_datetime(df['ata'], dayfirst=True, errors='coerce')
            df.dropna(subset=['ata', 'loading', 'service'], inplace=True)
            return df[df['loading'] >= 0]
        except Exception as e:
            st.error(f"Gagal memuat file histori '{filename}': {e}")
            return None
    st.warning(f"File histori '{filename}' tidak ditemukan.")
    return None

def create_time_features(df):
    """Membuat fitur berbasis waktu dari kolom 'ata'."""
    df_copy = df.copy()
    df_copy['hour'] = df_copy['ata'].dt.hour
    df_copy['day_of_week'] = df_copy['ata'].dt.dayofweek
    df_copy['day_of_month'] = df_copy['ata'].dt.day
    df_copy['day_of_year'] = df_copy['ata'].dt.dayofyear
    df_copy['week_of_year'] = df_copy['ata'].dt.isocalendar().week.astype(int)
    df_copy['month'] = df_copy['ata'].dt.month
    df_copy['year'] = df_copy['ata'].dt.year
    return df_copy

@st.cache_data
def run_per_service_rf_forecast(_df_history):
    """Menjalankan proses pembersihan outlier dan peramalan untuk setiap service."""
    all_results = []
    if _df_history is None or _df_history.empty: return pd.DataFrame(all_results)
    unique_services = _df_history['service'].unique()
    progress_bar = st.progress(0, text="Menganalisis services...")
    for i, service in enumerate(unique_services):
        progress_bar.progress((i + 1) / len(unique_services), text=f"Menganalisis service: {service}")
        service_df = _df_history[_df_history['service'] == service].copy()
        if service_df.empty or service_df['loading'].isnull().all():
            continue
        Q1, Q3 = service_df['loading'].quantile(0.25), service_df['loading'].quantile(0.75)
        IQR = Q3 - Q1
        upper_bound, lower_bound = Q3 + 1.5 * IQR, Q1 - 1.5 * IQR
        num_outliers = ((service_df['loading'] < lower_bound) | (service_df['loading'] > upper_bound)).sum()
        service_df['loading_cleaned'] = service_df['loading'].clip(lower=lower_bound, upper=upper_bound)
        
        forecast_val, moe_val, mape_val, method = (0, 0, 0, "")
        if len(service_df) >= 10:
            try:
                df_features = create_time_features(service_df)
                features_to_use = ['hour', 'day_of_week', 'day_of_month', 'day_of_year', 'week_of_year', 'month', 'year']
                X, y = df_features[features_to_use], df_features['loading_cleaned']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
                if len(X_train) == 0: raise ValueError("Data tidak cukup untuk training.")
                model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1, min_samples_leaf=2)
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                mape_val = mean_absolute_percentage_error(y_test, predictions) * 100 if len(y_test) > 0 else 0
                moe_val = 1.96 * np.std(y_test - predictions) if len(y_test) > 0 else 0
                future_eta = datetime.now().replace(hour=12, minute=0, second=0, microsecond=0) + timedelta(days=1)
                future_df = create_time_features(pd.DataFrame([{'ata': future_eta}]))
                forecast_val = model.predict(future_df[features_to_use])[0]
                method = f"Random Forest ({num_outliers} outliers dibersihkan)"
            except Exception:
                forecast_val, moe_val, mape_val, method = (service_df['loading_cleaned'].mean(), 1.96 * service_df['loading_cleaned'].std(), np.mean(np.abs((service_df['loading_cleaned'] - service_df['loading_cleaned'].mean()) / service_df['loading_cleaned'])) * 100 if not service_df['loading_cleaned'].empty else 0, f"Rata-rata Historis (RF Gagal, {num_outliers} outliers dibersihkan)")
        else:
            forecast_val, moe_val, mape_val, method = (service_df['loading_cleaned'].mean(), 1.96 * service_df['loading_cleaned'].std(), np.mean(np.abs((service_df['loading_cleaned'] - service_df['loading_cleaned'].mean()) / service_df['loading_cleaned'])) * 100 if not service_df['loading_cleaned'].empty else 0, f"Rata-rata Historis ({num_outliers} outliers dibersihkan)")
        all_results.append({"Service": service, "Loading Forecast": max(0, forecast_val), "Margin of Error (± box)": moe_val, "MAPE (%)": mape_val, "Method": method})
    progress_bar.empty()
    return pd.DataFrame(all_results)

@st.cache_data
def load_vessel_codes_from_repo(possible_names=['vessel codes.xlsx', 'vessel_codes.xls', 'vessel_codes.csv']):
    """Mencari dan memuat file kode kapal."""
    for filename in possible_names:
        if os.path.exists(filename):
            try:
                df = pd.read_excel(filename) if filename.lower().endswith(('.xls', '.xlsx')) else pd.read_csv(filename)
                df.columns = [col.strip() for col in df.columns]
                return df
            except Exception as e:
                st.error(f"Gagal membaca file '{filename}': {e}"); return None
    st.error(f"File kode kapal tidak ditemukan."); return None


# --- FUNGSI UNTUK MERENDER SETIAP TAB ---

def render_forecast_tab():
    """Fungsi untuk menampilkan seluruh konten tab peramalan."""
    st.header("📈 Peramalan Muatan dengan Machine Learning")
    st.write("Fitur ini menggunakan model **Random Forest** terpisah untuk setiap *service* guna memberikan prediksi yang lebih akurat.")
    if 'forecast_df' not in st.session_state:
        df_history = load_history_data()
        if df_history is not None and not df_history.empty:
            with st.spinner("Memproses data dan melatih model..."):
                st.session_state['forecast_df'] = run_per_service_rf_forecast(df_history)
        else:
            st.session_state['forecast_df'] = pd.DataFrame()
            if df_history is None: st.warning("File histori tidak ditemukan. Peramalan tidak tersedia.")
    
    if 'forecast_df' in st.session_state and not st.session_state.forecast_df.empty:
        results_df = st.session_state.forecast_df.copy()
        results_df['Loading Forecast'] = results_df['Loading Forecast'].round(2)
        results_df['Margin of Error (± box)'] = results_df['Margin of Error (± box)'].fillna(0).round(2)
        results_df['MAPE (%)'] = results_df['MAPE (%)'].replace([np.inf, -np.inf], 0).fillna(0).round(2)
                
        st.markdown("---")
        st.subheader("📊 Hasil Peramalan per Service")
        filter_option = st.radio("Filter Services:", ("All Services", "Current Services"), horizontal=True, key="forecast_filter")
        current_services_list = ['JPI-A', 'JPI-B', 'CIT', 'IN1', 'JKF', 'IN1-2', 'KCI', 'CMI3', 'CMI2', 'CMI', 'I15', 'SE8', 'IA8', 'IA1', 'SEAGULL', 'JTH', 'ICN']
        display_forecast_df = results_df[results_df['Service'].isin(current_services_list)] if filter_option == "Current Services" else results_df
        
        st.dataframe(display_forecast_df.sort_values(by="Loading Forecast", ascending=False).reset_index(drop=True), use_container_width=True, hide_index=True, column_config={"MAPE (%)": st.column_config.NumberColumn(format="%.2f%%")})
        st.markdown("---")
        st.subheader("💡 Cara Membaca Hasil Ini")
        st.markdown("- **Loading Forecast**: Estimasi jumlah box untuk kedatangan kapal berikutnya dari service tersebut.\n- **Margin of Error (± box)**: Tingkat ketidakpastian dalam prediksi. Contoh: 300 ±50 berarti nilai sebenarnya kemungkinan antara 250 dan 350.\n- **MAPE (%)**: Rata-rata persentase kesalahan model. **Semakin kecil, semakin baik.**\n- **Method**: Teknik yang digunakan untuk peramalan.")
    else:
        st.warning("Tidak ada data peramalan yang dapat dihasilkan.")

def render_clash_tab(process_button, schedule_file, unit_list_file, min_clash_distance):
    """Fungsi untuk menampilkan seluruh konten tab Analisis Bentrok."""
    
    # Inisialisasi session state jika belum ada
    for key in ['processed_df', 'summary_display', 'vessel_area_slots', 'clash_summary_df']:
        if key not in st.session_state: st.session_state[key] = None

    df_vessel_codes = load_vessel_codes_from_repo()
    
    if process_button:
        if schedule_file and unit_list_file and (df_vessel_codes is not None and not df_vessel_codes.empty):
            with st.spinner('Memuat dan memproses data...'):
                try:
                    # 1. PEMROSESAN DATA DASAR
                    df_schedule = pd.read_excel(schedule_file) if schedule_file.name.lower().endswith('.xlsx') else pd.read_csv(schedule_file)
                    df_schedule.columns = [col.strip().upper() for col in df_schedule.columns]
                    
                    df_unit_list = pd.read_excel(unit_list_file) if unit_list_file.name.lower().endswith('.xlsx') else pd.read_csv(unit_list_file)
                    df_unit_list.columns = [col.strip() for col in df_unit_list.columns]

                    for col in ['ETA', 'ETD', 'CLOSING PHYSIC']:
                        if col in df_schedule.columns: df_schedule[col] = pd.to_datetime(df_schedule[col], dayfirst=True, errors='coerce')
                    df_schedule.dropna(subset=['ETA', 'ETD'], inplace=True)
                    
                    # 2. EKSTRAKSI DATA SLOT (PENTING UNTUK DETEKSI BENTROK DETAIL)
                    if 'Row/bay (EXE)' not in df_unit_list.columns:
                        st.error("File 'Unit List' harus memiliki kolom 'Row/bay (EXE)' untuk deteksi bentrok detail."); st.stop()
                    df_unit_list['SLOT'] = df_unit_list['Row/bay (EXE)'].astype(str).str.split('-').str[-1]
                    df_unit_list['SLOT'] = pd.to_numeric(df_unit_list['SLOT'], errors='coerce')
                    df_unit_list.dropna(subset=['SLOT'], inplace=True)
                    df_unit_list['SLOT'] = df_unit_list['SLOT'].astype(int)

                    # 3. PENGGABUNGAN DATA
                    df_schedule_with_code = pd.merge(df_schedule, df_vessel_codes, left_on="VESSEL", right_on="Description", how="left").rename(columns={"Value": "CODE"})
                    merged_df = pd.merge(df_schedule_with_code, df_unit_list, left_on=['CODE', 'VOY_OUT'], right_on=['Carrier Out', 'Voyage Out'], how='inner')
                    if merged_df.empty: st.warning("Tidak ada data yang cocok ditemukan."); st.stop()
                        
                    # Menghapus filter area agar semua area ditampilkan
                    merged_df['Area (EXE)'] = merged_df['Area (EXE)'].astype(str)
                    
                    # 4. AGREGRASI DATA UNTUK TABEL & DETEKSI BENTROK
                    # Data untuk tabel pivot (tampilan utama)
                    pivot_for_display = merged_df.pivot_table(index=['VESSEL', 'VOY_OUT', 'ETA', 'ETD', 'SERVICE', 'CLOSING PHYSIC'], columns='Area (EXE)', fill_value=0, aggfunc='size')
                    pivot_for_display['TOTAL BOX'] = pivot_for_display.sum(axis=1)
                    pivot_for_display['TOTAL CLSTR'] = (pivot_for_display[[c for c in pivot_for_display.columns if c not in ['TOTAL BOX']]] > 0).sum(axis=1)
                    pivot_for_display.reset_index(inplace=True)
                    st.session_state.processed_df = pivot_for_display.sort_values(by='ETA', ascending=True)

                    # Data untuk deteksi bentrok detail (dengan MIN/MAX SLOT)
                    vessel_area_slots = merged_df.groupby(['VESSEL', 'VOY_OUT', 'ETA', 'ETD', 'SERVICE', 'Area (EXE)']).agg(MIN_SLOT=('SLOT', 'min'), MAX_SLOT=('SLOT', 'max'), BOX_COUNT=('SLOT', 'count')).reset_index()
                    st.session_state.vessel_area_slots = vessel_area_slots
                    
                    st.success("Data berhasil diproses!")
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat pemrosesan: {e}"); st.session_state.processed_df = None
        else:
            st.warning("Mohon unggah kedua file (Schedule dan Unit List).")                    

    # --- TAMPILAN UTAMA SETELAH DATA DIPROSES ---
    if st.session_state.get('processed_df') is not None:
        display_df = st.session_state.processed_df.copy()
        display_df['ETA_Display'] = display_df['ETA'].dt.strftime('%d/%m/%Y %H:%M')
        display_df['ETA_Date'] = display_df['ETA'].dt.strftime('%d/%m/%Y')

        # --- RINGKASAN KAPAL YANG AKAN DATANG ---
        st.subheader("🚢 Ringkasan Kapal Datang (Hari Ini + 3 Hari ke Depan)")
        forecast_df = st.session_state.get('forecast_df')
        if forecast_df is not None and not forecast_df.empty:
            today = pd.to_datetime(datetime.now().date())
            four_days_later = today + timedelta(days=4)
            upcoming_vessels_df = display_df[(display_df['ETA'] >= today) & (display_df['ETA'] < four_days_later)].copy()
            if not upcoming_vessels_df.empty:
                st.sidebar.markdown("---")
                st.sidebar.header("🛠️ Opsi Kapal Datang")
                priority_vessels = st.sidebar.multiselect("Pilih kapal prioritas untuk ditandai:", options=upcoming_vessels_df['VESSEL'].unique())
                adjusted_clstr_req = st.sidebar.number_input("Sesuaikan CLSTR REQ untuk kapal prioritas:", min_value=0, value=0, step=1, help="Masukkan nilai baru untuk CLSTR REQ. Biarkan 0 untuk tidak mengubah.")
                
                summary_df = pd.merge(upcoming_vessels_df, forecast_df[['Service', 'Loading Forecast']], left_on='SERVICE', right_on='Service', how='left')
                summary_df['Loading Forecast'] = summary_df['Loading Forecast'].fillna(0).round(0).astype(int)
                summary_df['DIFF'] = summary_df['TOTAL BOX'] - summary_df['Loading Forecast']
                summary_df['base_for_req'] = summary_df[['TOTAL BOX', 'Loading Forecast']].max(axis=1)
                summary_df['CLSTR REQ'] = summary_df['base_for_req'].apply(lambda v: 4 if v <= 450 else (5 if v <= 600 else (6 if v <= 800 else 8)))
                if priority_vessels and adjusted_clstr_req > 0:
                    summary_df.loc[summary_df['VESSEL'].isin(priority_vessels), 'CLSTR REQ'] = adjusted_clstr_req
                
                summary_display_cols = ['VESSEL', 'SERVICE', 'ETA', 'CLOSING PHYSIC', 'TOTAL BOX', 'Loading Forecast', 'DIFF', 'TOTAL CLSTR', 'CLSTR REQ']
                summary_display = summary_df[summary_display_cols].rename(columns={
                    'ETA': 'ETA', 
                    'CLOSING PHYSIC': 'CLOSING TIME',
                    'TOTAL BOX': 'BOX STACKED', 
                    'Loading Forecast': 'LOADING FORECAST'
                })
                st.session_state.summary_display = summary_display

                def style_diff(v): return f'color: {"#4CAF50" if v > 0 else ("#F44336" if v < 0 else "#757575")}; font-weight: bold;'
                def highlight_rows(row):
                    if row['TOTAL CLSTR'] < row['CLSTR REQ']: return ['background-color: #FFCDD2'] * len(row)
                    if row['VESSEL'] in priority_vessels: return ['background-color: #FFF3CD'] * len(row)
                    return [''] * len(row)
                
                styled_df = summary_display.style.apply(highlight_rows, axis=1).map(style_diff, subset=['DIFF']).format({
                    'ETA': '{:%d/%m/%Y %H:%M}',
                    'CLOSING TIME': '{:%d/%m/%Y %H:%M}'
                })
                st.dataframe(styled_df, use_container_width=True, hide_index=True)
            else:
                st.info("Tidak ada kapal yang dijadwalkan datang dalam 4 hari ke depan.")
        else:
            st.warning("Data peramalan tidak tersedia. Jalankan peramalan di tab 'Peramalan Muatan' terlebih dahulu.")

        # --- VISUALISASI SEBARAN CLUSTER ---
        st.markdown("---")
        st.subheader("📊 Visualisasi Sebaran Cluster")
        all_vessels_list = display_df['VESSEL'].unique().tolist()
        st.sidebar.markdown("---")
        st.sidebar.header("📊 Opsi Grafik")
        selected_vessels = st.sidebar.multiselect("Filter Kapal pada Grafik:", options=all_vessels_list, default=all_vessels_list)
        font_size = st.sidebar.slider("Sesuaikan Ukuran Font Grafik", min_value=6, max_value=20, value=10, step=1)
        if not selected_vessels:
            st.warning("Pilih setidaknya satu kapal untuk ditampilkan.")
        else:
            processed_df_chart = display_df[display_df['VESSEL'].isin(selected_vessels)]
            initial_cols_chart = ['VESSEL', 'VOY_OUT', 'ETA', 'ETD', 'SERVICE', 'TOTAL BOX', 'TOTAL CLSTR', 'ETA_Display', 'ETA_Date', 'CLOSING PHYSIC']
            cluster_cols_chart = sorted([col for col in processed_df_chart.columns if col not in initial_cols_chart])
            
            chart_data_long = pd.melt(processed_df_chart, id_vars=['VESSEL', 'ETA'], value_vars=cluster_cols_chart, var_name='Cluster', value_name='Box Count')
            chart_data_long = chart_data_long[chart_data_long['Box Count'] > 0]
            if not chart_data_long.empty:
                chart_data_long['combined_text'] = chart_data_long['Cluster'] + ' / ' + chart_data_long['Box Count'].astype(str)
                cluster_color_map = {'A01': '#5409DA', 'A02': '#4E71FF', 'A03': '#8DD8FF', 'A04': '#BBFBFF', 'A05': '#8DBCC7', 'B01': '#328E6E', 'B02': '#67AE6E', 'B03': '#90C67C', 'B04': '#E1EEBC', 'B05': '#D2FF72', 'C03': '#B33791', 'C04': '#C562AF', 'C05': '#DB8DD0'}
                vessel_order_by_eta = processed_df_chart.sort_values('ETA')['VESSEL'].tolist()
                fig = px.bar(chart_data_long, x='Box Count', y='VESSEL', color='Cluster', color_discrete_map=cluster_color_map, orientation='h', title='Distribusi Box per Cluster untuk Setiap Kapal', text='combined_text', hover_data={'VESSEL': False, 'Cluster': True, 'Box Count': True})
                fig.update_layout(xaxis_title=None, yaxis_title=None, height=len(vessel_order_by_eta) * 35 + 150, legend_title_text='Area Cluster', title_x=0)
                fig.update_yaxes(categoryorder='array', categoryarray=vessel_order_by_eta[::-1])
                fig.update_traces(textposition='inside', textfont_size=font_size, textangle=0)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Tidak ada data cluster untuk divisualisasikan untuk kapal yang dipilih.")
        
        # --- RINGKASAN POTENSI BENTROK (LOGIKA DARI SKRIP LAMA) ---
        st.markdown("---")
        st.header("💥 Ringkasan Potensi Bentrok")
        vessel_area_slots_df = st.session_state.get('vessel_area_slots')
        clash_details = {}
        clash_summary_data = []
        if vessel_area_slots_df is not None:
            active_vessels = vessel_area_slots_df[['VESSEL', 'VOY_OUT', 'ETA', 'ETD']].drop_duplicates()
            summary_exclude_blocks = ['BR9', 'RC9', 'C01', 'C02', 'D01', 'OOG']
            for (idx1, vessel1), (idx2, vessel2) in combinations(active_vessels.iterrows(), 2):
                if (vessel1['ETA'] < vessel2['ETD']) and (vessel2['ETA'] < vessel1['ETD']):
                    v1_slots = vessel_area_slots_df[(vessel_area_slots_df['VESSEL'] == vessel1['VESSEL']) & (vessel_area_slots_df['VOY_OUT'] == vessel1['VOY_OUT'])]
                    v2_slots = vessel_area_slots_df[(vessel_area_slots_df['VESSEL'] == vessel2['VESSEL']) & (vessel_area_slots_df['VOY_OUT'] == vessel2['VOY_OUT'])]
                    common_areas = pd.merge(v1_slots, v2_slots, on='Area (EXE)', suffixes=('_v1', '_v2'))
                    for _, row in common_areas.iterrows():
                        area = row['Area (EXE)']
                        if area in summary_exclude_blocks: continue
                        range1, range2 = (row['MIN_SLOT_v1'], row['MAX_SLOT_v1']), (row['MIN_SLOT_v2'], row['MAX_SLOT_v2'])
                        gap = max(range1[0], range2[0]) - min(range1[1], range2[1]) - 1
                        if gap <= min_clash_distance:
                            clash_date = max(vessel1['ETA'], vessel2['ETA']).normalize()
                            date_key = clash_date.strftime('%d/%m/%Y')
                            if date_key not in clash_details: clash_details[date_key] = []
                            clash_info = {"block": area, "vessel1_name": vessel1['VESSEL'], "vessel1_slots": f"{range1[0]}-{range1[1]}", "vessel1_box": row['BOX_COUNT_v1'], "vessel2_name": vessel2['VESSEL'], "vessel2_slots": f"{range2[0]}-{range2[1]}", "vessel2_box": row['BOX_COUNT_v2'], "gap": gap}
                            if not any(d['block'] == area and d['vessel1_name'] in [vessel1['VESSEL'], vessel2['VESSEL']] and d['vessel2_name'] in [vessel1['VESSEL'], vessel2['VESSEL']] for d in clash_details.get(date_key, [])):
                                clash_details[date_key].append(clash_info)
                                clash_summary_data.append({"Clash Date": date_key, "Block": area, "Gap (slots)": gap, "Vessel 1": vessel1['VESSEL'], "Vessel 2": vessel2['VESSEL'], "Notes": ""})

        st.session_state.clash_summary_df = pd.DataFrame(clash_summary_data)

        if not clash_details:
            st.info(f"✅ Tidak ada potensi bentrok yang ditemukan dengan jarak aman minimal {min_clash_distance} slot.")
        else:
            total_clash_days = len(clash_details)
            st.markdown(f"**🔥 Ditemukan {total_clash_days} hari dengan potensi bentrok.**")
            clash_dates = sorted(clash_details.keys(), key=lambda x: datetime.strptime(x, '%d/%m/%Y'))
            cols = st.columns(len(clash_dates) or 1)
            for i, date_key in enumerate(clash_dates):
                with cols[i]:
                    with st.container(border=True):
                        st.markdown(f"**Potensi Bentrok pada: {date_key}**")
                        clashes_for_date = clash_details.get(date_key, [])
                        for clash in clashes_for_date:
                            st.divider()
                            st.markdown(f"**Blok {clash['block']}** (Jarak: `{clash['gap']}` slot)")
                            st.markdown(f"*{clash['vessel1_name']}*: `{clash['vessel1_box']}` box (Slot: `{clash['vessel1_slots']}`)")
                            st.markdown(f"*{clash['vessel2_name']}*: `{clash['vessel2_box']}` box (Slot: `{clash['vessel2_slots']}`)")
        
        # --- HASIL ANALISIS DETAIL (TABEL INTERAKTIF) ---
        st.markdown("---")
        st.header("📋 Hasil Analisis Detail")
        
        df_for_grid = display_df.copy()
        
        # Membuat peta bentrok untuk highlight di tabel
        clash_map_for_grid = {date: [item['block'] for item in clashes] for date, clashes in clash_details.items()}
        
        # Konfigurasi AgGrid
        hide_zero_jscode = JsCode("""function(params) { if (params.value == 0 || params.value === null) { return ''; } return params.value; }""")
        clash_cell_style_jscode = JsCode(f"""function(params) {{ const clashMap = {json.dumps(clash_map_for_grid)}; const date = params.data.ETA_Date; const colId = params.colDef.field; const isClash = clashMap[date] ? clashMap[date].includes(colId) : false; if (isClash) {{ return {{'backgroundColor': '#FFAA33', 'color': 'black'}}; }} return null; }}""")
        
        unique_dates = df_for_grid['ETA_Date'].unique()
        date_color_map = {date: ['#F8F0E5', '#DAC0A3'][i % 2] for i, date in enumerate(unique_dates)}
        zebra_row_style_jscode = JsCode(f"""function(params) {{ const dateColorMap = {json.dumps(date_color_map)}; const date = params.data.ETA_Date; const color = dateColorMap[date]; return {{ 'background-color': color }}; }}""")
        
        default_col_def = {"suppressMenu": True, "sortable": True, "resizable": True, "editable": False, "minWidth": 40}
        
        column_defs = []
        pinned_cols = ['VESSEL', 'SERVICE', 'VOY_OUT', 'ETA_Display', 'TOTAL BOX', 'TOTAL CLSTR']
        for col in pinned_cols:
            header = "ETA" if col == 'ETA_Display' else col
            width = 110 if col == 'VESSEL' else (120 if col == 'ETA_Display' else (90 if col == 'SERVICE' else 80))
            col_def = {"field": col, "headerName": header, "pinned": "left", "width": width}
            if col in ["TOTAL BOX", "TOTAL CLSTR"]: col_def["cellRenderer"] = hide_zero_jscode
            column_defs.append(col_def)
        
        cluster_cols_aggrid = sorted([col for col in df_for_grid.columns if col not in pinned_cols + ['ETA', 'ETD', 'ETA_Display', 'ETA_Date', 'CLOSING PHYSIC']])
        for col in cluster_cols_aggrid:
            column_defs.append({"field": col, "headerName": col, "width": 60, "cellRenderer": hide_zero_jscode, "cellStyle": clash_cell_style_jscode})
        
        column_defs.append({"field": "ETA_Date", "hide": True})
        gridOptions = {"defaultColDef": default_col_def, "columnDefs": column_defs, "getRowStyle": zebra_row_style_jscode}
        AgGrid(df_for_grid, gridOptions=gridOptions, height=600, width='100%', theme='streamlit', allow_unsafe_jscode=True, key='detailed_analysis_grid')

        # --- PUSAT UNDUHAN ---
        st.markdown("---")
        st.subheader("📥 Pusat Unduhan")
        try:
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                workbook = writer.book
                center_format = workbook.add_format({'align': 'center', 'valign': 'vcenter'})

                def auto_adjust_and_format_sheet(df, sheet_name, writer_obj):
                    if df is not None and not df.empty:
                        df_to_write = df.copy()
                        # Format kolom tanggal jika ada
                        for col_date in ['ETA', 'CLOSING TIME', 'Clash Date', 'CLOSING PHYSIC']:
                            if col_date in df_to_write.columns:
                                # Cek jika tipenya datetime sebelum format
                                if pd.api.types.is_datetime64_any_dtype(df_to_write[col_date]):
                                     df_to_write[col_date] = pd.to_datetime(df_to_write[col_date]).dt.strftime('%d/%m/%Y %H:%M')
                        
                        df_to_write.to_excel(writer_obj, sheet_name=sheet_name, index=False)
                        worksheet = writer_obj.sheets[sheet_name]
                        for idx, col_name in enumerate(df_to_write.columns):
                            series = df_to_write[col_name].dropna()
                            max_len = max(([len(str(s)) for s in series] if not series.empty else [0]) + [len(str(col_name))]) + 5
                            worksheet.set_column(idx, idx, min(max_len, 50), center_format)
                
                # Menulis setiap DataFrame ke sheet yang berbeda
                auto_adjust_and_format_sheet(st.session_state.get('processed_df'), 'Analisis Detail', writer)
                auto_adjust_and_format_sheet(st.session_state.get('summary_display'), 'Ringkasan Kapal Datang', writer)
                auto_adjust_and_format_sheet(st.session_state.get('clash_summary_df'), 'Ringkasan Bentrok', writer)

            if output.tell() > 0:
                st.download_button(
                    label="📥 Unduh Laporan Analisis (Excel)",
                    data=output.getvalue(),
                    file_name=f"clash_analysis_report_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
        except Exception as e:
            st.error(f"Gagal membuat file unduhan: {e}")

    else:
        st.info("Selamat datang! Mohon unggah file Anda dan klik 'Proses Data' untuk memulai.")


# --- STRUKTUR UTAMA APLIKASI DENGAN TABS ---
st.sidebar.header("⚙️ Kontrol & Unggah")
schedule_file = st.sidebar.file_uploader("1. Unggah Jadwal Kapal", type=['xlsx', 'csv'], key="schedule_uploader")
unit_list_file = st.sidebar.file_uploader("2. Unggah Daftar Unit", type=['xlsx', 'csv'], key="unit_list_uploader")
min_clash_distance = st.sidebar.number_input("Jarak Aman Minimal (slot)", min_value=0, value=5, step=1, key="min_clash_dist_input", help="Bentrok terdeteksi jika jarak antar alokasi kapal kurang dari atau sama dengan nilai ini.")
process_button = st.sidebar.button("🚀 Proses Data", use_container_width=True, type="primary")
st.sidebar.button("🔄 Reset Data", on_click=reset_data, use_container_width=True, help="Hapus semua data yang telah diproses untuk memulai dari awal.")

tab1, tab2 = st.tabs(["🚨 Analisis Bentrok", "📈 Peramalan Muatan"])

with tab1:
    render_clash_tab(process_button, schedule_file, unit_list_file, min_clash_distance)
with tab2:
    render_forecast_tab()
