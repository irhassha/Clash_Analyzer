import streamlit as st
import pandas as pd
import os
from datetime import datetime, timedelta
import json
import io
import warnings
import numpy as np
import matplotlib.pyplot as plt

# --- Pustaka untuk Machine Learning ---
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

# Import pustaka yang diperlukan
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode

# Mengabaikan peringatan yang tidak krusial
warnings.filterwarnings("ignore", category=UserWarning)

# --- Konfigurasi Halaman & Judul ---
st.set_page_config(page_title="Ops Analyzer", layout="wide")
st.title("Yard Ops Analyzer")

# --- FUNGSI-FUNGSI UNTUK FORECASTING (MODEL BARU: PER-SERVICE RF) ---

@st.cache_data
def load_history_data(filename="History Loading.xlsx"):
    """Mencari dan memuat file data historis untuk forecasting."""
    if os.path.exists(filename):
        try:
            if filename.lower().endswith('.csv'):
                df = pd.read_csv(filename)
            else:
                df = pd.read_excel(filename)
            
            df.columns = [col.strip().lower() for col in df.columns]
            df['ata'] = pd.to_datetime(df['ata'], dayfirst=True, errors='coerce')
            df.dropna(subset=['ata', 'loading', 'service'], inplace=True)
            df = df[df['loading'] >= 0]
            return df
        except Exception as e:
            st.error(f"Gagal memuat file histori '{filename}': {e}")
            return None
    st.warning(f"File histori '{filename}' tidak ditemukan di repositori.")
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
def run_per_service_rf_forecast(df_history):
    """Menjalankan proses pembersihan outlier dan forecasting untuk setiap service."""
    all_results = []
    unique_services = df_history['service'].unique()

    progress_bar = st.progress(0, text="Menganalisis service...")
    total_services = len(unique_services)

    for i, service in enumerate(unique_services):
        progress_text = f"Menganalisis service: {service} ({i+1}/{total_services})"
        progress_bar.progress((i + 1) / total_services, text=progress_text)

        service_df = df_history[df_history['service'] == service].copy()
        
        if service_df.empty or service_df['loading'].isnull().all():
            continue

        # --- 1. Pembersihan Outlier ---
        Q1 = service_df['loading'].quantile(0.25)
        Q3 = service_df['loading'].quantile(0.75)
        IQR = Q3 - Q1
        upper_bound = Q3 + 1.5 * IQR
        lower_bound = Q1 - 1.5 * IQR
        num_outliers = ((service_df['loading'] < lower_bound) | (service_df['loading'] > upper_bound)).sum()
        service_df['loading_cleaned'] = service_df['loading'].clip(lower=lower_bound, upper=upper_bound)

        # --- 2. Pilih Model & Forecast ---
        forecast_val, moe_val, mape_val, method = (0, 0, 0, "")
        
        if len(service_df) >= 10:
            try:
                df_features = create_time_features(service_df)
                features_to_use = ['hour', 'day_of_week', 'day_of_month', 'day_of_year', 'week_of_year', 'month', 'year']
                target = 'loading_cleaned'

                X = df_features[features_to_use]
                y = df_features[target]

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
                
                if len(X_train) == 0:
                    raise ValueError("Tidak cukup data untuk melatih model.")

                model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1, min_samples_leaf=2)
                model.fit(X_train, y_train)
                
                predictions = model.predict(X_test)
                mape_val = mean_absolute_percentage_error(y_test, predictions) * 100 if len(y_test) > 0 else 0
                moe_val = 1.96 * np.std(y_test - predictions) if len(y_test) > 0 else 0
                
                future_eta = datetime.now().replace(hour=12, minute=0, second=0, microsecond=0) + timedelta(days=1)
                future_df = create_time_features(pd.DataFrame([{'ata': future_eta}]))
                forecast_val = model.predict(future_df[features_to_use])[0]
                
                method = f"Random Forest ({num_outliers} outlier dibersihkan)"
            except Exception:
                forecast_val = service_df['loading_cleaned'].mean()
                moe_val = 1.96 * service_df['loading_cleaned'].std()
                actuals = service_df['loading_cleaned']
                mape_val = np.mean(np.abs((actuals - forecast_val) / actuals)) * 100 if not actuals.empty else 0
                method = f"Rata-rata Historis (RF Gagal, {num_outliers} outlier dibersihkan)"
        else:
            forecast_val = service_df['loading_cleaned'].mean()
            moe_val = 1.96 * service_df['loading_cleaned'].std()
            actuals = service_df['loading_cleaned']
            mape_val = np.mean(np.abs((actuals - forecast_val) / actuals)) * 100 if not actuals.empty else 0
            method = f"Rata-rata Historis ({num_outliers} outlier dibersihkan)"

        all_results.append({
            "Service": service,
            "Prediksi Loading Berikutnya": max(0, forecast_val),
            "Margin of Error (± box)": moe_val,
            "MAPE (%)": mape_val,
            "Metode": method
        })
        
    progress_bar.empty()
    return pd.DataFrame(all_results)

def render_forecast_tab():
    """Fungsi untuk menampilkan seluruh konten tab forecasting."""
    st.header("📈 Forecast Loading dengan Machine Learning")
    st.write("""
    Fitur ini menggunakan model **Random Forest** terpisah untuk setiap service. 
    Model belajar dari pola waktu historis untuk memberikan prediksi yang lebih akurat, lengkap dengan pembersihan data anomali.
    """)
    
    st.info("Pastikan file `History Loading.xlsx` ada di repositori GitHub Anda.", icon="ℹ️")

    if 'forecast_df' not in st.session_state:
        df_history = load_history_data()
        if df_history is not None:
            with st.spinner("Memproses data dan melatih model untuk setiap service..."):
                forecast_df = run_per_service_rf_forecast(df_history)
                st.session_state.forecast_df = forecast_df
        else:
            st.session_state.forecast_df = pd.DataFrame()
            st.error("Tidak dapat memuat data historis. Proses dibatalkan.")
    
    if 'forecast_df' in st.session_state:
        results_df = st.session_state.forecast_df
        
        if not results_df.empty:
            results_df['Prediksi Loading Berikutnya'] = results_df['Prediksi Loading Berikutnya'].round(2)
            results_df['Margin of Error (± box)'] = results_df['Margin of Error (± box)'].fillna(0).round(2)
            results_df['MAPE (%)'] = results_df['MAPE (%)'].replace([np.inf, -np.inf], 0).fillna(0).round(2)

            st.markdown("---")
            st.subheader("📊 Hasil Forecast per Service")
            st.dataframe(
                results_df.sort_values(by="Prediksi Loading Berikutnya", ascending=False).reset_index(drop=True),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "MAPE (%)": st.column_config.NumberColumn(format="%.2f%%")
                }
            )
            
            st.markdown("---")
            st.subheader("💡 Cara Membaca Hasil Ini")
            st.markdown("""
            - **Prediksi Loading Berikutnya**: Estimasi jumlah box untuk kedatangan kapal selanjutnya dari service tersebut.
            - **Margin of Error (± box)**: Tingkat ketidakpastian prediksi. Prediksi **300** dengan MoE **±50** berarti nilai aktual kemungkinan besar berada di antara **250** dan **350**.
            - **MAPE (%)**: Rata-rata persentase kesalahan model saat diuji pada data historisnya. **Semakin kecil nilainya, semakin akurat modelnya di masa lalu.**
            - **Metode**: Teknik yang digunakan untuk forecast dan jumlah outlier yang ditangani.
            """)
        else:
            st.warning("Tidak ada data forecast yang dapat dibuat. File histori mungkin kosong atau tidak berisi data service yang valid.")

@st.cache_data
def load_vessel_codes_from_repo(possible_names=['vessel codes.xlsx', 'vessel_codes.xls', 'vessel_codes.csv']):
    """Mencari dan memuat file kode kapal."""
    for filename in possible_names:
        if os.path.exists(filename):
            try:
                if filename.lower().endswith('.csv'): df = pd.read_csv(filename)
                else: df = pd.read_excel(filename)
                df.columns = df.columns.str.strip()
                return df
            except Exception as e:
                st.error(f"Gagal membaca file '{filename}': {e}"); return None
    st.error(f"File kode kapal tidak ditemukan."); return None

def render_clash_tab():
    """Fungsi untuk menampilkan seluruh konten tab Analisis Clash."""
    st.sidebar.header("⚙️ Unggah File Anda")
    schedule_file = st.sidebar.file_uploader("1. Unggah Jadwal Kapal", type=['xlsx', 'csv'])
    unit_list_file = st.sidebar.file_uploader("2. Unggah Daftar Unit", type=['xlsx', 'csv'])
    process_button = st.sidebar.button("🚀 Proses Data Clash", type="primary")

    if 'processed_df' not in st.session_state:
        st.session_state.processed_df = None
    if 'clash_summary_df' not in st.session_state:
        st.session_state.clash_summary_df = None

    df_vessel_codes = load_vessel_codes_from_repo()

    if process_button:
        if schedule_file and unit_list_file and (df_vessel_codes is not None and not df_vessel_codes.empty):
            with st.spinner('Memuat dan memproses data...'):
                try:
                    # 1. Loading & Cleaning
                    if schedule_file.name.lower().endswith(('.xls', '.xlsx')): df_schedule = pd.read_excel(schedule_file)
                    else: df_schedule = pd.read_csv(schedule_file)
                    df_schedule.columns = [col.strip().upper() for col in df_schedule.columns] # Konsisten ke huruf besar
                    
                    if unit_list_file.name.lower().endswith(('.xls', '.xlsx')): df_unit_list = pd.read_excel(unit_list_file)
                    else: df_unit_list = pd.read_csv(unit_list_file)
                    df_unit_list.columns = [col.strip() for col in df_unit_list.columns]
                    
                    # 2. Main Processing
                    original_vessels_list = df_schedule['VESSEL'].unique().tolist()
                    df_schedule['ETA'] = pd.to_datetime(df_schedule['ETA'], errors='coerce')
                    df_schedule_with_code = pd.merge(df_schedule, df_vessel_codes, left_on="VESSEL", right_on="Description", how="left").rename(columns={"Value": "CODE"})
                    merged_df = pd.merge(df_schedule_with_code, df_unit_list, left_on=['CODE', 'VOY_OUT'], right_on=['Carrier Out', 'Voyage Out'], how='inner')
                    if merged_df.empty: st.warning("Tidak ada data yang cocok ditemukan."); st.session_state.processed_df = None; st.stop()
                    
                    # 3. Filtering
                    merged_df = merged_df[merged_df['VESSEL'].isin(original_vessels_list)]
                    excluded_areas = [str(i) for i in range(801, 809)]
                    merged_df['Area (EXE)'] = merged_df['Area (EXE)'].astype(str)
                    filtered_data = merged_df[~merged_df['Area (EXE)'].isin(excluded_areas)]
                    if filtered_data.empty: st.warning("Tidak ada data tersisa setelah filtering."); st.session_state.processed_df = None; st.stop()

                    # 4. Pivoting
                    grouping_cols = ['VESSEL', 'CODE', 'SERVICE', 'VOY_OUT', 'ETA']
                    pivot_df = filtered_data.pivot_table(index=grouping_cols, columns='Area (EXE)', aggfunc='size', fill_value=0)
                    
                    cluster_cols_for_calc = pivot_df.columns.tolist()
                    pivot_df['TTL BOX'] = pivot_df[cluster_cols_for_calc].sum(axis=1)
                    
                    # --- PERUBAHAN: Kalkulasi TTL CLSTR dengan pengecualian ---
                    exclude_for_clstr = ['D01', 'C01', 'C02', 'OOG', 'UNKNOWN', 'BR9', 'RC9']
                    clstr_calculation_cols = [col for col in cluster_cols_for_calc if col not in exclude_for_clstr]
                    pivot_df['TTL CLSTR'] = (pivot_df[clstr_calculation_cols] > 0).sum(axis=1)
                    
                    pivot_df = pivot_df.reset_index()
                    
                    # 5. Conditional Filtering
                    two_days_ago = pd.Timestamp.now() - timedelta(days=2)
                    condition_to_hide = (pivot_df['ETA'] < two_days_ago) & (pivot_df['TTL BOX'] < 50)
                    pivot_df = pivot_df[~condition_to_hide]
                    if pivot_df.empty: st.warning("Tidak ada data tersisa setelah filter ETA & Total."); st.session_state.processed_df = None; st.stop()

                    # 6. Sorting and Ordering
                    cols_awal = ['VESSEL', 'CODE', 'SERVICE', 'VOY_OUT', 'ETA', 'TTL BOX', 'TTL CLSTR']
                    final_cluster_cols = [col for col in pivot_df.columns if col not in cols_awal]
                    final_display_cols = cols_awal + sorted(final_cluster_cols)
                    pivot_df = pivot_df[final_display_cols]
                    
                    pivot_df['ETA_str'] = pd.to_datetime(pivot_df['ETA']).dt.strftime('%Y-%m-%d %H:%M')
                    
                    pivot_df = pivot_df.sort_values(by='ETA', ascending=True).reset_index(drop=True)
                    
                    st.session_state.processed_df = pivot_df
                    st.success("Data berhasil diproses!")

                except Exception as e:
                    st.error(f"Terjadi kesalahan saat pemrosesan: {e}")
                    st.session_state.processed_df = None
        else:
            st.warning("Mohon unggah kedua file.")

    # --- Area Tampilan ---
    if st.session_state.get('processed_df') is not None:
        display_df = st.session_state.processed_df
        
        # --- RINGKASAN BARU: KAPAL DATANG & FORECAST ---
        st.markdown("---")
        st.subheader("🚢 Ringkasan Kapal Datang (Hari Ini + 3 Hari ke Depan)")
        
        forecast_df = st.session_state.get('forecast_df')
        if forecast_df is not None and not forecast_df.empty:
            today = pd.to_datetime(datetime.now().date())
            four_days_later = today + timedelta(days=4)
            
            upcoming_vessels_df = display_df[
                (display_df['ETA'] >= today) & 
                (display_df['ETA'] < four_days_later)
            ].copy()

            if not upcoming_vessels_df.empty:
                forecast_lookup = forecast_df[['Service', 'Prediksi Loading Berikutnya']].copy()
                
                summary_df = pd.merge(
                    upcoming_vessels_df,
                    forecast_lookup,
                    left_on='SERVICE',
                    right_on='Service',
                    how='left'
                )
                summary_df['Prediksi Loading Berikutnya'] = summary_df['Prediksi Loading Berikutnya'].fillna(0).round(2)
                
                # --- PERUBAHAN: Tambahkan kolom selisih ---
                summary_df['Selisih'] = summary_df['TTL BOX'] - summary_df['Prediksi Loading Berikutnya']
                
                # --- PERUBAHAN: Urutan kolom baru ---
                summary_display_cols = [
                    'VESSEL', 'SERVICE', 'ETA_str', 'TTL BOX', 
                    'Prediksi Loading Berikutnya', 'Selisih', 'TTL CLSTR'
                ]
                summary_display = summary_df[summary_display_cols]
                summary_display = summary_display.rename(columns={'ETA_str': 'ETA'})
                
                st.dataframe(summary_display, use_container_width=True, hide_index=True)
            else:
                st.info("Tidak ada kapal yang dijadwalkan datang dalam 4 hari ke depan.")
        else:
            st.warning("Data forecast tidak tersedia. Silakan jalankan forecast di tab 'Forecast Loading' terlebih dahulu.")
        
        st.header("📋 Hasil Analisis Detail")

        # --- Persiapan untuk Styling AG Grid dan Summary ---
        df_for_grid = display_df.copy()
        df_for_grid['ETA_Date'] = pd.to_datetime(df_for_grid['ETA']).dt.strftime('%Y-%m-%d')
        df_for_grid['ETA'] = df_for_grid['ETA_str']
        
        unique_dates = df_for_grid['ETA_Date'].unique()
        zebra_colors = ['#F8F0E5', '#DAC0A3'] 
        date_color_map = {date: zebra_colors[i % 2] for i, date in enumerate(unique_dates)}

        clash_map = {}
        cluster_cols = [col for col in df_for_grid.columns if col not in ['VESSEL', 'CODE', 'SERVICE', 'VOY_OUT', 'ETA', 'TTL BOX', 'TTL CLSTR', 'ETA_Date', 'ETA_str']]
        for date, group in df_for_grid.groupby('ETA_Date'):
            clash_areas_for_date = []
            for col in cluster_cols:
                if (group[col] > 0).sum() > 1:
                    clash_areas_for_date.append(col)
            if clash_areas_for_date:
                clash_map[date] = clash_areas_for_date

        # --- TAMPILAN RINGKASAN CLASH DENGAN KARTU ---
        summary_data = []
        if clash_map:
            summary_exclude_blocks = ['BR9', 'RC9', 'C01', 'D01', 'OOG']

            with st.expander("Tampilkan Ringkasan Clash", expanded=True):
                total_clash_days = len(clash_map)
                total_conflicting_blocks = sum(len(areas) for areas in clash_map.values())
                st.markdown(f"**🔥 Ditemukan {total_clash_days} hari clash dengan total {total_conflicting_blocks} blok yang berkonflik.**")
                st.markdown("---")
                
                clash_dates = sorted(clash_map.keys())
                cols = st.columns(len(clash_dates) or 1)

                for i, date in enumerate(clash_dates):
                    with cols[i]:
                        areas = clash_map[date]
                        filtered_areas = [area for area in areas if area not in summary_exclude_blocks]
                        if not filtered_areas:
                            continue
                        
                        summary_html = f"""
                        <div style="background-color: #F8F9FA; border: 1px solid #E9ECEF; border-radius: 10px; padding: 15px; height: 100%;">
                            <strong style='font-size: 1.2em;'>Clash pada: {date}</strong>
                            <hr style='margin: 10px 0;'>
                            <div style='line-height: 1.7;'>
                        """
                        for area in sorted(filtered_areas):
                            clashing_rows = df_for_grid[(df_for_grid['ETA_Date'] == date) & (df_for_grid[area] > 0)]
                            clashing_vessels = clashing_rows['VESSEL'].tolist()
                            total_clash_boxes = clashing_rows[area].sum()
                            vessel_list_str = ", ".join(clashing_vessels)
                            
                            summary_html += f"<b>Blok {area}</b> (<span style='color:#E67E22; font-weight:bold;'>{total_clash_boxes} boxes</span>):<br><small>{vessel_list_str}</small><br>"
                            
                            summary_data.append({
                                "Tanggal Clash": date,
                                "Blok": area,
                                "Total Box": total_clash_boxes,
                                "Kapal": vessel_list_str,
                                "Keterangan": ""
                            })
                        summary_html += "</div></div>"
                        st.markdown(summary_html, unsafe_allow_html=True)
                
                st.session_state.clash_summary_df = pd.DataFrame(summary_data)

        st.markdown("---")

        # --- PENGGUNAAN AG-GRID ---
        hide_zero_jscode = JsCode("""function(params) { if (params.value == 0 || params.value === null) { return ''; } return params.value; }""")
        clash_cell_style_jscode = JsCode(f"""
            function(params) {{
                const clashMap = {json.dumps(clash_map)};
                const date = params.data.ETA_Date;
                const colId = params.colDef.field;
                const isClash = clashMap[date] ? clashMap[date].includes(colId) : false;
                if (isClash) {{ return {{'backgroundColor': '#FFAA33', 'color': 'black'}}; }}
                return null;
            }}
        """)
        zebra_row_style_jscode = JsCode(f"""
            function(params) {{
                const dateColorMap = {json.dumps(date_color_map)};
                const date = params.data.ETA_Date;
                const color = dateColorMap[date];
                return {{ 'background-color': color }};
            }}
        """)

        default_col_def = {"suppressMenu": True, "sortable": True, "resizable": True, "editable": False, "minWidth": 40}
        column_defs = []
        pinned_cols = ['VESSEL', 'CODE', 'SERVICE', 'VOY_OUT', 'ETA', 'TTL BOX', 'TTL CLSTR']
        for col in pinned_cols:
            width = 110 if col == 'VESSEL' else 80
            if col == 'ETA': width = 120 
            if col == 'SERVICE': width = 90
            col_def = {"field": col, "headerName": col, "pinned": "left", "width": width}
            if col in ["TTL BOX", "TTL CLSTR"]: col_def["cellRenderer"] = hide_zero_jscode
            column_defs.append(col_def)
        for col in cluster_cols:
            column_defs.append({"field": col, "headerName": col, "width": 60, "cellRenderer": hide_zero_jscode, "cellStyle": clash_cell_style_jscode})
        column_defs.append({"field": "ETA_Date", "hide": True})
        gridOptions = {"defaultColDef": default_col_def, "columnDefs": column_defs, "getRowStyle": zebra_row_style_jscode}

        AgGrid(df_for_grid.drop(columns=['ETA_str']), gridOptions=gridOptions, height=600, width='100%', theme='streamlit', allow_unsafe_jscode=True)
        
        # --- LOGIKA TOMBOL DOWNLOAD EXCEL ---
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            if 'clash_summary_df' in st.session_state and st.session_state.clash_summary_df is not None:
                summary_df_for_export = st.session_state.clash_summary_df
                
                final_summary_export = []
                last_date = None
                if not summary_df_for_export.empty:
                    for index, row in summary_df_for_export.iterrows():
                        current_date = row['Tanggal Clash']
                        if last_date is not None and current_date != last_date:
                            final_summary_export.append({}) 
                        final_summary_export.append(row.to_dict())
                        last_date = current_date
                
                final_summary_df = pd.DataFrame(final_summary_export)

                final_summary_df.to_excel(writer, sheet_name='Clash Summary', index=False)
                
                workbook = writer.book
                summary_sheet = writer.sheets['Clash Summary']
                center_format = workbook.add_format({'align': 'center', 'valign': 'vcenter'})

                for idx, col in enumerate(final_summary_df.columns):
                    series = final_summary_df[col].dropna()
                    if not series.empty:
                        max_len = max([len(str(s)) for s in series] + [len(col)]) + 4
                        summary_sheet.set_column(idx, idx, max_len, center_format)
                    else:
                        summary_sheet.set_column(idx, idx, len(col) + 4, center_format)

        st.download_button(
            label="📥 Unduh Ringkasan Clash (Excel)",
            data=output.getvalue(),
            file_name="clash_summary_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

# --- STRUKTUR UTAMA DENGAN TABS ---
tab1, tab2 = st.tabs(["🚨 Analisis Clash", "📈 Forecast Loading"])

with tab1:
    render_clash_tab()

with tab2:
    render_forecast_tab()
