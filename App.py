import streamlit as st
import pandas as pd
import os
from datetime import datetime, timedelta
import json
import io
import warnings

# --- Pustaka baru untuk Forecasting ---
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt

# Import pustaka yang diperlukan
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode

# Mengabaikan peringatan yang tidak krusial
warnings.filterwarnings("ignore", category=UserWarning)

# --- Konfigurasi Halaman & Judul ---
st.set_page_config(page_title="Ops Analyzer", layout="wide")
st.title("Yard Operations Analyzer")

# --- FUNGSI-FUNGSI UNTUK FORECASTING (BARU) ---

@st.cache_data
def load_history_data(filename="History Loading.xlsx"):
    """Mencari dan memuat file data historis untuk forecasting."""
    if os.path.exists(filename):
        try:
            df = pd.read_csv(filename)
            df.columns = [col.strip().lower() for col in df.columns]
            # Konversi kolom tanggal, asumsi format DD/MM/YYYY
            df['ata'] = pd.to_datetime(df['ata'], dayfirst=True, errors='coerce')
            # Hapus baris jika tanggal tidak valid
            df.dropna(subset=['ata'], inplace=True)
            return df
        except Exception as e:
            st.error(f"Gagal memuat file histori '{filename}': {e}")
            return None
    st.warning(f"File histori '{filename}' tidak ditemukan di repository.")
    return None

def run_forecasting_process(df_history):
    """Menjalankan seluruh proses forecasting per service."""
    all_results = []
    unique_services = df_history['service'].unique()

    progress_bar = st.progress(0, text="Menganalisis services...")
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
        
        # Capping: Ganti outlier dengan nilai batas
        service_df['loading_cleaned'] = service_df['loading'].clip(lower=lower_bound, upper=upper_bound)

        # --- 2. Pilih Model & Forecast ---
        forecast_val, moe_val, method = (None, None, "")
        
        # Gunakan SARIMA jika data cukup, jika tidak gunakan rata-rata
        if len(service_df) >= 30:
            try:
                # Siapkan data harian untuk model time series
                daily_data = service_df.set_index('ata')['loading_cleaned'].resample('D').sum().fillna(0)
                
                model = SARIMAX(daily_data, order=(1, 1, 1), seasonal_order=(1, 1, 0, 7), enforce_stationarity=False, enforce_invertibility=False)
                results = model.fit(disp=False)
                
                forecast_obj = results.get_forecast(steps=1)
                forecast_val = forecast_obj.predicted_mean.iloc[0]
                # Hitung Margin of Error (MoE)
                se = forecast_obj.summary_frame()['mean_se'].iloc[0]
                moe_val = se * 1.96 # 95% confidence interval
                method = f"Model SARIMA ({num_outliers} outlier dibersihkan)"
            except Exception:
                # Fallback ke rata-rata jika SARIMA gagal
                forecast_val = service_df['loading_cleaned'].mean()
                moe_val = 1.96 * service_df['loading_cleaned'].std()
                method = f"Rata-rata Historis (SARIMA Gagal, {num_outliers} outlier dibersihkan)"
        else:
            forecast_val = service_df['loading_cleaned'].mean()
            moe_val = 1.96 * service_df['loading_cleaned'].std()
            method = f"Rata-rata Historis ({num_outliers} outlier dibersihkan)"

        all_results.append({
            "Service": service,
            "Prediksi Loading Berikutnya": forecast_val,
            "Margin of Error (± box)": moe_val,
            "Keterangan": method
        })
        
    progress_bar.empty() # Hapus progress bar setelah selesai
    return pd.DataFrame(all_results)


def render_forecast_tab():
    """Fungsi untuk menampilkan seluruh konten tab forecasting."""
    st.header("📈 Prediksi Loading Kapal per Service")
    st.write("Fitur ini memprediksi jumlah muatan (loading) untuk kedatangan kapal berikutnya berdasarkan data historis. Proses ini sudah termasuk pembersihan data anomali (*outlier*) untuk hasil yang lebih akurat.")
    
    st.info("Pastikan file `History Loading.xlsx - Sheet1.csv` ada di dalam repository GitHub Anda.", icon="ℹ️")

    if st.button("🚀 Buat Prediksi Loading", type="primary"):
        df_history = load_history_data()
        
        if df_history is not None:
            with st.spinner("Memproses data historis dan melatih model... Ini mungkin memakan waktu beberapa saat."):
                forecast_df = run_forecasting_process(df_history)
            
            st.success("Prediksi berhasil dibuat!")
            
            # Memformat hasil untuk tampilan yang lebih baik
            forecast_df['Prediksi Loading Berikutnya'] = forecast_df['Prediksi Loading Berikutnya'].round(2)
            forecast_df['Margin of Error (± box)'] = forecast_df['Margin of Error (± box)'].fillna(0).round(2)
            
            st.dataframe(
                forecast_df.sort_values(by="Prediksi Loading Berikutnya", ascending=False).reset_index(drop=True),
                use_container_width=True,
                hide_index=True
            )
            
            st.markdown("---")
            st.subheader("💡 Apa Arti Hasil Di Atas?")
            st.markdown("""
            - **Prediksi Loading Berikutnya**: Estimasi jumlah box untuk kedatangan kapal selanjutnya dari service tersebut.
            - **Margin of Error (± box)**: Tingkat ketidakpastian prediksi. Contoh: Prediksi **300** dengan MoE **±50** berarti nilai aktual kemungkinan besar berada di antara **250** dan **350**.
            - **Keterangan**: Metode yang digunakan. **SARIMA** adalah model statistik time-series, sedangkan **Rata-rata Historis** digunakan jika data terlalu sedikit untuk model kompleks. Keterangan juga menunjukkan jumlah *outlier* yang telah ditangani.
            """)

# --- FUNGSI-FUNGSI INTI UNTUK CLASH MONITORING (TETAP SAMA) ---
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
                st.error(f"Failed to read file '{filename}': {e}"); return None
    st.error(f"Vessel code file not found."); return None

# --- STRUKTUR UTAMA DENGAN TABS ---
tab1, tab2 = st.tabs(["🚨 Clash Monitoring", "📈 Loading Forecast"])

# --- KONTEN TAB 1: CLASH MONITORING (KODE LAMA ANDA) ---
with tab1:
    st.sidebar.header("⚙️ Your File Uploads")
    schedule_file = st.sidebar.file_uploader("1. Upload Vessel Schedule", type=['xlsx', 'csv'])
    unit_list_file = st.sidebar.file_uploader("2. Upload Unit List", type=['xlsx', 'csv'])
    process_button = st.sidebar.button("🚀 Process Clash Data", type="primary")

    if 'processed_df' not in st.session_state:
        st.session_state.processed_df = None
    if 'clash_summary_df' not in st.session_state:
        st.session_state.clash_summary_df = None

    df_vessel_codes = load_vessel_codes_from_repo()

    if process_button:
        if schedule_file and unit_list_file and (df_vessel_codes is not None and not df_vessel_codes.empty):
            with st.spinner('Loading and processing data...'):
                try:
                    # 1. Loading & Cleaning
                    if schedule_file.name.lower().endswith(('.xls', '.xlsx')): df_schedule = pd.read_excel(schedule_file)
                    else: df_schedule = pd.read_csv(schedule_file)
                    df_schedule.columns = df_schedule.columns.str.strip()
                    if unit_list_file.name.lower().endswith(('.xls', '.xlsx')): df_unit_list = pd.read_excel(unit_list_file)
                    else: df_unit_list = pd.read_csv(unit_list_file)
                    df_unit_list.columns = df_unit_list.columns.str.strip()
                    
                    # 2. Main Processing
                    original_vessels_list = df_schedule['VESSEL'].unique().tolist()
                    df_schedule['ETA'] = pd.to_datetime(df_schedule['ETA'], errors='coerce')
                    df_schedule_with_code = pd.merge(df_schedule, df_vessel_codes, left_on="VESSEL", right_on="Description", how="left").rename(columns={"Value": "CODE"})
                    merged_df = pd.merge(df_schedule_with_code, df_unit_list, left_on=['CODE', 'VOY_OUT'], right_on=['Carrier Out', 'Voyage Out'], how='inner')
                    if merged_df.empty: st.warning("No matching data found."); st.session_state.processed_df = None; st.stop()
                    
                    # 3. Filtering
                    merged_df = merged_df[merged_df['VESSEL'].isin(original_vessels_list)]
                    excluded_areas = [str(i) for i in range(801, 809)]
                    merged_df['Area (EXE)'] = merged_df['Area (EXE)'].astype(str)
                    filtered_data = merged_df[~merged_df['Area (EXE)'].isin(excluded_areas)]
                    if filtered_data.empty: st.warning("No data remaining after filtering."); st.session_state.processed_df = None; st.stop()

                    # 4. Pivoting
                    grouping_cols = ['VESSEL', 'CODE', 'VOY_OUT', 'ETA']
                    pivot_df = filtered_data.pivot_table(index=grouping_cols, columns='Area (EXE)', aggfunc='size', fill_value=0)
                    
                    cluster_cols_for_calc = pivot_df.columns.tolist()
                    pivot_df['TTL BOX'] = pivot_df[cluster_cols_for_calc].sum(axis=1)
                    pivot_df['TTL CLSTR'] = (pivot_df[cluster_cols_for_calc] > 0).sum(axis=1)
                    pivot_df = pivot_df.reset_index()
                    
                    # 5. Conditional Filtering
                    two_days_ago = pd.Timestamp.now() - timedelta(days=2)
                    condition_to_hide = (pivot_df['ETA'] < two_days_ago) & (pivot_df['TTL BOX'] < 50)
                    pivot_df = pivot_df[~condition_to_hide]
                    if pivot_df.empty: st.warning("No data remaining after ETA & Total filter."); st.session_state.processed_df = None; st.stop()

                    # 6. Sorting and Ordering
                    cols_awal = ['VESSEL', 'CODE', 'VOY_OUT', 'ETA', 'TTL BOX', 'TTL CLSTR']
                    final_cluster_cols = [col for col in pivot_df.columns if col not in cols_awal]
                    final_display_cols = cols_awal + sorted(final_cluster_cols)
                    pivot_df = pivot_df[final_display_cols]
                    
                    pivot_df['ETA'] = pd.to_datetime(pivot_df['ETA']).dt.strftime('%Y-%m-%d %H:%M')
                    
                    pivot_df = pivot_df.sort_values(by='ETA', ascending=True).reset_index(drop=True)
                    
                    st.session_state.processed_df = pivot_df
                    st.success("Data processed successfully!")

                except Exception as e:
                    st.error(f"An error occurred during processing: {e}")
                    st.session_state.processed_df = None
        else:
            st.warning("Please upload both files.")

    # --- Area Tampilan ---
    if st.session_state.processed_df is not None:
        display_df = st.session_state.processed_df
        
        st.header("📋 Analysis Result")

        # --- Persiapan untuk Styling AG Grid dan Summary ---
        df_for_grid = display_df.copy()
        df_for_grid['ETA_Date'] = pd.to_datetime(df_for_grid['ETA']).dt.strftime('%Y-%m-%d')
        
        unique_dates = df_for_grid['ETA_Date'].unique()
        zebra_colors = ['#F8F0E5', '#DAC0A3'] 
        date_color_map = {date: zebra_colors[i % 2] for i, date in enumerate(unique_dates)}

        clash_map = {}
        cluster_cols = [col for col in df_for_grid.columns if col not in ['VESSEL', 'CODE', 'VOY_OUT', 'ETA', 'TTL BOX', 'TTL CLSTR', 'ETA_Date']]
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

            with st.expander("Show Clash Summary", expanded=True):
                total_clash_days = len(clash_map)
                total_conflicting_blocks = sum(len(areas) for areas in clash_map.values())
                st.markdown(f"**🔥 Found {total_clash_days} clash day(s) with a total of {total_conflicting_blocks} conflicting blocks.**")
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
                            <strong style='font-size: 1.2em;'>Clash on: {date}</strong>
                            <hr style='margin: 10px 0;'>
                            <div style='line-height: 1.7;'>
                        """
                        for area in sorted(filtered_areas):
                            clashing_rows = df_for_grid[(df_for_grid['ETA_Date'] == date) & (df_for_grid[area] > 0)]
                            clashing_vessels = clashing_rows['VESSEL'].tolist()
                            total_clash_boxes = clashing_rows[area].sum()
                            vessel_list_str = ", ".join(clashing_vessels)
                            
                            summary_html += f"<b>Block {area}</b> (<span style='color:#E67E22; font-weight:bold;'>{total_clash_boxes} boxes</span>):<br><small>{vessel_list_str}</small><br>"
                            
                            summary_data.append({
                                "Clash Date": date,
                                "Block": area,
                                "Total Boxes": total_clash_boxes,
                                "Vessels": vessel_list_str,
                                "Remark": ""
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
        pinned_cols = ['VESSEL', 'CODE', 'VOY_OUT', 'ETA', 'TTL BOX', 'TTL CLSTR']
        for col in pinned_cols:
            width = 110 if col == 'VESSEL' else 80
            if col == 'ETA': width = 120 
            col_def = {"field": col, "headerName": col, "pinned": "left", "width": width}
            if col in ["TTL BOX", "TTL CLSTR"]: col_def["cellRenderer"] = hide_zero_jscode
            column_defs.append(col_def)
        for col in cluster_cols:
            column_defs.append({"field": col, "headerName": col, "width": 60, "cellRenderer": hide_zero_jscode, "cellStyle": clash_cell_style_jscode})
        column_defs.append({"field": "ETA_Date", "hide": True})
        gridOptions = {"defaultColDef": default_col_def, "columnDefs": column_defs, "getRowStyle": zebra_row_style_jscode}

        AgGrid(df_for_grid, gridOptions=gridOptions, height=600, width='100%', theme='streamlit', allow_unsafe_jscode=True)
        
        # --- LOGIKA TOMBOL DOWNLOAD EXCEL ---
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            if 'clash_summary_df' in st.session_state and st.session_state.clash_summary_df is not None:
                summary_df_for_export = st.session_state.clash_summary_df
                
                final_summary_export = []
                last_date = None
                if not summary_df_for_export.empty:
                    for index, row in summary_df_for_export.iterrows():
                        current_date = row['Clash Date']
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
            label="📥 Download Clash Summary (Excel)",
            data=output.getvalue(),
            file_name="clash_summary_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
# --- KONTEN TAB 2: FORECASTING (KODE BARU) ---
with tab2:
    render_forecast_tab()
