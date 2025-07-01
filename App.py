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
st.title("Yard Operations Analyzer")

# --- FUNGSI-FUNGSI UNTUK FORECASTING (MODEL RANDOM FOREST) ---

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
            df = df[df['loading'] >= 0] # Pastikan tidak ada loading negatif
            return df
        except Exception as e:
            st.error(f"Gagal memuat file histori '{filename}': {e}")
            return None
    st.warning(f"File histori '{filename}' tidak ditemukan di repository.")
    return None

def create_features(df):
    """Membuat fitur dari data mentah untuk model Machine Learning."""
    df['hour'] = df['ata'].dt.hour
    df['day_of_week'] = df['ata'].dt.dayofweek # Senin=0, Minggu=6
    df['day_of_month'] = df['ata'].dt.day
    df['day_of_year'] = df['ata'].dt.dayofyear
    df['week_of_year'] = df['ata'].dt.isocalendar().week.astype(int)
    df['month'] = df['ata'].dt.month
    df['year'] = df['ata'].dt.year
    
    # One-Hot Encode untuk kolom 'service'
    df_with_dummies = pd.get_dummies(df, columns=['service'], prefix='service')
    return df_with_dummies

def train_evaluate_model(df_features):
    """Melatih model Random Forest dan mengevaluasi kinerjanya."""
    # Definisikan target dan fitur
    target = 'loading'
    features = [col for col in df_features.columns if col not in ['ata', 'loading']]
    
    X = df_features[features]
    y = df_features[target]
    
    # Split data: 80% untuk latihan, 20% untuk pengujian
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    
    # Inisialisasi dan latih model
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, min_samples_leaf=2)
    model.fit(X_train, y_train)
    
    # Buat prediksi pada data uji
    predictions = model.predict(X_test)
    
    # Hitung metrik evaluasi
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mape = mean_absolute_percentage_error(y_test, predictions) * 100
    
    # Simpan hasil untuk visualisasi
    results = {
        "model": model,
        "features": features,
        "X_test": X_test,
        "y_test": y_test,
        "predictions": predictions,
        "mae": mae,
        "rmse": rmse,
        "mape": mape
    }
    return results

def render_forecast_tab():
    """Fungsi untuk menampilkan seluruh konten tab forecasting."""
    st.header("📈 Prediksi Loading dengan Machine Learning")
    st.write("""
    Fitur ini menggunakan model **Random Forest** untuk memprediksi jumlah muatan. 
    Model ini belajar dari berbagai faktor seperti waktu (jam, hari, bulan) dan jenis service untuk memberikan prediksi yang lebih akurat.
    """)
    st.info("Pastikan file `History Loading.xlsx` ada di dalam repository GitHub Anda.", icon="ℹ️")

    if st.button("🚀 Latih Model & Evaluasi Kinerja", type="primary"):
        df_history = load_history_data()
        
        if df_history is not None:
            with st.spinner("Membuat fitur dan melatih model..."):
                df_features = create_features(df_history)
                model_results = train_evaluate_model(df_features)
                
                # Simpan hasil ke session state untuk digunakan nanti
                st.session_state.model_results = model_results
                st.session_state.unique_services = df_history['service'].unique().tolist()
            
            st.success("Model berhasil dilatih dan dievaluasi!")
        else:
            st.error("Data historis tidak dapat dimuat. Proses dibatalkan.")

    # Tampilkan hasil jika model sudah dilatih
    if 'model_results' in st.session_state:
        results = st.session_state.model_results
        
        st.markdown("---")
        st.subheader("📊 Kinerja Model pada Data Uji")
        col1, col2, col3 = st.columns(3)
        col1.metric("MAPE (Mean Absolute Percentage Error)", f"{results['mape']:.2f} %", help="Rata-rata persentase kesalahan. Semakin kecil, semakin baik.")
        col2.metric("MAE (Mean Absolute Error)", f"{results['mae']:.2f} box", help="Rata-rata kesalahan absolut dalam satuan box.")
        col3.metric("RMSE (Root Mean Squared Error)", f"{results['rmse']:.2f} box", help="Akar dari rata-rata kuadrat kesalahan. Memberi bobot lebih pada kesalahan besar.")

        # Visualisasi Fitur Penting
        st.subheader("🔍 Fitur Paling Berpengaruh")
        feature_importance = pd.Series(results['model'].feature_importances_, index=results['features']).nlargest(10)
        fig, ax = plt.subplots()
        feature_importance.sort_values().plot(kind='barh', ax=ax)
        ax.set_title("Top 10 Fitur Paling Penting")
        st.pyplot(fig)

        # Visualisasi Prediksi vs Aktual
        st.subheader("📉 Prediksi Model vs. Nilai Aktual (pada Data Uji)")
        plot_df = pd.DataFrame({'Aktual': results['y_test'], 'Prediksi': results['predictions']})
        st.line_chart(plot_df)

        # --- Bagian untuk membuat prediksi baru ---
        st.markdown("---")
        st.subheader("🔮 Buat Prediksi Baru")
        
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                service_to_predict = st.selectbox("Pilih Service", options=st.session_state.unique_services)
            with col2:
                eta_date = st.date_input("Pilih Tanggal ETA")
            with col3:
                eta_time = st.time_input("Pilih Waktu ETA")
            
            submit_button = st.form_submit_button("Prediksi Loading")

        if submit_button:
            # Gabungkan tanggal dan waktu
            eta_full = datetime.combine(eta_date, eta_time)
            
            # Buat DataFrame untuk data baru
            new_data = pd.DataFrame([{'ata': eta_full, 'service': service_to_predict}])
            new_data_features = create_features(new_data)
            
            # Pastikan semua kolom fitur ada
            model_features = results['features']
            new_data_aligned = pd.DataFrame(columns=model_features)
            new_data_aligned = pd.concat([new_data_aligned, new_data_features], axis=0, ignore_index=True).fillna(0)
            new_data_aligned = new_data_aligned[model_features] # Jaga urutan kolom
            
            # Lakukan prediksi
            prediction = results['model'].predict(new_data_aligned)[0]
            
            st.success(f"**Prediksi jumlah loading untuk service `{service_to_predict}` pada `{eta_full.strftime('%d-%m-%Y %H:%M')}` adalah: `{prediction:.2f} box`**")


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
                    # ( ... Kode proses clash monitoring Anda tetap sama persis di sini ... )
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
        # ( ... Kode tampilan AG-Grid dan download Anda tetap sama persis di sini ... )
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
