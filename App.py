import streamlit as st
import pandas as pd
import os
from datetime import datetime, timedelta
import json
import io
import warnings
import numpy as np

# --- Libraries for Machine Learning ---
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error

# --- Libraries for UI and PDF Generation ---
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode
from fpdf import FPDF # Library untuk membuat PDF

# Ignore non-critical warnings
warnings.filterwarnings("ignore", category=UserWarning)

# --- Page Configuration & Title ---
st.set_page_config(page_title="Yard Cluster Monitoring", layout="wide")
st.title("Yard Cluster Monitoring")

# --- KELAS UNTUK MEMBUAT LAPORAN PDF ---
class PDFReport(FPDF):
    def header(self):
        """Mendefinisikan header untuk setiap halaman PDF."""
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Yard Cluster Analysis Report', 0, 1, 'C')
        self.set_font('Arial', '', 8)
        self.cell(0, 5, f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        """Mendefinisikan footer untuk setiap halaman PDF."""
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def add_section_title(self, title):
        """Fungsi bantuan untuk membuat judul bagian."""
        self.set_font('Arial', 'B', 11)
        self.set_fill_color(230, 230, 230)
        self.cell(0, 8, title, 0, 1, 'L', fill=True)
        self.ln(4)

    def create_table_from_df(self, df, col_widths=None):
        """Membuat tabel dari Pandas DataFrame dengan penanganan error encoding."""
        if df.empty:
            self.set_font('Arial', 'I', 10)
            self.cell(0, 10, "No data available for this section.", 0, 1)
            self.ln(5)
            return

        self.set_font('Arial', 'B', 8)
        header = df.columns.tolist()

        if col_widths is None:
            page_width = self.w - 2 * self.l_margin
            col_widths = [page_width / len(header)] * len(header)

        # Header Tabel
        for i, h in enumerate(header):
            self.cell(col_widths[i], 7, str(h), 1, 0, 'C')
        self.ln()

        # Body Tabel
        self.set_font('Arial', '', 7)
        for _, row in df.iterrows():
            for i, h in enumerate(header):
                data = row[h]
                cell_text = "" if pd.isna(data) else str(data)
                
                # Membersihkan teks untuk memastikan kompatibel dengan encoding latin-1
                # Karakter yang tidak didukung akan diganti dengan '?'
                sanitized_text = cell_text.encode('latin-1', 'replace').decode('latin-1')
                
                self.cell(col_widths[i], 6, sanitized_text, 1, 0, 'L')
            self.ln()
        self.ln(10)

# --- FUNCTIONS FOR FORECASTING ---
@st.cache_data
def load_history_data(filename="History Loading.xlsx"):
    """Finds and loads the historical data file for forecasting."""
    if os.path.exists(filename):
        try:
            df = pd.read_excel(filename) if filename.lower().endswith(('.xls', '.xlsx')) else pd.read_csv(filename)
            df.columns = [col.strip().lower() for col in df.columns]
            df['ata'] = pd.to_datetime(df['ata'], dayfirst=True, errors='coerce')
            df.dropna(subset=['ata', 'loading', 'service'], inplace=True)
            df = df[df['loading'] >= 0]
            return df
        except Exception as e:
            st.error(f"Failed to load history file '{filename}': {e}")
            return None
    st.warning(f"History file '{filename}' not found in the repository.")
    return None

def create_time_features(df):
    """Creates time-based features from the 'ata' column."""
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
    """Runs the outlier cleaning and forecasting process for each service."""
    all_results = []
    unique_services = df_history['service'].unique()
    progress_bar = st.progress(0, text="Analyzing services...")
    total_services = len(unique_services)
    for i, service in enumerate(unique_services):
        progress_text = f"Analyzing service: {service} ({i+1}/{total_services})"
        progress_bar.progress((i + 1) / total_services, text=progress_text)
        service_df = df_history[df_history['service'] == service].copy()
        if service_df.empty or service_df['loading'].isnull().all():
            continue
        Q1 = service_df['loading'].quantile(0.25)
        Q3 = service_df['loading'].quantile(0.75)
        IQR = Q3 - Q1
        upper_bound = Q3 + 1.5 * IQR
        lower_bound = Q1 - 1.5 * IQR
        num_outliers = ((service_df['loading'] < lower_bound) | (service_df['loading'] > upper_bound)).sum()
        service_df['loading_cleaned'] = service_df['loading'].clip(lower=lower_bound, upper=upper_bound)
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
                    raise ValueError("Not enough data to train the model.")
                model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1, min_samples_leaf=2)
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                mape_val = mean_absolute_percentage_error(y_test, predictions) * 100 if len(y_test) > 0 else 0
                moe_val = 1.96 * np.std(y_test - predictions) if len(y_test) > 0 else 0
                future_eta = datetime.now().replace(hour=12, minute=0, second=0, microsecond=0) + timedelta(days=1)
                future_df = create_time_features(pd.DataFrame([{'ata': future_eta}]))
                forecast_val = model.predict(future_df[features_to_use])[0]
                method = f"Random Forest ({num_outliers} outliers cleaned)"
            except Exception:
                forecast_val = service_df['loading_cleaned'].mean()
                moe_val = 1.96 * service_df['loading_cleaned'].std()
                actuals = service_df['loading_cleaned']
                mape_val = np.mean(np.abs((actuals - forecast_val) / actuals)) * 100 if not actuals.empty else 0
                method = f"Historical Average (RF Failed, {num_outliers} outliers cleaned)"
        else:
            forecast_val = service_df['loading_cleaned'].mean()
            moe_val = 1.96 * service_df['loading_cleaned'].std()
            actuals = service_df['loading_cleaned']
            mape_val = np.mean(np.abs((actuals - forecast_val) / actuals)) * 100 if not actuals.empty else 0
            method = f"Historical Average ({num_outliers} outliers cleaned)"
        all_results.append({
            "Service": service, "Loading Forecast": max(0, forecast_val),
            "Margin of Error (± box)": moe_val, "MAPE (%)": mape_val, "Method": method
        })
    progress_bar.empty()
    return pd.DataFrame(all_results)

def render_forecast_tab():
    """Function to display the entire content of the forecasting tab."""
    st.header("📈 Loading Forecast with Machine Learning")
    st.write("This feature uses a separate **Random Forest** model for each service. The model learns from historical time patterns to provide more accurate predictions, complete with anomaly data cleaning.")
    if 'forecast_df' not in st.session_state:
        df_history = load_history_data()
        if df_history is not None:
            with st.spinner("Processing data and training models for each service..."):
                forecast_df = run_per_service_rf_forecast(df_history)
                st.session_state.forecast_df = forecast_df
        else:
            st.session_state.forecast_df = pd.DataFrame()
            st.error("Could not load historical data. Process canceled.")
    if 'forecast_df' in st.session_state and not st.session_state.forecast_df.empty:
        results_df = st.session_state.forecast_df.copy()
        results_df['Loading Forecast'] = results_df['Loading Forecast'].round(2)
        results_df['Margin of Error (± box)'] = results_df['Margin of Error (± box)'].fillna(0).round(2)
        results_df['MAPE (%)'] = results_df['MAPE (%)'].replace([np.inf, -np.inf], 0).fillna(0).round(2)
        st.markdown("---")
        st.subheader("📊 Forecast Results per Service")
        st.dataframe(
            results_df.sort_values(by="Loading Forecast", ascending=False).reset_index(drop=True),
            use_container_width=True, hide_index=True,
            column_config={"MAPE (%)": st.column_config.NumberColumn(format="%.2f%%")}
        )
        st.markdown("---")
        st.subheader("💡 How to Read These Results")
        st.markdown("- **Loading Forecast**: The estimated number of boxes for the next vessel arrival of that service.\n- **Margin of Error (± box)**: The level of uncertainty in the prediction. A prediction of **300** with a MoE of **±50** means the actual value is likely between **250** and **350**.\n- **MAPE (%)**: The average percentage error of the model when tested on its historical data. **The smaller the value, the more accurate the model has been in the past.**\n- **Method**: The technique used for the forecast and the number of outliers handled.")
    else:
        st.warning("No forecast data could be generated. The history file might be empty or contain no valid service data.")

@st.cache_data
def load_vessel_codes_from_repo(possible_names=['vessel codes.xlsx', 'vessel_codes.xls', 'vessel_codes.csv']):
    """Finds and loads the vessel codes file."""
    for filename in possible_names:
        if os.path.exists(filename):
            try:
                if filename.lower().endswith(('.csv')): df = pd.read_csv(filename)
                else: df = pd.read_excel(filename)
                df.columns = df.columns.str.strip()
                return df
            except Exception as e:
                st.error(f"Failed to read file '{filename}': {e}"); return None
    st.error(f"Vessel codes file not found."); return None

def render_clash_tab():
    """Function to display the entire content of the Clash Analysis tab."""
    st.sidebar.header("⚙️ Upload Your Files")
    schedule_file = st.sidebar.file_uploader("1. Upload Vessel Schedule", type=['xlsx', 'csv'])
    unit_list_file = st.sidebar.file_uploader("2. Upload Unit List", type=['xlsx', 'csv'])
    process_button = st.sidebar.button("🚀 Process Clash Data", type="primary")

    # PERBAIKAN 1: Inisialisasi variabel untuk menghindari error
    summary_display = pd.DataFrame()
    clash_summary_df = pd.DataFrame()

    if 'processed_df' not in st.session_state: st.session_state.processed_df = None
    if 'clash_summary_df' not in st.session_state: st.session_state.clash_summary_df = None
    
    df_vessel_codes = load_vessel_codes_from_repo()
    if process_button:
        if schedule_file and unit_list_file and (df_vessel_codes is not None and not df_vessel_codes.empty):
            with st.spinner('Loading and processing data...'):
                try:
                    # ( ... Logika pemrosesan data Anda yang kompleks ada di sini ... )
                    # (Saya sederhanakan untuk contoh, gunakan logika Anda yang sudah ada)
                    df_schedule = pd.read_excel(schedule_file) if schedule_file.name.lower().endswith(('.xls', '.xlsx')) else pd.read_csv(schedule_file)
                    df_unit_list = pd.read_excel(unit_list_file) if unit_list_file.name.lower().endswith(('.xls', '.xlsx')) else pd.read_csv(unit_list_file)
                    df_schedule.columns = [col.strip().upper() for col in df_schedule.columns] 
                    df_unit_list.columns = [col.strip() for col in df_unit_list.columns]
                    df_schedule['ETA'] = pd.to_datetime(df_schedule['ETA'], errors='coerce')
                    df_schedule['CLOSING PHYSIC'] = pd.to_datetime(df_schedule['CLOSING PHYSIC'], errors='coerce')
                    df_schedule_with_code = pd.merge(df_schedule, df_vessel_codes, left_on="VESSEL", right_on="Description", how="left").rename(columns={"Value": "CODE"})
                    merged_df = pd.merge(df_schedule_with_code, df_unit_list, left_on=['CODE', 'VOY_OUT'], right_on=['Carrier Out', 'Voyage Out'], how='inner')
                    
                    if merged_df.empty:
                        st.warning("No matching data found.")
                        st.session_state.processed_df = None
                        return

                    pivot_df = merged_df.pivot_table(index=['VESSEL', 'CODE', 'SERVICE', 'VOY_OUT', 'ETA', 'CLOSING PHYSIC'], columns='Area (EXE)', aggfunc='size', fill_value=0)
                    pivot_df['TOTAL BOX'] = pivot_df.sum(axis=1)
                    clstr_cols = [col for col in pivot_df.columns if col not in ['TOTAL BOX', 'D01', 'C01', 'C02', 'OOG', 'UNKNOWN', 'BR9', 'RC9']]
                    pivot_df['TOTAL CLSTR'] = (pivot_df[clstr_cols] > 0).sum(axis=1)
                    pivot_df = pivot_df.reset_index()
                    st.session_state.processed_df = pivot_df.sort_values(by='ETA', ascending=True)
                    st.success("Data processed successfully!")

                except Exception as e:
                    st.error(f"An error occurred during processing: {e}")
                    st.session_state.processed_df = None
        else:
            st.warning("Please upload both files.")

    if st.session_state.get('processed_df') is not None:
        display_df = st.session_state.processed_df.copy()
        display_df['ETA_str'] = pd.to_datetime(display_df['ETA']).dt.strftime('%Y-%m-%d %H:%M')
        display_df['CLOSING_PHYSIC_str'] = pd.to_datetime(display_df['CLOSING PHYSIC']).dt.strftime('%Y-%m-%d %H:%M')
        
        st.subheader("🚢 Upcoming Vessel Summary (Today + Next 3 Days)")
        forecast_df = st.session_state.get('forecast_df')
        if forecast_df is not None and not forecast_df.empty:
            today = pd.to_datetime(datetime.now().date())
            four_days_later = today + timedelta(days=4)
            upcoming_vessels_df = display_df[(display_df['ETA'] >= today) & (display_df['ETA'] < four_days_later)].copy()
            if not upcoming_vessels_df.empty:
                # ... (Logika sidebar Anda di sini) ...
                summary_df = pd.merge(upcoming_vessels_df, forecast_df[['Service', 'Loading Forecast']], left_on='SERVICE', right_on='Service', how='left')
                summary_df['Loading Forecast'] = summary_df['Loading Forecast'].fillna(0).round(0).astype(int)
                summary_df['DIFF'] = summary_df['TOTAL BOX'] - summary_df['Loading Forecast']
                summary_df['base_for_req'] = summary_df[['TOTAL BOX', 'Loading Forecast']].max(axis=1)
                def get_clstr_requirement(value):
                    if value <= 450: return 4
                    elif 451 <= value <= 600: return 5
                    elif 601 <= value <= 800: return 6
                    else: return 8
                summary_df['CLSTR REQ'] = summary_df['base_for_req'].apply(get_clstr_requirement)
                summary_display = summary_df[['VESSEL', 'SERVICE', 'ETA_str', 'CLOSING_PHYSIC_str', 'TOTAL BOX', 'Loading Forecast', 'DIFF', 'TOTAL CLSTR', 'CLSTR REQ']].rename(columns={
                    'ETA_str': 'ETA', 'CLOSING_PHYSIC_str': 'CLOSING TIME', 'TOTAL BOX': 'BOX STACKED', 'Loading Forecast': 'LOADING FORECAST'
                })
                st.dataframe(summary_display, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        st.header("📋 Detailed Analysis Results")
        # Logika untuk membuat clash_summary_df
        df_for_grid = display_df.copy()
        df_for_grid['ETA_Date'] = pd.to_datetime(df_for_grid['ETA']).dt.strftime('%Y-%m-%d')
        clash_map = {}
        cluster_cols = [col for col in df_for_grid.columns if col not in ['VESSEL', 'CODE', 'SERVICE', 'VOY_OUT', 'ETA', 'CLOSING PHYSIC', 'TOTAL BOX', 'TOTAL CLSTR', 'ETA_Date', 'ETA_str', 'CLOSING_PHYSIC_str']]
        summary_data = []
        for date, group in df_for_grid.groupby('ETA_Date'):
            for col in cluster_cols:
                if (group[col] > 0).sum() > 1:
                    clashing_rows = df_for_grid[(df_for_grid['ETA_Date'] == date) & (df_for_grid[col] > 0)]
                    clashing_vessels = clashing_rows['VESSEL'].tolist()
                    total_clash_boxes = clashing_rows[col].sum()
                    vessel_list_str = ", ".join(clashing_vessels)
                    summary_data.append({"Clash Date": date, "Block": col, "Total Boxes": total_clash_boxes, "Vessel(s)": vessel_list_str})
        if summary_data:
            clash_summary_df = pd.DataFrame(summary_data)
            st.session_state.clash_summary_df = clash_summary_df # Simpan ke session state
        
        # Tampilkan AgGrid
        AgGrid(df_for_grid, height=600, width='100%', theme='streamlit')


        # --- DOWNLOAD CENTER ---
        st.markdown("---")
        st.subheader("📥 Download Center")
        col1, col2 = st.columns(2)

        with col1: # Tombol Download Excel
            output_excel = io.BytesIO()
            try:
                with pd.ExcelWriter(output_excel, engine='xlsxwriter') as writer:
                    if not display_df.empty:
                        display_df.drop(columns=['ETA_str', 'CLOSING_PHYSIC_str'], errors='ignore').to_excel(writer, sheet_name='Detailed Analysis', index=False)
                    if not summary_display.empty:
                        summary_display.to_excel(writer, sheet_name='Upcoming Summary', index=False)
                    if not clash_summary_df.empty:
                        clash_summary_df.to_excel(writer, sheet_name='Clash Summary', index=False)
                
                if output_excel.tell() > 0:
                    st.download_button("📥 Download as Excel", output_excel.getvalue(), "clash_analysis_report.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
            except Exception as e: st.error(f"Failed to create Excel file: {e}")

        # Tombol Download PDF (Menggunakan kode yang sudah Anda konfirmasi)
        with col2:
            try:
                pdf = PDFReport()
                pdf.add_page(orientation='L') # Landscape

                if not summary_display.empty:
                    pdf.add_section_title("Upcoming Vessel Summary")
                    summary_widths = [35, 20, 35, 35, 25, 30, 15, 25, 25]
                    pdf.create_table_from_df(summary_display, col_widths=summary_widths)

                # Gunakan variabel clash_summary_df yang sudah kita definisikan
                if not clash_summary_df.empty:
                    pdf.add_section_title("Clash Summary")
                    clash_pdf_df = clash_summary_df.rename(columns={"Total Boxes": "Boxes", "Vessel(s)": "Vessels"})
                    clash_widths = [30, 25, 25, 120, 40]
                    pdf.create_table_from_df(clash_pdf_df, col_widths=clash_widths)
                
                pdf_output = pdf.output(dest='S').encode('latin-1')
                st.download_button(
                    label="📄 Download as PDF",
                    data=pdf_output,
                    file_name="clash_analysis_report.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Failed to create PDF file: {e}")

# --- MAIN STRUCTURE WITH TABS ---
tab1, tab2 = st.tabs(["🚨 Clash Analysis", "📈 Loading Forecast"])
with tab1:
    render_clash_tab()
with tab2:
    render_forecast_tab()
