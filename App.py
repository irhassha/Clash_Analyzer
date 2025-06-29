import streamlit as st
import pandas as pd
import os
from datetime import datetime, timedelta
# Import pustaka baru
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode

# --- Konfigurasi Halaman & Judul ---
st.set_page_config(page_title="Clash Analyzer", layout="wide")
st.title("🚢 Vessel Clash Analyzer")

# --- Fungsi-fungsi Inti (tidak ada fungsi styling lama) ---
@st.cache_data
def load_vessel_codes_from_repo(possible_names=['vessel codes.xlsx', 'vessel_codes.xls', 'vessel_codes.csv']):
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

# --- Sidebar & Proses Utama ---
st.sidebar.header("⚙️ Your File Uploads")
schedule_file = st.sidebar.file_uploader("1. Upload Vessel Schedule", type=['xlsx', 'csv'])
unit_list_file = st.sidebar.file_uploader("2. Upload Unit List", type=['xlsx', 'csv'])
process_button = st.sidebar.button("🚀 Process Data", type="primary")

if 'processed_df' not in st.session_state:
    st.session_state.processed_df = None

df_vessel_codes = load_vessel_codes_from_repo()

if process_button:
    if schedule_file and unit_list_file and (df_vessel_codes is not None and not df_vessel_codes.empty):
        with st.spinner('Loading and processing data...'):
            try:
                # ... (semua proses loading hingga pivot tidak berubah) ...
                if schedule_file.name.lower().endswith(('.xls', '.xlsx')): df_schedule = pd.read_excel(schedule_file)
                else: df_schedule = pd.read_csv(schedule_file)
                df_schedule.columns = df_schedule.columns.str.strip()
                if unit_list_file.name.lower().endswith(('.xls', '.xlsx')): df_unit_list = pd.read_excel(unit_list_file)
                else: df_unit_list = pd.read_csv(unit_list_file)
                df_unit_list.columns = df_unit_list.columns.str.strip()
                original_vessels_list = df_schedule['VESSEL'].unique().tolist()
                df_schedule['ETA'] = pd.to_datetime(df_schedule['ETA'], errors='coerce')
                df_schedule_with_code = pd.merge(df_schedule, df_vessel_codes, left_on="VESSEL", right_on="Description", how="left").rename(columns={"Value": "CODE"})
                merged_df = pd.merge(df_schedule_with_code, df_unit_list, left_on=['CODE', 'VOY_OUT'], right_on=['Carrier Out', 'Voyage Out'], how='inner')
                if merged_df.empty: st.warning("No matching data found."); st.session_state.processed_df = None; st.stop()
                merged_df = merged_df[merged_df['VESSEL'].isin(original_vessels_list)]
                excluded_areas = [str(i) for i in range(801, 809)]
                merged_df['Area (EXE)'] = merged_df['Area (EXE)'].astype(str)
                filtered_data = merged_df[~merged_df['Area (EXE)'].isin(excluded_areas)]
                if filtered_data.empty: st.warning("No data remaining after filtering."); st.session_state.processed_df = None; st.stop()
                grouping_cols = ['VESSEL', 'CODE', 'VOY_OUT', 'ETA']
                pivot_df = filtered_data.pivot_table(index=grouping_cols, columns='Area (EXE)', aggfunc='size', fill_value=0)
                cluster_cols_for_calc = pivot_df.columns.tolist()
                pivot_df['Total Box'] = pivot_df[cluster_cols_for_calc].sum(axis=1)
                pivot_df['Total cluster'] = (pivot_df[cluster_cols_for_calc] > 0).sum(axis=1)
                pivot_df = pivot_df.reset_index()
                two_days_ago = pd.Timestamp.now() - timedelta(days=2)
                condition_to_hide = (pivot_df['ETA'] < two_days_ago) & (pivot_df['Total Box'] < 50)
                pivot_df = pivot_df[~condition_to_hide]
                if pivot_df.empty: st.warning("No data remaining after ETA & Total filter."); st.session_state.processed_df = None; st.stop()
                cols_awal = ['VESSEL', 'CODE', 'VOY_OUT', 'ETA', 'Total Box', 'Total cluster']
                final_cluster_cols = [col for col in pivot_df.columns if col not in cols_awal]
                final_display_cols = cols_awal + sorted(final_cluster_cols)
                pivot_df = pivot_df[final_display_cols]
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
    
    st.header("✅ Analysis Result")
    st.markdown("---")
    
    # --- PENGGUNAAN AG-GRID DIMULAI DI SINI ---
    
    # 1. Buat GridOptionsBuilder dari DataFrame Anda
    gb = GridOptionsBuilder.from_dataframe(display_df)

    # 2. Konfigurasi "Freeze Panes" (disebut 'pinning' di AG Grid)
    gb.configure_column("VESSEL", pinned="left")
    gb.configure_column("CODE", pinned="left")
    gb.configure_column("VOY_OUT", pinned="left")
    gb.configure_column("ETA", pinned="left", type=["dateColumnFilter","customDateTimeFormat"], custom_format_string='yyyy-MM-dd HH:mm:ss', pivot=True)
    gb.configure_column("Total Box", pinned="left")
    gb.configure_column("Total cluster", pinned="left")
    
    # 3. Konfigurasi default untuk semua kolom lainnya (misal: bisa di-resize)
    gb.configure_default_column(resizable=True, filterable=True, sortable=True, editable=False, minWidth=100)
    
    # 4. Bangun GridOptions
    gridOptions = gb.build()

    # 5. Tampilkan tabel menggunakan AgGrid
    AgGrid(
        display_df,
        gridOptions=gridOptions,
        height=600,
        width='100%',
        # Pilih tema, 'streamlit' adalah tema default yang bersih
        theme='streamlit', 
        # Izinkan konversi tipe data yang tidak aman (diperlukan untuk beberapa fitur)
        allow_unsafe_jscode=True,
        # Fitur tambahan agar pas dengan lebar
        fit_columns_on_grid_load=False
    )
    
    csv_export = display_df.to_csv(index=False).encode('utf-8')
    st.download_button(label="📥 Download Result as CSV", data=csv_export, file_name='analysis_result.csv', mime='text/csv')
