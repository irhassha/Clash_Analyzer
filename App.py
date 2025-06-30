import streamlit as st
import pandas as pd
import os
from datetime import datetime, timedelta
import json
import io

# Import pustaka yang diperlukan
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode

# --- Konfigurasi Halaman & Judul ---
st.set_page_config(page_title="Clash Analyzer", layout="wide")
st.title("🚨 Yard Clash Monitoring")

# --- Fungsi-fungsi Inti ---
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

def format_bay(bay_val):
    """Membersihkan dan memformat data 'Bay'."""
    if pd.isna(bay_val):
        return None
    s = str(bay_val).replace('..', '-')
    parts = s.split('-')
    cleaned_parts = [str(int(float(p))) for p in parts]
    return '-'.join(cleaned_parts)

# --- STRUKTUR TAB BARU ---
tab1, tab2 = st.tabs(["Clash Monitoring", "Crane Sequence"])

# --- KONTEN TAB 1: CLASH MONITORING ---
with tab1:
    # --- Sidebar & Proses Utama ---
    st.sidebar.header("⚙️ Your File Uploads")
    schedule_file = st.sidebar.file_uploader("1. Upload Vessel Schedule (for Clash Monitoring)", type=['xlsx', 'csv'])
    unit_list_file = st.sidebar.file_uploader("2. Upload Unit List (for both features)", type=['xlsx', 'csv'])
    
    process_button = st.button("🚀 Process Clash Data", type="primary", key="clash_button")

    if 'processed_df' not in st.session_state:
        st.session_state.processed_df = None
    if 'clash_summary_df' not in st.session_state:
        st.session_state.clash_summary_df = None


    df_vessel_codes = load_vessel_codes_from_repo()

    if process_button:
        if schedule_file and unit_list_file and (df_vessel_codes is not None and not df_vessel_codes.empty):
            with st.spinner('Loading and processing data...'):
                try:
                    # (Logika proses untuk Tab 1 tidak berubah)
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
                    pivot_df['TTL BOX'] = pivot_df[cluster_cols_for_calc].sum(axis=1)
                    pivot_df['TTL CLSTR'] = (pivot_df[cluster_cols_for_calc] > 0).sum(axis=1)
                    pivot_df = pivot_df.reset_index()
                    two_days_ago = pd.Timestamp.now() - timedelta(days=2)
                    condition_to_hide = (pivot_df['ETA'] < two_days_ago) & (pivot_df['TTL BOX'] < 50)
                    pivot_df = pivot_df[~condition_to_hide]
                    if pivot_df.empty: st.warning("No data remaining after ETA & Total filter."); st.session_state.processed_df = None; st.stop()
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
            st.warning("Please upload both 'Vessel Schedule' and 'Unit List' files.")

    if st.session_state.processed_df is not None:
        # (Logika tampilan Tab 1 tidak berubah)
        pass


# --- KONTEN TAB 2: CRANE SEQUENCE ---
with tab2:
    st.header("🏗️ Crane Tools")
    
    # Letakkan uploader di dalam tabnya
    crane_file_tab2 = st.file_uploader("Upload Crane Sequence File", type=['xlsx', 'csv'], key="crane_uploader_tab2")

    st.markdown("---")
    
    # --- Fitur 1: Crane Sequence Visualizer ---
    st.subheader("Crane Sequence Visualizer")
    if crane_file_tab2:
        try:
            df_crane_sheet2_viz = pd.read_excel(crane_file_tab2, sheet_name=1)
            df_crane_sheet2_viz.columns = df_crane_sheet2_viz.columns.str.strip()
            
            # Ganti nama kolom dan bersihkan kolom 'Bay'
            df_crane_sheet2_viz.rename(columns={'Main Bay': 'Bay', 'Sequence': 'Seq.', 'QC': 'Crane'}, inplace=True)
            df_crane_sheet2_viz = df_crane_sheet2_viz.dropna(subset=['Bay'])

            df_crane_sheet2_viz['Bay'] = df_crane_sheet2_viz['Bay'].apply(format_bay)
            
            pivot_crane = df_crane_sheet2_viz.pivot(index='Seq.', columns='Bay', values='Crane').fillna('')

            sorted_bays = sorted(pivot_crane.columns, key=lambda x: int(x.split('-')[0]))
            pivot_crane = pivot_crane[sorted_bays]
            
            unique_cranes = df_crane_sheet2_viz['Crane'].unique()
            crane_colors = ['#8dd3c7','#ffffb3','#bebada','#fb8072','#80b1d3','#fdb462','#b3de69','#fccde5','#d9d9d9','#bc80bd']
            color_map = {crane: crane_colors[i % len(crane_colors)] for i, crane in enumerate(unique_cranes)}
            
            def color_crane_cells(val):
                return f'background-color: {color_map[val]}' if val in color_map else ''
            
            st.dataframe(pivot_crane.style.applymap(color_crane_cells), use_container_width=True)

        except Exception as e:
            st.error(f"Failed to process Crane Sequence Visualizer: {e}")
    else:
        st.info("Upload the 'Crane Sequence File' to use this feature.")

    st.markdown("---")

    # --- Fitur 2: Container Area Lookup ---
    st.subheader("Container Area Lookup")
    if crane_file_tab2 and unit_list_file:
        try:
            df_crane_s1 = pd.read_excel(crane_file_tab2, sheet_name=0)
            df_crane_s1.columns = df_crane_s1.columns.str.strip()

            df_crane_s2 = pd.read_excel(crane_file_tab2, sheet_name=1)
            df_crane_s2.columns = df_crane_s2.columns.str.strip()
            df_crane_s2.rename(columns={'Main Bay': 'Bay', 'QC': 'Crane', 'Sequence': 'Seq.'}, inplace=True)


            if unit_list_file.name.lower().endswith(('.xls', '.xlsx')):
                df_unit_list = pd.read_excel(unit_list_file)
            else:
                df_unit_list = pd.read_csv(unit_list_file)
            df_unit_list.columns = df_unit_list.columns.str.strip()
            
            # Pastikan kolom-kolom yang dibutuhkan ada
            required_cols_s1 = ['Container', 'Pos (Vessel)']
            required_cols_s2 = ['Bay', 'Crane', 'Direction', 'Seq.'] 
            required_cols_unit = ['Unit', 'Area (EXE)']
            
            if all(col in df_crane_s1.columns for col in required_cols_s1) and \
               all(col in df_crane_s2.columns for col in required_cols_s2) and \
               all(col in df_unit_list.columns for col in required_cols_unit):
                
                # --- LOGIKA BARU UNTUK MENCARI CRANE & SEQ ---
                
                # 1. Buat peta dari Pos ke Crane dan Seq, HANYA untuk 'Loading'
                pos_to_crane_map = {}
                pos_to_seq_map = {}
                df_crane_s2_loading = df_crane_s2[df_crane_s2['Direction'] == 'Loading'].copy()
                df_crane_s2_cleaned = df_crane_s2_loading.dropna(subset=['Bay', 'Crane', 'Seq.'])
                
                for _, row in df_crane_s2_cleaned.iterrows():
                    bay_range_str = format_bay(row['Bay'])
                    crane = row['Crane']
                    seq = row['Seq.']
                    if bay_range_str:
                        if '-' in bay_range_str:
                            start, end = map(int, bay_range_str.split('-'))
                            for pos in range(start, end + 1):
                                pos_to_crane_map[pos] = crane
                                pos_to_seq_map[pos] = seq
                        else:
                            pos_to_crane_map[int(bay_range_str)] = crane
                            pos_to_seq_map[int(bay_range_str)] = seq
                
                # 2. Proses Sheet1 dan tambahkan kolom-kolom baru
                df_crane_s1['Pos (Vessel)'] = pd.to_numeric(df_crane_s1['Pos (Vessel)'], errors='coerce')
                df_crane_s1.dropna(subset=['Pos (Vessel)'], inplace=True)
                df_crane_s1['Pos (Vessel)'] = df_crane_s1['Pos (Vessel)'].astype(int)
                
                def extract_pos(pos):
                    pos_str = str(pos)
                    return pos_str[0] if len(pos_str) == 5 else pos_str[:2] if len(pos_str) == 6 else ''
                
                df_crane_s1['Pos'] = df_crane_s1['Pos (Vessel)'].apply(extract_pos)
                
                # --- PERBAIKAN FINAL DI SINI ---
                # Gunakan kolom 'Pos' yang sudah diproses untuk mencocokkan
                # Ubah 'Pos' ke numerik untuk lookup yang benar
                df_crane_s1['Crane'] = pd.to_numeric(df_crane_s1['Pos'], errors='coerce').map(pos_to_crane_map).fillna('N/A')
                df_crane_s1['Seq.'] = pd.to_numeric(df_crane_s1['Pos'], errors='coerce').map(pos_to_seq_map).fillna('N/A')
                
                # 3. Gabungkan dengan Unit List untuk mendapatkan Area
                df_crane_s1['Container'] = df_crane_s1['Container'].astype(str).str.strip()
                df_unit_list['Unit'] = df_unit_list['Unit'].astype(str).str.strip()
                
                merged_df = pd.merge(
                    df_crane_s1[['Container', 'Pos', 'Crane', 'Seq.']],
                    df_unit_list[['Unit', 'Area (EXE)']],
                    left_on='Container',
                    right_on='Unit',
                    how='inner'
                )
                
                if not merged_df.empty:
                    # 4. Tampilkan hasil
                    result_df = merged_df[['Container', 'Pos', 'Crane', 'Seq.', 'Area (EXE)']].drop_duplicates()
                    st.write(f"Found area information for {len(result_df)} matching containers.")
                    st.dataframe(result_df, use_container_width=True)
                else:
                    st.info("No matching containers found between the files.")
            else:
                st.warning("Required columns not found. Please check your files.")
        except Exception as e:
            st.error(f"Failed to process Container Area Lookup: {e}")
    else:
        st.info("Upload both 'Crane Sequence File' and 'Unit List' to use this feature.")
