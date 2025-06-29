import streamlit as st
import pandas as pd
import os
from datetime import datetime, timedelta

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

# --- FUNGSI STYLING DENGAN PERBAIKAN INDEX ---
def apply_all_styles(df, selected_dates):
    styler = pd.DataFrame('', index=df.index, columns=df.columns)
    
    # PERBAIKAN: Buat salinan dan langsung reset index di awal
    df_copy = df.copy().reset_index(drop=True)
    styler.index = df_copy.index

    df_copy['ETA'] = pd.to_datetime(df_copy['ETA'])
    df_copy['ETA_Date'] = df_copy['ETA'].dt.date

    is_faded_row = pd.Series(False, index=df_copy.index)
    if selected_dates and len(selected_dates) < len(df_copy['ETA_Date'].unique()):
        is_faded_row[~df_copy['ETA_Date'].isin(selected_dates)] = True
        
    clash_mask = pd.DataFrame(False, index=df_copy.index, columns=df.columns)
    non_cluster_cols = ['VESSEL', 'CODE', 'VOY_OUT', 'ETA', 'Total Box', 'Total cluster']
    cluster_cols = [col for col in df_copy.columns if col not in non_cluster_cols]
    
    for col in cluster_cols:
        numeric_col = pd.to_numeric(df_copy[col], errors='coerce').fillna(0)
        subset_df = df_copy[numeric_col > 0]
        clash_dates = subset_df[subset_df.duplicated(subset='ETA_Date', keep=False)]['ETA_Date'].unique()
        for date in clash_dates:
            clashing_indices = subset_df[subset_df['ETA_Date'] == date].index
            clash_mask.loc[clashing_indices, col] = True
            
    for r_idx in df_copy.index:
        for c_name in df_copy.columns:
            is_faded = is_faded_row[r_idx]
            is_clash = clash_mask.loc[r_idx, c_name] if c_name in clash_mask.columns else False
            if is_clash and is_faded:
                styler.loc[r_idx, c_name] = 'background-color: #FFE8D6; color: #BDBDBD;'
            elif is_clash and not is_faded:
                styler.loc[r_idx, c_name] = 'background-color: #FFAA33; color: black;'
            elif not is_clash and is_faded:
                styler.loc[r_idx, c_name] = 'color: #E0E0E0;'
    return styler

def general_formatter(val):
    if isinstance(val, (int, float)):
        if val == 0: return ''
        return f'{val:.0f}'
    return val

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

# --- Display Area ---
if st.session_state.processed_df is not None:
    display_df = st.session_state.processed_df
    st.header("✅ Analysis Result")
    col1, col2 = st.columns([1, 2])
    with col1:
        display_df_copy = display_df.copy()
        display_df_copy['ETA_Date_Only'] = pd.to_datetime(display_df_copy['ETA']).dt.date
        unique_dates_in_data = sorted(display_df_copy['ETA_Date_Only'].unique())
        selected_dates = st.multiselect(
            "**Focus on Date(s):**",
            options=unique_dates_in_data,
            format_func=lambda date: date.strftime('%Y-%m-%d')
        )
    st.markdown("---")
    df_to_style = display_df.copy()
    header_style = {'selector': 'th', 'props': [('font-weight', 'bold')]}
    numeric_cols = [col for col in df_to_style.columns if df_to_style[col].dtype in ['int64', 'float64']]
    formatter = {col: lambda x: '' if x == 0 else f'{x:.0f}' for col in numeric_cols}
    formatter['ETA'] = '{:%Y-%m-%d %H:%M:%S}'
    styled_df = (
        df_to_style.style
        .apply(apply_all_styles, axis=None, selected_dates=selected_dates)
        .format(formatter)
        .set_table_styles([header_style])
    )
    st.dataframe(styled_df, use_container_width=True)
    csv_export = display_df.to_csv(index=False).encode('utf-8')
    st.download_button(label="📥 Download Result as CSV", data=csv_export, file_name='analysis_result.csv', mime='text/csv')
