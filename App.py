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

# (Semua fungsi dari awal sampai load_vessel_codes_from_repo tetap sama)
# ...

def render_recommendation_tab():
    """Function to display the stacking recommendation simulation."""
    st.header("💡 Stacking Recommendation Simulation")

    if 'vessel_area_slots' not in st.session_state or st.session_state.vessel_area_slots is None:
        st.warning("Please process data on the 'Clash Analysis' tab first.")
        return

    # --- Tombol untuk menjalankan simulasi ---
    run_simulation = st.button("🚀 Run Stacking Recommendation", type="primary")

    if run_simulation:
        with st.spinner("Running simulation... This might take a moment."):
            try:
                # --- FASE 0: DATA LOADING ---
                vessel_area_slots_df = st.session_state.vessel_area_slots.copy()
                forecast_df = st.session_state.get('forecast_df')
                
                # Load Stacking Trend File
                if not os.path.exists("stacking_trend.xlsx"):
                    st.error("File 'stacking_trend.xlsx' not found in the repository.")
                    st.stop()
                trend_df = pd.read_excel("stacking_trend.xlsx").set_index('STACKING TREND')

                # --- FASE 1: INITIAL YARD STATE ---
                # Key: 'AREA-SLOT', Value: 'VESSEL/VOY'
                yard_occupancy = {}
                for _, row in vessel_area_slots_df.iterrows():
                    for slot in range(row['MIN_SLOT'], row['MAX_SLOT'] + 1):
                        key = f"{row['Area (EXE)']}-{slot}"
                        yard_occupancy[key] = f"{row['VESSEL']}/{row['VOY_OUT']}"
                
                # --- FASE 2: GENERATE DAILY REQUIREMENTS ---
                recommendations = []
                failed_allocations = []
                
                # Gunakan data dari processed_df untuk info kapal
                planning_df = st.session_state.processed_df.copy()
                planning_df = pd.merge(planning_df, forecast_df[['Service', 'Loading Forecast']], on='SERVICE', how='left')
                planning_df['Loading Forecast'].fillna(planning_df['TOTAL BOX'], inplace=True) # Fallback to total box if no forecast
                
                # Hitung CLUSTER REQ
                planning_df['CLSTR REQ'] = planning_df['Loading Forecast'].apply(lambda v: 4 if v <= 450 else (5 if v <= 600 else (6 if v <= 800 else 8)))

                # Simulasi dari hari ini sampai ETD terakhir
                sim_start_date = pd.to_datetime(datetime.now().date())
                sim_end_date = planning_df['ETD'].max().normalize()

                for current_date in pd.date_range(start=sim_start_date, end=sim_end_date):
                    # Cari kapal yang sedang dalam periode stacking
                    daily_tasks = planning_df[
                        (planning_df['ETA'] < current_date + timedelta(days=1)) & # ETA sudah lewat atau hari ini
                        (planning_df['ETD'] > current_date) # Belum berangkat
                    ].sort_values(by='ETA')

                    for _, vessel in daily_tasks.iterrows():
                        # Hitung berapa box yang harus di-stack hari ini
                        # (Logika ini perlu disempurnakan berdasarkan kolom 'Open Stacking' jika ada)
                        # Untuk sekarang, kita asumsikan pembagian merata sederhana
                        
                        # --- (Logika Stacking Trend akan menjadi sangat kompleks, kita mulai dengan alokasi sederhana dulu) ---
                        # Untuk contoh ini, kita akan coba alokasikan TOTAL BOX yang belum teralokasi
                        
                        # Cek apakah kapal ini sudah dialokasikan
                        is_allocated = any(rec['Vessel'] == vessel['VESSEL'] for rec in recommendations)
                        if is_allocated:
                            continue

                        # --- FASE 3: ALGORITHM ---
                        # (Ini adalah versi sederhana dari algoritma yang kita diskusikan)
                        
                        # TODO: Implementasi algoritma alokasi yang kompleks di sini
                        # 1. Cek blok yang sudah dipakai
                        # 2. Jika tidak ada, cari blok baru
                        # 3. Patuhi semua rules (kapasitas, jarak, CLUSTER REQ, dll)
                        
                        # Untuk sekarang, kita tampilkan pesan placeholder
                        recommendations.append({
                            "Vessel": vessel['VESSEL'],
                            "Service": vessel['SERVICE'],
                            "Total Boxes": vessel['TOTAL BOX'],
                            "Recommendation": "Algorithm logic to be implemented."
                        })


                # --- FASE 4: OUTPUT ---
                st.subheader("Simulation Results")
                if recommendations:
                    st.dataframe(pd.DataFrame(recommendations), use_container_width=True)
                else:
                    st.info("No new allocations were recommended in the simulation period.")
                
                if failed_allocations:
                    st.subheader("Failed Allocations (ACTION NEEDED)")
                    st.dataframe(pd.DataFrame(failed_allocations), use_container_width=True)

            except Exception as e:
                st.error(f"An error occurred during the simulation: {e}")

# --- MAIN RENDER FUNCTIONS ---
# (render_clash_tab dan render_forecast_tab tetap sama persis seperti di skrip final sebelumnya)

# --- MAIN STRUCTURE WITH TABS ---
tab1, tab2, tab3 = st.tabs(["🚨 Clash Analysis", "📈 Loading Forecast", "💡 Stacking Recommendation"])
with tab1:
    render_clash_tab()
with tab2:
    render_forecast_tab()
with tab3:
    render_recommendation_tab()
