import streamlit as st
from datetime import time
from collections import defaultdict
import pandas as pd
import re

st.set_page_config(layout="wide")

st.markdown("""
<style>
    .timeline {
        display: flex;
        flex-direction: row;
        align-items: flex-start;
        overflow-x: auto;
        gap: 4px;
        padding: 10px;
    }
    .column {
        display: flex;
        flex-direction: column;
        position: relative;
        min-height: 1000px;
        width: 85px;
    }
    .column-title {
        text-align: center;
        font-weight: bold;
        font-size: 12px;
        margin-bottom: 4px;
    }
    .step {
        background-color: #f0f0f0;
        color: #333;
        border-radius: 4px;
        padding: 1px;
        font-size: 6px;
        position: absolute;
        width: 65px;
        left: 10px;
        border: 1px solid #444;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
    }
    .step.loading-step::before {
        content: "\2192 ";
        color: #3498db;
        font-size: 10px;
        margin-right: 2px;
    }
    .step.discharge-step::before {
        content: "\2190 ";
        color: #e74c3c;
        font-size: 10px;
        margin-right: 2px;
    }
    .red { background-color: #e74c3c; color: white; }
    .blue { background-color: #3498db; color: white; }
    .yellow { background-color: #f1c40f; color: white; }
    .green { background-color: #2ecc71; color: white; }
    .step h3 {
        margin: 0 0 2px;
        font-size: 8px;
    }
    .step p {
        margin: 0;
    }
    .time-grid {
        position: sticky;
        left: 0;
        top: 0;
        width: 50px;
        height: 100%;
        background-color: #111;
        z-index: 2;
    }
    .time-label {
        height: 40px;
        font-size: 10px;
        color: #888;
        text-align: right;
        padding-right: 4px;
        box-sizing: border-box;
    }
</style>
""", unsafe_allow_html=True)

st.title("\U0001F4CA Crane Sequence by Bay with Time Axis")

# Upload Excel
uploaded_file = st.sidebar.file_uploader("\U0001F4C2 Upload Excel", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df.columns = df.columns.str.strip().str.replace('.', '', regex=False)
    required_cols = {'Seq', 'Direction', 'Mvs', 'Bay', 'Crane'}
    if not required_cols.issubset(set(df.columns)):
        st.error(f"❌ Excel harus mengandung kolom: {', '.join(required_cols)}")
        st.stop()
    df = df.dropna(subset=['Seq', 'Direction', 'Mvs', 'Bay', 'Crane'])
    df['Bay'] = df['Bay'].astype(str).str.strip()
    data = df.to_dict(orient="records")
else:
    data = []

# Sidebar input waktu mulai per crane
st.sidebar.header("\U0001F552 Set Start Time per Crane")
crane_ids = sorted(set(item["Crane"] for item in data))
start_times = {}
for crane in crane_ids:
    t = st.sidebar.time_input(f"Crane {crane} Start", value=time(1, 0), key=f"cr_{crane}")
    start_times[crane] = t.hour + t.minute / 60

# Color per crane
color_pool = ["blue", "green", "yellow", "red", "orange", "purple"]
crane_colors = {crane: color_pool[i % len(color_pool)] for i, crane in enumerate(crane_ids)}

duration_per_mv = 1 / 30  # 30 moves = 1 hour

timeline = defaultdict(list)
crane_last_time = defaultdict(lambda: 0)

for item in data:
    crane = item['Crane']
    if crane_last_time[crane] == 0:
        crane_last_time[crane] = start_times[crane]
    item['StartTime'] = crane_last_time[crane]
    item['EndTime'] = item['StartTime'] + item['Mvs'] * duration_per_mv
    timeline[item['Bay']].append(item)
    crane_last_time[crane] = item['EndTime']

def sort_key(bay):
    match = re.search(r'\d+', bay)
    return int(match.group()) if match else float('inf')

html = "<div class='timeline'>"

# Time scale background
html += "<div class='time-grid'>"
for h in range(24):
    html += f"<div class='time-label'>{h:02d}:00</div>"
html += "</div>"

if timeline:
    for bay_index, (bay, items) in enumerate(sorted(timeline.items(), key=lambda x: sort_key(x[0]))):
        html += f"<div class='column'><div class='column-title'>Bay {bay}</div>"
        for i, item in enumerate(items):
            color_class = crane_colors.get(item['Crane'], 'red')
            direction_class = 'loading-step' if item['Direction'].lower() == 'loading' else 'discharge-step'
            top_offset = int(item['StartTime'] * 40)
            height = max(6, int((item['EndTime'] - item['StartTime']) * 40))
            html += (
                f"<div class='step {color_class} {direction_class}' style='top:{top_offset}px;height:{height}px;'>"
                f"</div>"
            )
        html += "</div>"
else:
    html += "<p style='color:red;'>⚠️ Tidak ada data valid untuk ditampilkan.</p>"

html += "</div>"

st.markdown(html, unsafe_allow_html=True)
