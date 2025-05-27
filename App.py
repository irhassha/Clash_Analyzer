import streamlit as st
from datetime import time
from collections import defaultdict
import pandas as pd

st.set_page_config(layout="wide")

st.markdown("""
<style>
    .column {
        display: inline-block;
        vertical-align: top;
        margin: 10px;
    }
    .column-title {
        text-align: center;
        font-weight: bold;
        font-size: 20px;
        margin-bottom: 10px;
    }
    .step {
        background-color: #f0f0f0;
        color: #333;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
        width: 220px;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
    }
    .red { background-color: #e74c3c; color: white; }
    .blue { background-color: #3498db; color: white; }
    .yellow { background-color: #f1c40f; color: white; }
    .green { background-color: #2ecc71; color: white; }
    .step h3 {
        margin: 0;
        font-size: 22px;
    }
    .step p {
        margin: 5px 0;
        font-size: 14px;
    }
    .time-labels {
        position: absolute;
        left: 0;
        width: 80px;
        text-align: right;
        padding-right: 10px;
        font-weight: bold;
        color: #555;
    }
</style>
""", unsafe_allow_html=True)

st.title("📊 Crane Sequence by Bay with Time Axis")

# Upload Excel
uploaded_file = st.sidebar.file_uploader("📂 Upload Excel", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
df.columns = df.columns.str.strip().str.replace('.', '', regex=False)
    data = df.to_dict(orient="records")
else:
    data = [
        {"Seq": 1, "Direction": "Discharge", "Mvs": 45, "Bay": "14", "Crane": 806, "Icon": "👷"},
        {"Seq": 2, "Direction": "Discharge", "Mvs": 10, "Bay": "10", "Crane": 806, "Icon": "📦"},
        {"Seq": 3, "Direction": "Discharge", "Mvs": 15, "Bay": "30", "Crane": 807, "Icon": "⏱️"},
        {"Seq": 4, "Direction": "Discharge", "Mvs": 40, "Bay": "26", "Crane": 807, "Icon": "🏗️"},
    ]

# Sidebar input waktu mulai per crane
st.sidebar.header("🕒 Set Start Time per Crane")
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

html = """
<div style='display: flex;'>
    <div style='display: flex; flex-direction: column; margin-right: 30px;'>
"""

for t in range(6, 0, -1):
    html += f"<div style='height:40px;' class='time-labels'>{t:02d}:00</div>"

html += "</div>"

for bay_index, (bay, items) in enumerate(sorted(timeline.items())):
    html += f"<div class='column'><div class='column-title'>Bay {bay}</div>"
    for i, item in enumerate(items):
        color_class = crane_colors.get(item['Crane'], 'red')
        top_offset = int((item['StartTime'] - 1) * 40)  # 40px per hour from 01:00
        height = int((item['EndTime'] - item['StartTime']) * 40)
        html += f"""
        <div class='step {color_class}' style='margin-top:{top_offset}px;height:{height}px;'>
            <h3>{item['Seq']} {item.get('Icon', '')}</h3>
            <p><strong>{item['Direction']}</strong></p>
            <p>{item['Mvs']} Moves</p>
            <p>Crane {item['Crane']}</p>
        </div>
        """
    html += "</div>"
html += "</div>"

st.markdown(html, unsafe_allow_html=True)
