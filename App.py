import streamlit as st
from datetime import time
from collections import defaultdict

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
</style>
""", unsafe_allow_html=True)

st.title("📊 Crane Sequence by Bay (Vertical Layout)")

# Data dinamis
data = [
    {"Seq": 1, "Direction": "Discharge", "Mvs": 45, "Bay": "14", "Crane": 806, "Icon": "👷"},
    {"Seq": 2, "Direction": "Discharge", "Mvs": 10, "Bay": "10", "Crane": 806, "Icon": "📦"},
    {"Seq": 3, "Direction": "Discharge", "Mvs": 15, "Bay": "30", "Crane": 807, "Icon": "⏱️"},
    {"Seq": 4, "Direction": "Discharge", "Mvs": 40, "Bay": "26", "Crane": 807, "Icon": "🏗️"},
]

# Warna rotasi
colors = ["red", "blue", "yellow", "green"]

# Kelompokkan berdasarkan Bay
grouped = defaultdict(list)
for item in data:
    grouped[item['Bay']].append(item)

# Urutkan berdasarkan Sequence
for bay in grouped:
    grouped[bay] = sorted(grouped[bay], key=lambda x: x['Seq'])

# Bangun HTML kolom per bay
html = "<div style='display: flex;'>"
for bay_index, (bay, items) in enumerate(sorted(grouped.items())):
    html += f"<div class='column'><div class='column-title'>Bay {bay}</div>"
    for i, item in enumerate(items):
        color_class = colors[(bay_index + i) % len(colors)]
        html += f"""
        <div class='step {color_class}'>
            <h3>{item['Seq']} {item['Icon']}</h3>
            <p><strong>{item['Direction']}</strong></p>
            <p>{item['Mvs']} Moves</p>
            <p>Crane {item['Crane']}</p>
        </div>
        """
    html += "</div>"
html += "</div>"

st.markdown(html, unsafe_allow_html=True)

st.sidebar.header("🕒 Set Start Time (Dummy)")
st.sidebar.time_input("Start Time", time(1, 0))
