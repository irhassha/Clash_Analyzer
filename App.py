import streamlit as st
from datetime import time

st.set_page_config(layout="wide")

st.markdown("""
<style>
    .step {
        display: inline-block;
        background-color: #f0f0f0;
        color: #333;
        border-radius: 10px;
        padding: 20px;
        margin: 10px;
        width: 220px;
        position: relative;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
    }
    .step:after {
        content: "";
        position: absolute;
        top: 50%;
        right: -20px;
        margin-top: -10px;
        border-left: 20px solid #f0f0f0;
        border-top: 10px solid transparent;
        border-bottom: 10px solid transparent;
    }
    .red { background-color: #e74c3c; color: white; }
    .blue { background-color: #3498db; color: white; }
    .yellow { background-color: #f1c40f; color: white; }
    .green { background-color: #2ecc71; color: white; }
    .step h3 {
        margin: 0;
        font-size: 24px;
    }
    .step p {
        margin: 5px 0 0;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

st.title("📊 Crane Sequence Timeline (Infographic Style)")

# Data dinamis (dummy)
data = [
    {"Seq": 1, "Direction": "Discharge", "Mvs": 45, "Bay": "14", "Crane": 806, "Icon": "👷"},
    {"Seq": 2, "Direction": "Discharge", "Mvs": 10, "Bay": "10", "Crane": 806, "Icon": "📦"},
    {"Seq": 3, "Direction": "Discharge", "Mvs": 15, "Bay": "30", "Crane": 807, "Icon": "⏱️"},
    {"Seq": 4, "Direction": "Discharge", "Mvs": 40, "Bay": "26", "Crane": 807, "Icon": "🏗️"},
]

# Warna rotasi agar variatif
colors = ["red", "blue", "yellow", "green"]

html = "<div style='display:flex; flex-wrap: wrap; align-items: flex-start;'>"

for i, item in enumerate(data):
    color_class = colors[i % len(colors)]
    html += f"""
    <div class='step {color_class}'>
        <h3>{item['Seq']} {item['Icon']}</h3>
        <p><strong>{item['Direction']}</strong></p>
        <p>{item['Mvs']} Moves</p>
        <p>Bay {item['Bay']}<br>Crane {item['Crane']}</p>
    </div>
    """

html += "</div>"

st.markdown(html, unsafe_allow_html=True)

st.sidebar.header("🕒 Set Start Time (Dummy)")
st.sidebar.time_input("Start Time", time(1, 0))
