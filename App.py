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

# Langsung tampilkan blok HTML statis (seperti yang diminta)
st.markdown("""
<div style='display:flex; align-items: center;'>
    <div class='step blue'>
        <h3>2 📦</h3>
        <p><strong>Discharge</strong></p>
        <p>10 Moves</p>
        <p>Bay 10<br>Crane 806</p>
    </div>

    <div class='step yellow'>
        <h3>3 ⏱️</h3>
        <p><strong>Discharge</strong></p>
        <p>15 Moves</p>
        <p>Bay 30<br>Crane 807</p>
    </div>

    <div class='step green'>
        <h3>4 🏗️</h3>
        <p><strong>Discharge</strong></p>
        <p>40 Moves</p>
        <p>Bay 26<br>Crane 807</p>
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.header("🕒 Set Start Time (Dummy)")
st.sidebar.time_input("Start Time", time(1, 0))
