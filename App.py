import streamlit as st
import plotly.graph_objects as go
from collections import defaultdict

st.set_page_config(layout="wide")

st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
    }
    .css-18e3th9 {
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
## 🧱 Crane Timeline
**Vessel:** EVER BEAMY &nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;**Operation Window:** Based on crane start time
---
""")

# Data struktur bay dan sub-bay
main_bay_labels = [10, 10, 14, 14, 22, 22, 26, 26, 30, 30]
sub_bay_labels = [9, 11, 13, 15, 21, 23, 25, 27, 29, 31]

# Dummy data untuk sequence
sequence_data = [
    {"Bay": "13..15", "Main bay": 14, "Seq": 1, "Direction": "Discharge", "Mvs": 45, "Crane": 806},
    {"Bay": "10", "Main bay": 10, "Seq": 2, "Direction": "Discharge", "Mvs": 10, "Crane": 806},
    {"Bay": "30", "Main bay": 30, "Seq": 1, "Direction": "Discharge", "Mvs": 15, "Crane": 807},
    {"Bay": "26", "Main bay": 26, "Seq": 2, "Direction": "Discharge", "Mvs": 40, "Crane": 807},
    {"Bay": "22", "Main bay": 22, "Seq": 3, "Direction": "Discharge", "Mvs": 3, "Crane": 807},
]

# Ambil daftar crane unik
unique_cranes = sorted(set(seq["Crane"] for seq in sequence_data))

# Sidebar Input Start Time
st.sidebar.header("🕒 Start Time per Crane")
crane_start_times = {}
for crane in unique_cranes:
    start_time_str = st.sidebar.time_input(f"Crane {crane}", value=None, key=f"crane_{crane}")
    if start_time_str is not None:
        crane_start_times[crane] = int(start_time_str.hour) + int(start_time_str.minute) / 60

# Tentukan waktu paling awal dari semua crane
if crane_start_times:
    min_start = min(crane_start_times.values())
else:
    min_start = 0

# Warna per crane
crane_colors = {
    806: "#1f77b4",   # Blue
    807: "#2ca02c"    # Green
}

# Koordinat horizontal bay
bay_x_pos = {10: 1, 14: 3, 22: 5, 26: 7, 30: 9}
main_bay_positions = {10: [0, 2], 14: [2, 4], 22: [4, 6], 26: [6, 8], 30: [8, 10]}

# Inisialisasi plot
fig = go.Figure()

# Buat blok sequence per crane
crane_sequences = defaultdict(list)
for seq in sequence_data:
    crane_sequences[seq['Crane']].append(seq)

max_end_time = min_start
for crane in crane_sequences:
    if crane not in crane_start_times:
        continue
    start_time = crane_start_times[crane]
    sequences = sorted(crane_sequences[crane], key=lambda x: x['Seq'])
    current_time = start_time

    for seq in sequences:
        x_center = bay_x_pos[seq['Main bay']]
        y_base = -current_time
        duration_hours = seq['Mvs'] / 30
        y_top = y_base - duration_hours
        color = crane_colors.get(crane, "gray")

        fig.add_shape(type="rect", x0=x_center - 0.9, x1=x_center + 0.9,
                      y0=y_base, y1=y_top, fillcolor=color,
                      line=dict(color="#111", width=2), opacity=1, layer="above")

        fig.add_annotation(x=x_center, y=(y_base + y_top) / 2 + 0.3,
                           text=f"⬅️ {seq['Direction']}", showarrow=False, font=dict(size=13, color="white"))
        fig.add_annotation(x=x_center, y=(y_base + y_top) / 2 - 0.1,
                           text=f"{seq['Mvs']} moves", showarrow=False, font=dict(size=13, color="white"))
        fig.add_annotation(x=x_center, y=y_top - 0.2,
                           text=f"CR-{crane}", showarrow=False, font=dict(size=11, color="#eee"))

        current_time += duration_hours
        max_end_time = max(max_end_time, current_time)

# Sumbu Y berdasarkan waktu dinamis
start_tick = int(min_start)
end_tick = int(max_end_time) + 1
yticks = [-t for t in range(start_tick, end_tick + 1)]
yticklabels = [f"{t:02d}:00" for t in range(start_tick, end_tick + 1)]
yticklabels.reverse()

# Grid layout sub bay
for i, bay in enumerate(sub_bay_labels):
    fig.add_shape(type="rect", x0=i, x1=i+1, y0=-min_start + 1.5, y1=-min_start + 2.5,
                  line=dict(color="#DDDDDD", width=1), fillcolor="#FBFBFB")
    fig.add_annotation(x=i+0.5, y=-min_start + 2.0, text=str(bay), showarrow=False, font=dict(size=11, color="#555"))

# Grid layout main bay
for bay, (start, end) in main_bay_positions.items():
    fig.add_shape(type="rect", x0=start, x1=end, y0=-min_start + 2.5, y1=-min_start + 3.5,
                  line=dict(color="#BBBBBB", width=1.5), fillcolor="#F5F5F5")
    fig.add_annotation(x=(start+end)/2, y=-min_start + 3.0, text=str(bay), showarrow=False, font=dict(size=13, color="#333"))

fig.update_layout(
    width=1100,
    height=700,
    margin=dict(l=20, r=20, t=30, b=20),
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 10]),
    yaxis=dict(showgrid=True, gridcolor="#f0f0f0", zeroline=False,
               tickvals=yticks, ticktext=yticklabels, range=[-end_tick, -min_start + 4]),
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(family="Open Sans, sans-serif", size=12, color="#333")
)

st.plotly_chart(fig, use_container_width=True)
