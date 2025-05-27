import streamlit as st
import plotly.graph_objects as go

st.set_page_config(layout="wide")

st.title("Bay Header Visualisation + Sequence")

# Data struktur bay dan sub-bay
main_bay_labels = [10, 10, 14, 14]
sub_bay_labels = [9, 11, 13, 15]

# Dummy data untuk sequence
sequence_data = [
    {"Bay": "13..15", "Main bay": 14, "Seq": 1},
    {"Bay": "10", "Main bay": 10, "Seq": 2},
]

# Buat figure
fig = go.Figure()

# Tambahkan kotak sub-bay
for i, bay in enumerate(sub_bay_labels):
    fig.add_shape(type="rect",
                  x0=i, x1=i+1, y0=0, y1=1,
                  line=dict(color="black"), fillcolor="white")
    fig.add_annotation(x=i+0.5, y=0.5, text=str(bay), showarrow=False, font=dict(size=14))

# Tambahkan kotak main bay
main_bay_positions = {10: [0, 2], 14: [2, 4]}
for bay, (start, end) in main_bay_positions.items():
    fig.add_shape(type="rect",
                  x0=start, x1=end, y0=1, y1=2,
                  line=dict(color="black"), fillcolor="white")
    fig.add_annotation(x=(start+end)/2, y=1.5, text=str(bay), showarrow=False, font=dict(size=14))

# Tambahkan blok sequence di bawah
sequence_start_y = -1
sequence_height = 0.8

# Tentukan posisi X berdasarkan main bay
bay_x_pos = {10: 1, 14: 3}  # tengah dari kolom sub-bay

for seq in sequence_data:
    x_center = bay_x_pos[seq['Main bay']]
    y_base = sequence_start_y - (seq['Seq'] - 1) * (sequence_height + 0.2)
    fig.add_shape(type="rect",
                  x0=x_center - 0.9, x1=x_center + 0.9,
                  y0=y_base, y1=y_base + sequence_height,
                  fillcolor="deepskyblue", line=dict(color="black"))
    fig.add_annotation(x=x_center, y=y_base + sequence_height / 2,
                       text=f"Seq {seq['Seq']}", showarrow=False, font=dict(size=12, color="black"))

fig.update_layout(
    width=600,
    height=400,
    margin=dict(l=20, r=20, t=20, b=20),
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 4]),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-3, 2]),
    plot_bgcolor="white",
    paper_bgcolor="white"
)

st.plotly_chart(fig, use_container_width=False)
