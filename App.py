import streamlit as st
import plotly.graph_objects as go

st.set_page_config(layout="wide")

st.title("Bay Header Visualisation + Sequence")

# Data struktur bay dan sub-bay
main_bay_labels = [10, 10, 14, 14, 22, 22, 26, 26, 30, 30]
sub_bay_labels = [9, 11, 13, 15, 21, 23, 25, 27, 29, 31]

# Dummy data untuk sequence
sequence_data = [
    {"Bay": "13..15", "Main bay": 14, "Seq": 1, "Direction": "Discharge", "Mvs": 45, "Queue": "806:01:00", "Crane": 806},
    {"Bay": "10", "Main bay": 10, "Seq": 2, "Direction": "Discharge", "Mvs": 10, "Queue": "806:02:00", "Crane": 806},
    {"Bay": "30", "Main bay": 30, "Seq": 1, "Direction": "Discharge", "Mvs": 15, "Queue": "807:01:00", "Crane": 807},
    {"Bay": "26", "Main bay": 26, "Seq": 2, "Direction": "Discharge", "Mvs": 40, "Queue": "807:02:00", "Crane": 807},
    {"Bay": "22", "Main bay": 22, "Seq": 3, "Direction": "Discharge", "Mvs": 3, "Queue": "807:03:00", "Crane": 807},
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
main_bay_positions = {10: [0, 2], 14: [2, 4], 22: [4, 6], 26: [6, 8], 30: [8, 10]}
for bay, (start, end) in main_bay_positions.items():
    fig.add_shape(type="rect",
                  x0=start, x1=end, y0=1, y1=2,
                  line=dict(color="black"), fillcolor="white")
    fig.add_annotation(x=(start+end)/2, y=1.5, text=str(bay), showarrow=False, font=dict(size=14))

# Warna per crane
crane_colors = {
    806: "deepskyblue",
    807: "lightgreen"
}

# Tambahkan blok sequence di bawah
sequence_height = 0.8
sequence_padding = 0.2

# Tentukan posisi X berdasarkan main bay
bay_x_pos = {
    10: 1,
    14: 3,
    22: 5,
    26: 7,
    30: 9
}  # tengah dari kolom sub-bay

# Ambil jam dari queue dan buat sumbu Y berdasarkan waktu
for seq in sequence_data:
    x_center = bay_x_pos[seq['Main bay']]
    time_str = seq['Queue'][-5:]  # ambil 5 karakter terakhir, contoh "01:00"
    time_float = int(time_str[:2]) + int(time_str[3:]) / 60  # contoh "01:30" jadi 1.5
    y_base = -time_float

    color = crane_colors.get(seq['Crane'], "gray")

    fig.add_shape(type="rect",
                  x0=x_center - 0.9, x1=x_center + 0.9,
                  y0=y_base, y1=y_base + sequence_height,
                  fillcolor=color, line=dict(color="black"))
    fig.add_annotation(x=x_center, y=y_base + sequence_height * 0.65,
                       text=seq['Direction'], showarrow=False, font=dict(size=12, color="black"))
    fig.add_annotation(x=x_center, y=y_base + sequence_height * 0.35,
                       text=f"{seq['Mvs']} mv", showarrow=False, font=dict(size=12, color="black"))

# Tambahkan label waktu di sumbu Y
yticks = [-3, -2.5, -2, -1.5, -1, -0.5, 0]
yticklabels = ["03:00", "02:30", "02:00", "01:30", "01:00", "00:30", "00:00"]

fig.update_layout(
    width=1000,
    height=600,
    margin=dict(l=20, r=20, t=20, b=20),
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 10]),
    yaxis=dict(showgrid=False, zeroline=False, tickvals=yticks, ticktext=yticklabels, range=[-4, 2]),
    plot_bgcolor="white",
    paper_bgcolor="white"
)

st.plotly_chart(fig, use_container_width=False)
