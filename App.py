import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(layout="wide")

st.title("Crane Sequence Matrix View")

# Dummy data
bay_data = pd.DataFrame([
    {"Bay": "10", "Deck": "WD", "Crane": 806, "Seq.": 2},
    {"Bay": "13..15", "Deck": "WD", "Crane": 806, "Seq.": 1}
])

# Ekstrak semua bay menjadi list integer
all_bays = set()
for bay in bay_data['Bay']:
    if ".." in bay:
        start, end = map(int, bay.split(".."))
        all_bays.update(range(start, end + 1))
    else:
        all_bays.add(int(bay))

unique_bays = sorted(all_bays)
unique_seq = sorted(bay_data['Seq.'].unique())

# Buat matriks kosong
matrix = pd.DataFrame("", index=unique_seq, columns=unique_bays)

# Isi cell dengan info crane di semua bay terkait
for _, row in bay_data.iterrows():
    bay_range = []
    if ".." in row['Bay']:
        start, end = map(int, row['Bay'].split(".."))
        bay_range = list(range(start, end + 1))
    else:
        bay_range = [int(row['Bay'])]

    for b in bay_range:
        matrix.at[row['Seq.'], b] = f"{row['Crane']}\n{row['Deck']}"

# Buat heatmap dummy
z = [[1 if cell != "" else 0 for cell in row] for row in matrix.values]

fig = go.Figure(data=go.Heatmap(
    z=z,
    x=matrix.columns,
    y=matrix.index,
    text=matrix.values,
    texttemplate="%{text}",
    colorscale=[[0, 'white'], [1, 'deepskyblue']],
    showscale=False
))

fig.update_layout(
    title="Crane Sequence by Bay",
    xaxis_title="Bay",
    yaxis_title="Sequence",
    height=500
)

st.plotly_chart(fig, use_container_width=True)

# Tampilkan data mentah
with st.expander("Lihat Data Mentah"):
    st.dataframe(bay_data)
