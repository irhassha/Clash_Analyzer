import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(layout="wide")

st.title("Crane Sequence Matrix View")

# Dummy data
bay_data = pd.DataFrame([
    {"Bay": 10, "Main bay": 10, "Deck": "WD", "Crane": 806, "Seq.": 2},
    {"Bay": "13..15", "Main bay": 14, "Deck": "WD", "Crane": 806, "Seq.": 1}
])

# Dapatkan daftar bay unik untuk kolom dan sequence untuk baris
unique_bays = sorted(bay_data['Main bay'].unique())
unique_seq = sorted(bay_data['Seq.'].unique())

# Buat grid kosong
matrix = pd.DataFrame("", index=unique_seq, columns=unique_bays)

# Isi cell dengan informasi Crane
for _, row in bay_data.iterrows():
    matrix.at[row['Seq.'], row['Main bay']] = f"{row['Crane']}\n{row['Deck']}"

# Tampilkan sebagai heatmap text
fig = go.Figure(data=go.Heatmap(
    z=[[0]*len(matrix.columns)]*len(matrix.index),
    x=matrix.columns,
    y=matrix.index,
    text=matrix.values,
    texttemplate="%{text}",
    colorscale=[[0, 'lightblue'], [1, 'lightblue']],
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
