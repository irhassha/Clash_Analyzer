import streamlit as st
import plotly.graph_objects as go

st.set_page_config(layout="wide")

st.title("Bay Header Visualisation")

# Data struktur bay dan sub-bay
main_bay_labels = [10, 10, 14, 14]
sub_bay_labels = [9, 11, 13, 15]

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

fig.update_layout(
    width=600,
    height=200,
    margin=dict(l=20, r=20, t=20, b=20),
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 4]),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 2]),
    plot_bgcolor="white",
    paper_bgcolor="white"
)

st.plotly_chart(fig, use_container_width=False)
