
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide")

# Load data
uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)

    # Convert Queue to datetime
    df['start_time'] = pd.to_datetime(df['Queue'], errors='coerce')

    # Assume a default 30-minute duration per sequence
    df['end_time'] = df['start_time'] + pd.to_timedelta(30, unit='m')

    # Convert Bay to string and Main bay to numeric
    df['Main bay'] = pd.to_numeric(df['Main bay'], errors='coerce')

    # Visualize
    st.subheader("Crane Sequence Gantt Chart")
    fig = px.timeline(
        df,
        x_start='start_time',
        x_end='end_time',
        y='Main bay',
        color='Crane',
        text='Mvs',
        title="Crane Sequence Timeline"
    )
    fig.update_yaxes(title="Bay Position", autorange="reversed")
    fig.update_layout(xaxis_title="Time", height=800)
    st.plotly_chart(fig, use_container_width=True)

    # Optional: Display raw data
    with st.expander("Lihat Data Mentah"):
        st.dataframe(df)
