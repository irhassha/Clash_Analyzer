import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide")

st.title("Crane Sequence Timeline")

# Upload file
uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)

    # Pilih tanggal operasi (default ke 2025-05-24)
    selected_date = st.date_input("Pilih tanggal operasi", value=pd.to_datetime("2025-05-24"))

    # Ambil hanya jam dari kolom Queue (formatnya time-only)
    df['Queue_time'] = pd.to_datetime(df['Queue'], errors='coerce').dt.time

    # Gabungkan tanggal terpilih dengan jam dari Queue
    df['start_time'] = df['Queue_time'].apply(lambda t: pd.Timestamp.combine(selected_date, t))

    # Estimasikan durasi tiap crane block (misalnya 30 menit)
    df['end_time'] = df['start_time'] + pd.to_timedelta(30, unit='m')

    # Ubah Main bay jadi numerik untuk sumbu y
    df['Main bay'] = pd.to_numeric(df['Main bay'], errors='coerce')

    # Buat plot Gantt
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

    # Tampilkan data mentah jika diinginkan
    with st.expander("Lihat Data Mentah"):
        st.dataframe(df)
else:
    st.info("Silakan upload file Excel crane sequence.")
