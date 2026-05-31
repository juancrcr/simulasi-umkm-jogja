import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from simulasi_umkm_jogja import (
    simulate_umkm_vectorized,
    ringkasan_statistik,
    kurva_lorenz,
    hitung_gini,
    WARNA_SKENARIO,
    get_nama_skenario
)

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="Simulasi UMKM Yogyakarta",
    page_icon="📈",
    layout="wide"
)

# ============================================================
# WARNA SKENARIO
# ============================================================

WARNA_CARD = {
    "A": "#27ae60",
    "B": "#e74c3c",
    "C": "#2980b9",
    "D": "#8e44ad"
}

# ============================================================
# HEADER
# ============================================================

st.title("📈 Simulasi Monte Carlo UMKM Yogyakarta")

st.markdown("""
Model distribusi modal UMKM berbasis **pertukaran acak (Boltzmann-Gibbs)**.

Simulasi menggunakan **Vectorized NumPy Engine**.
""")

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Parameter Simulasi")

n_umkm = st.sidebar.slider(
    "Jumlah UMKM",
    20,
    1000,
    100,
    step=10
)

modal_awal = st.sidebar.number_input(
    "Modal Awal (Rp)",
    value=10_000_000,
    step=1_000_000
)

unit_transaksi = st.sidebar.number_input(
    "Unit Transaksi (Rp)",
    value=100_000,
    step=10_000
)

n_hari = st.sidebar.slider(
    "Hari Transaksi",
    100,
    5000,
    1000,
    step=100
)

skenario = st.sidebar.selectbox(
    "Skenario",
    ["A", "B", "C", "D"],
    format_func=get_nama_skenario
)

p_musibah = 0.005
p_viral = 0.003

if skenario == "D":
    st.sidebar.markdown("### Parameter Kejadian Tak Terduga")

    p_musibah = st.sidebar.slider(
        "Probabilitas Musibah",
        0.001,
        0.05,
        0.005
    )

    p_viral = st.sidebar.slider(
        "Probabilitas Viral",
        0.001,
        0.02,
        0.003
    )

random_seed = st.sidebar.number_input(
    "Random Seed",
    value=42
)

jalankan = st.sidebar.button(
    "🚀 Jalankan Simulasi",
    use_container_width=True
)

# ============================================================
# SIMULASI
# ============================================================

if jalankan:

    with st.spinner("Menjalankan simulasi..."):

        hasil = simulate_umkm_vectorized(
            n_umkm=n_umkm,
            modal_awal=modal_awal,
            unit_transaksi=unit_transaksi,
            n_hari=n_hari,
            skenario=skenario,
            p_musibah=p_musibah,
            p_viral=p_viral,
            random_seed=random_seed,
            simpan_history=True,
            n_snapshot=50
        )

    modal = hasil["modal_akhir"]

    gini = hitung_gini(modal)
    bangkrut = int((modal <= 0).sum())

    warna = WARNA_CARD[skenario]

    # ============================================================
    # METRIC CARDS
    # ============================================================

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Koefisien Gini",
        f"{gini:.3f}"
    )

    c2.metric(
        "UMKM Bangkrut",
        bangkrut
    )

    c3.metric(
        "Modal Tertinggi",
        f"Rp {modal.max()/1e6:.1f} jt"
    )

    c4.metric(
        "Runtime",
        f"{hasil['runtime_detik']:.2f} s"
    )

    st.markdown("---")

    # ============================================================
    # RINGKASAN
    # ============================================================

    st.subheader("📋 Ringkasan Statistik")

    st.dataframe(
        ringkasan_statistik(hasil),
        use_container_width=True
    )

    # ============================================================
    # HISTOGRAM
    # ============================================================

    st.subheader("📊 Distribusi Modal Akhir")

    fig_hist = px.histogram(
        x=modal / 1e6,
        nbins=20,
        color_discrete_sequence=[warna]
    )

    fig_hist.update_layout(
        xaxis_title="Modal (Rp Juta)",
        yaxis_title="Jumlah UMKM",
        template="plotly_white"
    )

    st.plotly_chart(
        fig_hist,
        use_container_width=True
    )

    # ============================================================
    # SORTED DISTRIBUTION
    # ============================================================

    st.subheader("📈 Ranking Modal UMKM")

    modal_sorted = np.sort(modal)

    fig_sorted = px.area(
        x=np.arange(1, len(modal_sorted)+1),
        y=modal_sorted/1e6
    )

    fig_sorted.update_traces(
        line_color=warna
    )

    fig_sorted.update_layout(
        xaxis_title="Ranking UMKM",
        yaxis_title="Modal (Rp Juta)",
        template="plotly_white"
    )

    st.plotly_chart(
        fig_sorted,
        use_container_width=True
    )

    # ============================================================
    # LORENZ
    # ============================================================

    st.subheader("📉 Kurva Lorenz")

    pop_share, income_share = kurva_lorenz(modal)

    fig_lorenz = go.Figure()

    fig_lorenz.add_trace(
        go.Scatter(
            x=[0,1],
            y=[0,1],
            mode="lines",
            line=dict(dash="dash"),
            name="Kesetaraan Sempurna"
        )
    )

    fig_lorenz.add_trace(
        go.Scatter(
            x=pop_share,
            y=income_share,
            mode="lines",
            fill="tozeroy",
            line=dict(color=warna,width=3),
            name=f"Skenario {skenario}"
        )
    )

    fig_lorenz.update_layout(
        template="plotly_white",
        xaxis_title="Proporsi UMKM",
        yaxis_title="Proporsi Modal"
    )

    st.plotly_chart(
        fig_lorenz,
        use_container_width=True
    )

    # ============================================================
    # EVOLUSI STD
    # ============================================================

    st.subheader("📈 Evolusi Standar Deviasi")

    df_std = hasil["std_history"]

    fig_std = px.line(
        df_std,
        x="hari",
        y=df_std["std"]/1e6
    )

    fig_std.update_traces(
        line_color=warna
    )

    fig_std.update_layout(
        template="plotly_white",
        xaxis_title="Hari",
        yaxis_title="Std Dev (Rp Juta)"
    )

    st.plotly_chart(
        fig_std,
        use_container_width=True
    )

    # ============================================================
    # EVOLUSI GINI
    # ============================================================

    st.subheader("📊 Evolusi Koefisien Gini")

    df_gini = hasil["gini_history"]

    fig_gini = px.line(
        df_gini,
        x="hari",
        y="gini"
    )

    fig_gini.update_traces(
        line_color="#e67e22"
    )

    fig_gini.update_layout(
        template="plotly_white",
        yaxis_range=[0,1]
    )

    st.plotly_chart(
        fig_gini,
        use_container_width=True
    )

    # ============================================================
    # INSIGHT PANEL
    # ============================================================

    st.markdown("---")

    st.markdown(
        f"""
        <div style="
            border-left:8px solid {warna};
            padding:15px;
            background:#f8f9fa;
            border-radius:10px;
        ">
        <h4 style="color:{warna};">
        Insight Skenario {skenario}
        </h4>

        <p>
        Koefisien Gini akhir sebesar <b>{gini:.3f}</b>,
        dengan <b>{bangkrut}</b> UMKM mengalami kebangkrutan.
        Simulasi menunjukkan bahwa distribusi modal secara alami
        berkembang menuju ketimpangan meskipun seluruh UMKM
        memulai dari kondisi yang identik.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )
