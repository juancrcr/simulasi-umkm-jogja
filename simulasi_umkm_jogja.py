"""
================================================================
SIMULASI STOKASTIK DINAMIKA PASAR & KETIMPANGAN MODAL UMKM
DI YOGYAKARTA
================================================================
Referensi: Adaptasi dari "A Wealth Distribution Game" (Zhihu)
           dan "How to Become Rich? - A Simulation of Social
           Wealth Distribution" (Zhihu / CSDN)
           
Transformasi konteks:
  Agen        → UMKM (Usaha Mikro, Kecil, Menengah)
  Uang        → Modal Usaha (Rp)
  Iterasi     → Hari Transaksi

Upgrade teknis:
  - Nested loop → Vectorized NumPy (O(N) per iterasi)
  - GPU-ready: ganti `import numpy as np` dengan `import cupy as cp`
  - Support hingga 100.000 UMKM
  
Skenario:
  A: Pasar Konvensional (tanpa pinjaman)
  B: Dengan Pinjaman / KUR
  C: UMKM Inovatif (+1% keunggulan kompetitif)
  D: Dengan Kejadian Tak Terduga (musibah & viral medsos)
================================================================
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
from typing import Optional
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# 1. CORE SIMULATION ENGINE — VECTORIZED NUMPY
# ============================================================

def simulate_umkm_vectorized(
    n_umkm: int = 100,
    modal_awal: float = 10_000_000,
    unit_transaksi: float = 100_000,
    n_hari: int = 3650,
    skenario: str = 'A',
    p_musibah: float = 0.005,
    p_viral: float = 0.003,
    boost_inovatif: float = 0.011,
    random_seed: int = 42,
    simpan_history: bool = True,
    n_snapshot: int = 50
) -> dict:
    """
    Simulasi Monte Carlo distribusi modal UMKM menggunakan operasi
    vektor NumPy. GPU-ready: ganti np dengan cupy untuk akselerasi CUDA.
    
    Parameters
    ----------
    n_umkm         : Jumlah UMKM dalam simulasi
    modal_awal     : Modal awal setiap UMKM (Rupiah)
    unit_transaksi : Nilai transaksi harian (Rupiah)
    n_hari         : Jumlah hari transaksi
    skenario       : 'A' (konvensional), 'B' (pinjaman),
                     'C' (inovatif), 'D' (tak terduga)
    p_musibah      : Probabilitas musibah per UMKM per hari (Skenario D)
    p_viral        : Probabilitas efek viral per UMKM per hari (Skenario D)
    boost_inovatif : Probabilitas penerimaan UMKM inovatif (Skenario C)
    random_seed    : Seed untuk reprodusibilitas
    simpan_history : Apakah menyimpan riwayat distribusi (memory-intensive)
    n_snapshot     : Jumlah snapshot distribusi yang disimpan
    
    Returns
    -------
    dict berisi: modal_akhir, history (opsional), std_history,
                 gini_history, runtime_detik
    """
    rng = np.random.default_rng(random_seed)
    start_time = time.time()
    
    # Inisialisasi modal semua UMKM
    modal = np.full(n_umkm, modal_awal, dtype=np.float64)
    
    # Setup probabilitas skenario C (10% UMKM inovatif)
    if skenario == 'C':
        prob = np.full(n_umkm, (1.0 - boost_inovatif * (n_umkm // 10)) / (n_umkm - n_umkm // 10))
        idx_inovatif = np.arange(0, n_umkm, 10)  # ID UMKM: 0, 10, 20, ..., 90, ...
        prob[idx_inovatif] = boost_inovatif
        prob = prob / prob.sum()  # Normalisasi total = 1.0
    else:
        prob = None  # Uniform — lebih cepat dengan randint
    
    # Tentukan hari snapshot
    hari_snapshot = np.unique(np.linspace(0, n_hari, n_snapshot, dtype=int))
    
    # Storage
    history = {} if simpan_history else None
    std_history = []
    gini_history = []
    
    if simpan_history:
        history[0] = modal.copy()
    
    print(f"\n{'='*50}")
    print(f"Skenario {skenario}: {get_nama_skenario(skenario)}")
    print(f"UMKM: {n_umkm:,} | Hari Transaksi: {n_hari:,} | Modal Awal: Rp{modal_awal/1e6:.1f}jt")
    print(f"{'='*50}")
    
    for t in range(1, n_hari + 1):
        
        # ─── TRANSFER TRANSAKSI (Vectorized) ─────────────────
        if skenario == 'A':
            # Hanya UMKM dengan modal > 0 yang membayar
            mask_aktif = modal > 0
            n_aktif = mask_aktif.sum()
            
            if n_aktif > 0:
                # Setiap UMKM aktif memilih penerima secara acak
                penerima = rng.integers(0, n_umkm, size=n_aktif)
                # Hitung total yang diterima setiap UMKM (vectorized bincount)
                transfer_masuk = np.bincount(penerima, minlength=n_umkm).astype(np.float64) * unit_transaksi
                modal[mask_aktif] -= unit_transaksi
                modal += transfer_masuk
        else:
            # Skenario B, C, D: semua UMKM bertransaksi
            if skenario == 'C':
                penerima = rng.choice(n_umkm, size=n_umkm, p=prob)
            else:
                penerima = rng.integers(0, n_umkm, size=n_umkm)
            
            transfer_masuk = np.bincount(penerima, minlength=n_umkm).astype(np.float64) * unit_transaksi
            modal -= unit_transaksi
            modal += transfer_masuk
        
        # ─── KEJADIAN TAK TERDUGA (Skenario D) ───────────────
        if skenario == 'D':
            # Musibah: kurangi modal 30-50% secara acak
            mask_musibah = rng.random(n_umkm) < p_musibah
            if mask_musibah.any():
                pemotongan = rng.uniform(0.3, 0.5, size=n_umkm)
                modal[mask_musibah] *= (1.0 - pemotongan[mask_musibah])
            
            # Viral: gandakan modal
            mask_viral = rng.random(n_umkm) < p_viral
            modal[mask_viral] *= 2.0
        
        # ─── REKAM STATISTIK ─────────────────────────────────
        if t % 10 == 0 or t == n_hari:
            std_history.append({'hari': t, 'std': np.std(modal)})
            gini_history.append({'hari': t, 'gini': hitung_gini(modal)})
        
        if simpan_history and t in hari_snapshot:
            history[t] = modal.copy()
        
        # Progress (hanya untuk simulasi besar)
        if n_hari >= 5000 and t % (n_hari // 10) == 0:
            print(f"  Progress: {t/n_hari*100:.0f}% | Std Dev: Rp{np.std(modal)/1e6:.2f}jt | Gini: {hitung_gini(modal):.3f}")
    
    runtime = time.time() - start_time
    
    print(f"\n✓ Selesai dalam {runtime:.2f} detik")
    print(f"  Modal Tertinggi : Rp {modal.max()/1e6:.2f} juta")
    print(f"  Modal Terendah  : Rp {modal.min()/1e6:.2f} juta")
    print(f"  Koefisien Gini  : {hitung_gini(modal):.4f}")
    print(f"  UMKM Bangkrut   : {(modal <= 0).sum()} ({(modal <= 0).mean()*100:.1f}%)")
    
    return {
        'modal_akhir': modal,
        'history': history,
        'std_history': pd.DataFrame(std_history),
        'gini_history': pd.DataFrame(gini_history),
        'runtime_detik': runtime,
        'skenario': skenario,
        'params': {
            'n_umkm': n_umkm, 'modal_awal': modal_awal,
            'n_hari': n_hari, 'p_musibah': p_musibah, 'p_viral': p_viral
        }
    }


def get_nama_skenario(sku: str) -> str:
    return {
        'A': 'Pasar Konvensional (Tanpa Pinjaman)',
        'B': 'Dengan Pinjaman / KUR',
        'C': 'UMKM Inovatif (+1% Keunggulan Kompetitif)',
        'D': 'Dengan Kejadian Tak Terduga (Musibah & Viral)',
    }.get(sku, sku)


# ============================================================
# 2. FUNGSI STATISTIK
# ============================================================

def hitung_gini(modal: np.ndarray) -> float:
    """
    Menghitung Koefisien Gini dari array distribusi modal.
    Koefisien Gini = 0: merata sempurna, = 1: satu entitas kuasai semua.
    
    Formula: G = (2 * sum((i+1) * x_i) - (n+1) * sum(x_i)) / (n * sum(x_i))
    di mana x_i adalah nilai modal yang diurutkan secara ascending.
    """
    modal_positif = modal[modal > 0]
    if len(modal_positif) < 2:
        return 1.0 if len(modal_positif) == 0 else 0.0
    
    modal_sorted = np.sort(modal_positif)
    n = len(modal_sorted)
    cumsum = modal_sorted.cumsum()
    
    gini = (2 * np.arange(1, n+1).dot(modal_sorted) - (n + 1) * cumsum[-1]) / (n * cumsum[-1])
    return float(np.clip(gini, 0, 1))


def kurva_lorenz(modal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Menghitung titik-titik Kurva Lorenz.
    
    Returns
    -------
    (pop_share, income_share): proporsi kumulatif populasi dan pendapatan
    """
    modal_sorted = np.sort(modal[modal >= 0])
    n = len(modal_sorted)
    if n == 0:
        return np.array([0, 1]), np.array([0, 1])
    
    cumulative_income = np.cumsum(modal_sorted) / modal_sorted.sum()
    cumulative_pop = np.arange(1, n + 1) / n
    
    pop_share = np.concatenate([[0], cumulative_pop])
    income_share = np.concatenate([[0], cumulative_income])
    
    return pop_share, income_share


def ringkasan_statistik(hasil: dict) -> pd.DataFrame:
    """Membuat ringkasan statistik distribusi modal akhir."""
    modal = hasil['modal_akhir']
    params = hasil['params']
    
    return pd.DataFrame({
        'Metrik': [
            'Jumlah UMKM',
            'Modal Awal (Rp)',
            'Hari Transaksi',
            'Modal Rata-rata (Rp)',
            'Modal Median (Rp)',
            'Std Deviasi (Rp)',
            'Modal Tertinggi (Rp)',
            'Modal Terendah (Rp)',
            'Koefisien Gini',
            'UMKM Bangkrut (n)',
            'UMKM Bangkrut (%)',
            'UMKM Modal < Awal (%)',
            '10% Terkaya Kuasai (%)',
            '20% Terkaya Kuasai (%)',
        ],
        'Nilai': [
            f"{params['n_umkm']:,}",
            f"Rp {params['modal_awal']/1e6:.0f} juta",
            f"{params['n_hari']:,} hari",
            f"Rp {modal.mean()/1e6:.2f} juta",
            f"Rp {np.median(modal)/1e6:.2f} juta",
            f"Rp {modal.std()/1e6:.2f} juta",
            f"Rp {modal.max()/1e6:.2f} juta",
            f"Rp {modal.min()/1e6:.2f} juta",
            f"{hitung_gini(modal):.4f}",
            f"{(modal <= 0).sum()}",
            f"{(modal <= 0).mean()*100:.1f}%",
            f"{(modal < params['modal_awal']).mean()*100:.1f}%",
            f"{_kekayaan_top_persen(modal, 0.1)*100:.1f}%",
            f"{_kekayaan_top_persen(modal, 0.2)*100:.1f}%",
        ]
    })


def _kekayaan_top_persen(modal: np.ndarray, persen: float) -> float:
    """Proporsi total modal yang dikuasai oleh top X% UMKM."""
    n_top = max(1, int(len(modal) * persen))
    modal_sorted = np.sort(modal)[::-1]
    return modal_sorted[:n_top].sum() / modal.sum()


# ============================================================
# 3. VISUALISASI INTERAKTIF — PLOTLY
# ============================================================

WARNA_SKENARIO = {
    'A': '#27ae60',  # Hijau: pasar konvensional (paling "sehat")
    'B': '#e74c3c',  # Merah: risiko pinjaman
    'C': '#2980b9',  # Biru: inovasi
    'D': '#8e44ad',  # Ungu: tak terduga
}


def plot_distribusi_histogram_interaktif(
    hasil: dict,
    hari_dipilih: Optional[list] = None
) -> go.Figure:
    """
    Histogram interaktif distribusi modal UMKM dengan slider waktu.
    Menampilkan evolusi distribusi dari awal hingga akhir simulasi.
    """
    history = hasil.get('history', {})
    skenario = hasil['skenario']
    warna = WARNA_SKENARIO[skenario]
    
    if not history:
        raise ValueError("Jalankan simulasi dengan simpan_history=True")
    
    hari_tersedia = sorted(history.keys())
    if hari_dipilih is None:
        # Pilih ~10 hari yang representatif
        idx = np.round(np.linspace(0, len(hari_tersedia) - 1, 10)).astype(int)
        hari_dipilih = [hari_tersedia[i] for i in idx]
    
    fig = go.Figure()
    
    modal_global_max = max(history[h].max() for h in hari_dipilih)
    modal_global_min = min(history[h].min() for h in hari_dipilih)
    bin_size = (modal_global_max - modal_global_min) / 20 / 1e6
    
    for i, hari in enumerate(hari_dipilih):
        modal_hari = history[hari] / 1e6  # Konversi ke juta Rp
        fig.add_trace(go.Histogram(
            x=modal_hari,
            name=f'Hari ke-{hari}',
            visible=(i == 0),
            marker_color=warna,
            opacity=0.80,
            xbins=dict(size=max(bin_size, 0.1)),
            hovertemplate='Modal: Rp %{x:.1f} juta<br>Jumlah UMKM: %{y}<extra></extra>'
        ))
    
    # Slider untuk navigasi waktu
    steps = [dict(
        method="update",
        args=[{"visible": [j == i for j in range(len(hari_dipilih))]}],
        label=f"Hari {hari}"
    ) for i, hari in enumerate(hari_dipilih)]
    
    fig.update_layout(
        title=dict(
            text=f'Distribusi Modal UMKM — Skenario {skenario}: {get_nama_skenario(skenario)}',
            font=dict(size=15)
        ),
        xaxis_title='Modal Usaha (Rp juta)',
        yaxis_title='Jumlah UMKM',
        sliders=[dict(active=0, steps=steps, pad={"t": 60},
                      currentvalue={"prefix": "Hari Transaksi: ", "font": {"size": 13}})],
        template='plotly_white',
        height=500,
        font=dict(family='Arial', size=12),
        showlegend=False,
        annotations=[dict(
            text='← Geser slider untuk melihat evolusi distribusi modal dari waktu ke waktu →',
            xref='paper', yref='paper', x=0.5, y=-0.25,
            showarrow=False, font=dict(size=11, color='gray')
        )]
    )
    
    return fig


def plot_lorenz_interaktif(hasil_skenario: dict) -> go.Figure:
    """
    Kurva Lorenz interaktif untuk membandingkan semua skenario.
    Menyertakan Koefisien Gini dan area antara Lorenz dengan diagonal.
    """
    fig = go.Figure()
    
    # Garis kesetaraan sempurna
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Kesetaraan Sempurna',
        line=dict(color='#95a5a6', dash='dot', width=1.5),
        hoverinfo='skip'
    ))
    
    for sku, hasil in hasil_skenario.items():
        pop_share, income_share = kurva_lorenz(hasil['modal_akhir'])
        gini = hitung_gini(hasil['modal_akhir'])
        warna = WARNA_SKENARIO[sku]
        
        # Area di bawah Lorenz (shaded)
        fig.add_trace(go.Scatter(
            x=pop_share, y=income_share,
            mode='lines',
            name=f'Skenario {sku}: {get_nama_skenario(sku)}<br>Gini = {gini:.3f}',
            line=dict(color=warna, width=2.5),
            fill='tozeroy',
            fillcolor=f'rgba({int(warna[1:3],16)},{int(warna[3:5],16)},{int(warna[5:7],16)},0.08)',
            hovertemplate=f'Skenario {sku}<br>%{{x:.1%}} UMKM → %{{y:.1%}} Modal<extra></extra>'
        ))
    
    # Annotasi Gini untuk tiap skenario
    annotations = []
    for i, (sku, hasil) in enumerate(hasil_skenario.items()):
        gini = hitung_gini(hasil['modal_akhir'])
        annotations.append(dict(
            x=0.02, y=0.85 - i * 0.12,
            xref='paper', yref='paper',
            text=f'<b>Skenario {sku}</b>: Gini = {gini:.3f}',
            showarrow=False,
            font=dict(size=11, color=WARNA_SKENARIO[sku]),
            align='left'
        ))
    
    fig.update_layout(
        title='Kurva Lorenz — Ketimpangan Distribusi Modal UMKM Yogyakarta',
        xaxis_title='Proporsi Kumulatif UMKM (termiskin → terkaya)',
        yaxis_title='Proporsi Kumulatif Modal',
        xaxis=dict(tickformat='.0%', range=[0, 1]),
        yaxis=dict(tickformat='.0%', range=[0, 1]),
        template='plotly_white',
        height=550,
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.95)',
                    bordercolor='lightgray', borderwidth=0.5),
        font=dict(family='Arial', size=12),
        annotations=annotations
    )
    
    return fig


def plot_koefisien_gini_evolusi(hasil_skenario: dict) -> go.Figure:
    """Menampilkan evolusi Koefisien Gini sepanjang waktu untuk setiap skenario."""
    fig = go.Figure()
    
    for sku, hasil in hasil_skenario.items():
        df_gini = hasil['gini_history']
        warna = WARNA_SKENARIO[sku]
        
        fig.add_trace(go.Scatter(
            x=df_gini['hari'],
            y=df_gini['gini'],
            mode='lines',
            name=f'Skenario {sku}: {get_nama_skenario(sku)}',
            line=dict(color=warna, width=2),
            opacity=0.85,
            hovertemplate=f'Skenario {sku}<br>Hari %{{x}}: Gini = %{{y:.4f}}<extra></extra>'
        ))
    
    # Garis referensi interpretasi Gini
    for level, label, color in [(0.3, 'Ketimpangan Rendah', '#27ae60'),
                                  (0.5, 'Ketimpangan Tinggi', '#e74c3c')]:
        fig.add_hline(y=level, line_dash='dash', line_color=color, opacity=0.5,
                      annotation_text=label, annotation_position='right')
    
    fig.update_layout(
        title='Evolusi Koefisien Gini Modal UMKM Sepanjang Waktu',
        xaxis_title='Hari Transaksi',
        yaxis_title='Koefisien Gini',
        yaxis=dict(range=[0, 1]),
        template='plotly_white',
        height=450,
        legend=dict(x=0.02, y=0.98),
        font=dict(family='Arial', size=12)
    )
    
    return fig


def plot_standar_deviasi_evolusi(hasil_skenario: dict) -> go.Figure:
    """Menampilkan evolusi standar deviasi modal (indikator penyebaran/ketimpangan)."""
    fig = go.Figure()
    
    for sku, hasil in hasil_skenario.items():
        df_std = hasil['std_history']
        warna = WARNA_SKENARIO[sku]
        
        fig.add_trace(go.Scatter(
            x=df_std['hari'],
            y=df_std['std'] / 1e6,  # Konversi ke juta Rp
            mode='lines',
            name=f'Skenario {sku}',
            line=dict(color=warna, width=2),
            fill='tozeroy',
            fillcolor=f'rgba({int(warna[1:3],16)},{int(warna[3:5],16)},{int(warna[5:7],16)},0.05)',
            hovertemplate=f'Hari %{{x}}: Std Dev = Rp %{{y:.2f}}jt<extra></extra>'
        ))
    
    fig.update_layout(
        title='Standar Deviasi Modal UMKM — Indikator Ketimpangan yang Tumbuh Secara Spontan',
        xaxis_title='Hari Transaksi',
        yaxis_title='Standar Deviasi Modal (Rp juta)',
        template='plotly_white',
        height=450,
        legend=dict(x=0.02, y=0.98),
        font=dict(family='Arial', size=12),
        annotations=[dict(
            text='Ketimpangan tumbuh paling cepat di 500-1000 hari pertama',
            xref='paper', yref='paper', x=0.5, y=0.95,
            showarrow=False, font=dict(size=11, color='gray'), bgcolor='rgba(255,255,255,0.8)'
        )]
    )
    
    return fig


def plot_distribusi_akhir_sorted(hasil_skenario: dict) -> go.Figure:
    """
    Bar chart distribusi modal akhir yang diurutkan (seperti referensi asli),
    dalam format subplots untuk membandingkan keempat skenario.
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[f'Skenario {sku}: {get_nama_skenario(sku)}'
                        for sku in hasil_skenario.keys()],
        vertical_spacing=0.15,
        horizontal_spacing=0.08
    )
    
    posisi = [(1, 1), (1, 2), (2, 1), (2, 2)]
    
    for (sku, hasil), (row, col) in zip(hasil_skenario.items(), posisi):
        modal_sorted = np.sort(hasil['modal_akhir']) / 1e6
        n = len(modal_sorted)
        
        # Warnai UMKM yang bangkrut (modal < 0) merah
        colors = ['#e74c3c' if v < 0 else WARNA_SKENARIO[sku] for v in modal_sorted]
        
        fig.add_trace(
            go.Bar(
                x=list(range(1, n + 1)),
                y=modal_sorted,
                marker_color=colors,
                opacity=0.85,
                name=f'Skenario {sku}',
                showlegend=False,
                hovertemplate=f'Skenario {sku}<br>UMKM ke-%{{x}}: Rp %{{y:.1f}}jt<extra></extra>'
            ),
            row=row, col=col
        )
        
        # Garis modal awal sebagai referensi
        modal_awal_jt = hasil['params']['modal_awal'] / 1e6
        fig.add_hline(
            y=modal_awal_jt, line_dash='dot', line_color='gray',
            opacity=0.5, row=row, col=col,
            annotation_text=f'Modal Awal: Rp{modal_awal_jt:.0f}jt'
        )
    
    fig.update_layout(
        title='Distribusi Modal Akhir UMKM (Diurutkan dari Termiskin ke Terkaya)',
        template='plotly_white',
        height=700,
        font=dict(family='Arial', size=11)
    )
    fig.update_xaxes(title_text='Rangking UMKM')
    fig.update_yaxes(title_text='Modal (Rp juta)')
    
    return fig


def plot_pareto_umkm(hasil: dict) -> go.Figure:
    """
    Grafik Pareto UMKM: menampilkan distribusi modal + kurva kumulatif.
    Mengidentifikasi berapa % UMKM yang menguasai 80% total modal.
    """
    modal = hasil['modal_akhir']
    skenario = hasil['skenario']
    
    df = pd.DataFrame({'modal': modal / 1e6})
    df.sort_values('modal', ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['proporsi'] = df['modal'] / df['modal'].sum()
    df['kumulatif'] = df['proporsi'].cumsum()
    
    # Temukan titik 80%
    idx_80 = df[df['kumulatif'] >= 0.8].index[0]
    pct_umkm_80 = (idx_80 + 1) / len(df) * 100
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Bar chart modal
    fig.add_trace(go.Bar(
        x=df.index + 1, y=df['modal'],
        marker_color=WARNA_SKENARIO[skenario],
        opacity=0.75,
        name='Modal (Rp juta)',
        hovertemplate='UMKM ke-%{x}: Rp %{y:.1f}jt<extra></extra>'
    ), secondary_y=False)
    
    # Kurva kumulatif
    fig.add_trace(go.Scatter(
        x=df.index + 1, y=df['kumulatif'],
        mode='lines',
        name='Kumulatif (%)',
        line=dict(color='#27ae60', width=2, dash='dash'),
        hovertemplate='%{x} UMKM terkaya menguasai %{y:.1%} modal<extra></extra>'
    ), secondary_y=True)
    
    # Garis vertikal & anotasi 80%
    fig.add_vline(x=idx_80 + 1, line_dash='dash', line_color='red', opacity=0.7)
    fig.add_annotation(
        x=idx_80 + 1, y=0.85, yref='y2',
        text=f'{pct_umkm_80:.0f}% UMKM terkaya<br>menguasai 80% modal',
        showarrow=True, arrowhead=2, ax=60, ay=-40,
        font=dict(size=11, color='red')
    )
    
    fig.update_layout(
        title=f'Grafik Pareto UMKM — Skenario {skenario}: {get_nama_skenario(skenario)}',
        xaxis_title='Rangking UMKM (dari Terkaya)',
        template='plotly_white',
        height=500,
        font=dict(family='Arial', size=12)
    )
    fig.update_yaxes(title_text='Modal (Rp juta)', secondary_y=False)
    fig.update_yaxes(title_text='Proporsi Modal Kumulatif', secondary_y=True,
                     tickformat='.0%', range=[0, 1.05])
    
    return fig


# ============================================================
# 4. ANALISIS SENSITIVITAS
# ============================================================

def analisis_sensitivitas_skenario_d(
    n_umkm: int = 100,
    modal_awal: float = 10_000_000,
    n_hari: int = 1000,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Analisis sensitivitas: pengaruh probabilitas musibah dan viral
    terhadap Koefisien Gini dan persentase UMKM bangkrut.
    """
    p_musibah_range = [0.001, 0.003, 0.005, 0.01, 0.02]
    p_viral_range = [0.001, 0.003, 0.005, 0.01]
    
    results = []
    total = len(p_musibah_range) * len(p_viral_range)
    print(f"Menjalankan {total} simulasi untuk analisis sensitivitas...")
    
    for i, pm in enumerate(p_musibah_range):
        for pv in p_viral_range:
            hasil = simulate_umkm_vectorized(
                n_umkm=n_umkm, modal_awal=modal_awal,
                n_hari=n_hari, skenario='D',
                p_musibah=pm, p_viral=pv,
                random_seed=random_seed, simpan_history=False
            )
            modal = hasil['modal_akhir']
            results.append({
                'p_musibah': pm,
                'p_viral': pv,
                'gini': round(hitung_gini(modal), 4),
                'pct_bangkrut': round((modal <= 0).mean() * 100, 1),
                'modal_tertinggi_jt': round(modal.max() / 1e6, 1),
                'modal_terendah_jt': round(modal.min() / 1e6, 1)
            })
    
    return pd.DataFrame(results)


def plot_heatmap_sensitivitas(df_sensitivity: pd.DataFrame) -> go.Figure:
    """Heatmap sensitivitas Gini terhadap parameter musibah dan viral."""
    pivot_gini = df_sensitivity.pivot(
        index='p_musibah', columns='p_viral', values='gini'
    )
    
    fig = go.Figure(go.Heatmap(
        z=pivot_gini.values,
        x=[f'{v*100:.1f}%' for v in pivot_gini.columns],
        y=[f'{v*100:.1f}%' for v in pivot_gini.index],
        colorscale='RdYlGn_r',  # Merah = Gini tinggi (buruk), hijau = rendah (baik)
        text=pivot_gini.values.round(3),
        texttemplate='%{text}',
        colorbar=dict(title='Koefisien Gini')
    ))
    
    fig.update_layout(
        title='Sensitivitas Koefisien Gini terhadap Probabilitas Musibah & Viral',
        xaxis_title='Probabilitas Viral per Hari (%)',
        yaxis_title='Probabilitas Musibah per Hari (%)',
        template='plotly_white',
        height=400,
        font=dict(family='Arial', size=12)
    )
    
    return fig


# ============================================================
# 5. MAIN — MENJALANKAN SELURUH SIMULASI
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("SIMULASI MONTE CARLO: DINAMIKA MODAL UMKM YOGYAKARTA")
    print("=" * 60)
    
    PARAMS_DASAR = dict(
        n_umkm=100,
        modal_awal=10_000_000,
        unit_transaksi=100_000,
        n_hari=1000,
        random_seed=42
    )
    
    # Jalankan semua skenario
    hasil_semua = {}
    for sku in ['A', 'B', 'C', 'D']:
        hasil_semua[sku] = simulate_umkm_vectorized(
            **PARAMS_DASAR,
            skenario=sku,
            simpan_history=True,
            n_snapshot=30
        )
    
    # Tampilkan ringkasan statistik
    print("\n" + "=" * 60)
    print("RINGKASAN STATISTIK AKHIR")
    print("=" * 60)
    for sku, hasil in hasil_semua.items():
        print(f"\nSkenario {sku}:")
        df_ring = ringkasan_statistik(hasil)
        print(df_ring.to_string(index=False))
    
    # Buat semua visualisasi
    print("\nMembuat visualisasi...")
    
    # 1. Kurva Lorenz semua skenario
    fig_lorenz = plot_lorenz_interaktif(hasil_semua)
    fig_lorenz.write_html('lorenz_curve_umkm.html')
    print("  ✓ lorenz_curve_umkm.html")
    
    # 2. Evolusi Gini
    fig_gini = plot_koefisien_gini_evolusi(hasil_semua)
    fig_gini.write_html('gini_evolusi_umkm.html')
    print("  ✓ gini_evolusi_umkm.html")
    
    # 3. Standar deviasi
    fig_std = plot_standar_deviasi_evolusi(hasil_semua)
    fig_std.write_html('std_evolusi_umkm.html')
    print("  ✓ std_evolusi_umkm.html")
    
    # 4. Distribusi akhir sorted
    fig_dist = plot_distribusi_akhir_sorted(hasil_semua)
    fig_dist.write_html('distribusi_akhir_umkm.html')
    print("  ✓ distribusi_akhir_umkm.html")
    
    # 5. Grafik Pareto
    for sku, hasil in hasil_semua.items():
        fig_pareto = plot_pareto_umkm(hasil)
        fig_pareto.write_html(f'pareto_skenario_{sku}.html')
        print(f"  ✓ pareto_skenario_{sku}.html")
    
    # 6. Histogram interaktif per skenario
    for sku, hasil in hasil_semua.items():
        fig_hist = plot_distribusi_histogram_interaktif(hasil)
        fig_hist.write_html(f'histogram_skenario_{sku}.html')
        print(f"  ✓ histogram_skenario_{sku}.html")
    
    print("\n✓ Semua visualisasi berhasil dibuat!")
    print("  Buka file .html di browser untuk tampilan interaktif Plotly.")
