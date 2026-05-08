"""
ShopLens Dashboard — E-Ticaret Analiz ve Öneri Sistemi
========================================================
Kullanım: streamlit run dashboard.py
"""

import warnings
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from joblib import load

warnings.filterwarnings("ignore")

# =============================================================================
# SAYFA AYARI
# =============================================================================
st.set_page_config(
    page_title="ShopLens",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,300..700,0..1,-50..200&display=swap');

html, body, [data-testid="stAppViewContainer"], [data-testid="stMarkdownContainer"] {
  font-family: 'Inter', sans-serif !important;
}
/* Streamlit ikon fontlarını ezmemek için Material Symbols kendi fontunda bırakılır. */
.material-symbols-rounded,
.material-symbols-outlined,
.material-icons,
[data-testid="stIconMaterial"],
[data-testid="stSidebarCollapseButton"] span,
[data-testid="stBaseButton-header"] span,
[data-testid="stBaseButton-headerNoPadding"] span,
[data-testid="collapsedControl"] span {
  font-family: 'Material Symbols Rounded', 'Material Symbols Outlined', 'Material Icons' !important;
  font-weight: normal !important;
  font-style: normal !important;
  font-size: 1.45rem !important;
  line-height: 1 !important;
  letter-spacing: normal !important;
  text-transform: none !important;
  white-space: nowrap !important;
  word-wrap: normal !important;
  direction: ltr !important;
  font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24 !important;
  -webkit-font-feature-settings: 'liga' !important;
  font-feature-settings: 'liga' !important;
  -webkit-font-smoothing: antialiased !important;
}
/* Streamlit'in sidebar aç/kapa ikonu bazı tarayıcılarda yazı olarak göründüğü için gizlenir. */
[data-testid="stSidebarCollapseButton"],
[data-testid="collapsedControl"],
button[aria-label="Close sidebar"],
button[aria-label="Open sidebar"] {
  display: none !important;
}
header[data-testid="stHeader"] {
  background: transparent !important;
  height: 2.25rem !important;
}
#MainMenu, footer { display:none !important; }
.stApp { background:#F1F5F9; }
.block-container {
  padding: 0 24px 2.5rem 32px !important;
  max-width: 100% !important;
}
@media (max-width: 900px) {
  .block-container { padding: 0 14px 2rem 14px !important; }
}

/* ── BANNER ────────────────────────────────────────────── */
.banner {
  background: linear-gradient(135deg, #1E3A5F 0%, #0F2441 60%, #1a3a6b 100%);
  padding: 0;
  position: relative;
  overflow: hidden;
}
.banner::before {
  content: '';
  position: absolute; top: 0; left: 0; right: 0; bottom: 0;
  background: radial-gradient(ellipse at 80% 50%, rgba(249,115,22,0.12) 0%, transparent 60%),
              radial-gradient(ellipse at 20% 80%, rgba(59,130,246,0.08) 0%, transparent 50%);
}
.banner-inner {
  position: relative; z-index: 1;
  padding: 22px 36px 18px;
  display: flex; align-items: center; justify-content: space-between;
}
.banner-left { flex: 1; }
.brand-row { display: flex; align-items: baseline; gap: 6px; }
.brand-name {
  font-size: 1.9rem; font-weight: 800; color: white;
  letter-spacing: -0.03em; line-height: 1;
}
.brand-accent { color: #F97316; }
.brand-ver {
  font-size: 0.6rem; font-weight: 600; color: rgba(249,115,22,0.7);
  letter-spacing: 0.12em; text-transform: uppercase; margin-left: 4px;
}
.banner-tagline {
  font-size: 0.72rem; color: rgba(255,255,255,0.45);
  font-weight: 400; margin-top: 4px; letter-spacing: 0.04em;
}
.banner-problem {
  font-size: 0.82rem; color: rgba(255,255,255,0.72);
  margin-top: 10px; line-height: 1.55; max-width: 520px;
  border-left: 2px solid rgba(249,115,22,0.5);
  padding-left: 12px;
}
.banner-chips {
  display: flex; gap: 6px; margin-top: 12px; flex-wrap: wrap;
}
.chip {
  display: inline-block;
  background: rgba(255,255,255,0.07);
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 20px;
  padding: 3px 10px;
  font-size: 0.62rem; font-weight: 600;
  color: rgba(255,255,255,0.65);
  letter-spacing: 0.05em;
}
.banner-stats {
  display: flex; gap: 20px; align-items: center;
}
.bstat {
  text-align: center; min-width: 70px;
}
.bstat-num {
  font-size: 1.45rem; font-weight: 800; color: white; line-height: 1;
  display: block;
}
.bstat-num.accent { color: #F97316; }
.bstat-lbl {
  font-size: 0.58rem; color: rgba(255,255,255,0.4);
  font-weight: 500; text-transform: uppercase;
  letter-spacing: 0.08em; margin-top: 3px; display: block;
}
.bstat-div {
  width: 1px; height: 32px;
  background: rgba(255,255,255,0.1);
}

/* ── NAVİGASYON — SOL SIDEBAR TARZ ────────────────────── */
.nav-bar {
  background: white;
  border-right: 1px solid #E2E8F0;
  padding: 16px 10px;
  min-height: calc(100vh - 120px);
}
.nav-title {
  font-size: 0.55rem; font-weight: 700; color: #94A3B8;
  text-transform: uppercase; letter-spacing: 0.1em;
  padding: 0 8px; margin-bottom: 8px;
}
.nav-item {
  display: flex; align-items: center; gap: 9px;
  padding: 9px 10px; border-radius: 8px;
  cursor: pointer; margin-bottom: 2px;
  font-size: 0.78rem; font-weight: 500; color: #64748B;
  transition: all 0.15s ease;
  border: 1px solid transparent;
}
.nav-item:hover {
  background: #FFF7ED; color: #C2410C; border-color: #FED7AA;
}
.nav-item.active {
  background: linear-gradient(135deg, #FFF7ED, #FEF3C7);
  color: #C2410C; font-weight: 700;
  border-color: #FED7AA;
}
.nav-icon { font-size: 0.9rem; width: 18px; text-align: center; }
.nav-sep {
  height: 1px; background: #F1F5F9; margin: 10px 8px;
}

/* ── KPI KARTLARI ────────────────────────────────────── */
.kpi {
  background: white; border-radius: 14px;
  border: 1px solid #E2E8F0;
  padding: 18px 20px 16px;
  position: relative; overflow: hidden;
  box-shadow: 0 1px 4px rgba(0,0,0,0.04), 0 4px 16px rgba(0,0,0,0.03);
}
.kpi::before {
  content: ''; position: absolute; top: 0; left: 0; right: 0;
  height: 3px; border-radius: 14px 14px 0 0;
}
.kpi.orange::before { background: linear-gradient(90deg,#F97316,#FB923C); }
.kpi.blue::before   { background: linear-gradient(90deg,#3B82F6,#60A5FA); }
.kpi.green::before  { background: linear-gradient(90deg,#16A34A,#22C55E); }
.kpi.amber::before  { background: linear-gradient(90deg,#D97706,#FBBF24); }
.kpi.red::before    { background: linear-gradient(90deg,#DC2626,#F87171); }
.kpi.navy::before   { background: linear-gradient(90deg,#1E3A5F,#3B82F6); }

.kpi-label {
  font-size: 0.63rem; font-weight: 700; text-transform: uppercase;
  letter-spacing: 0.09em; color: #94A3B8; margin-bottom: 9px;
}
.kpi-num {
  font-size: 1.95rem; font-weight: 800; color: #1E293B; line-height: 1;
  letter-spacing: -0.02em;
}
.kpi-num small { font-size: 1rem; color: #94A3B8; font-weight: 500; }
.kpi-badge {
  display: inline-block; font-size: 0.65rem; font-weight: 700;
  padding: 2px 7px; border-radius: 20px; margin-top: 8px;
}
.kpi-badge.up   { background:#DCFCE7; color:#16A34A; }
.kpi-badge.down { background:#FEE2E2; color:#DC2626; }
.kpi-badge.neu  { background:#F1F5F9; color:#64748B; }

/* ── BÖLÜM BAŞLIĞI ─────────────────────────────────── */
.sec-head {
  font-size: 0.88rem; font-weight: 700; color: #1E293B;
  margin: 24px 0 16px; padding-bottom: 12px;
  border-bottom: 2px solid #F1F5F9;
  display: flex; align-items: center; gap: 8px;
}
.sec-tag {
  display: inline-block; font-size: 0.58rem; font-weight: 700;
  letter-spacing: 0.08em; text-transform: uppercase;
  background: #FFF7ED; border: 1px solid #FED7AA;
  color: #EA580C; padding: 2px 8px; border-radius: 4px;
}

/* ── KART ────────────────────────────────────────────── */
.card {
  background: white; border-radius: 12px;
  border: 1px solid #E2E8F0;
  padding: 18px;
  box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}
.table-card {
  padding: 0;
  overflow: hidden;
}
.table-title {
  padding: 16px 18px 12px;
  font-size: 0.74rem;
  font-weight: 800;
  color: #334155;
  letter-spacing: 0.02em;
  border-bottom: 1px solid #E2E8F0;
  background: #FFFFFF;
}
.table-body {
  padding: 0;
}
.eda-html-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.78rem;
  color: #334155;
}
.eda-html-table th {
  background: #F8FAFC;
  color: #64748B;
  font-weight: 700;
  text-align: left;
  padding: 11px 12px;
  border-bottom: 1px solid #E2E8F0;
}
.eda-html-table td {
  padding: 10px 12px;
  border-bottom: 1px solid #E2E8F0;
}
.eda-html-table tr:last-child td {
  border-bottom: 0;
}
.eda-html-table td:nth-child(2),
.eda-html-table td:nth-child(3) {
  font-weight: 700;
}
.merge-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 14px;
  margin-top: 12px;
}
.merge-card {
  background: #FFFFFF;
  border: 1px solid #E2E8F0;
  border-radius: 12px;
  padding: 16px 16px 14px;
  box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}
.merge-card.blue { border-top: 3px solid #3B82F6; }
.merge-card.green { border-top: 3px solid #16A34A; }
.merge-card.orange { border-top: 3px solid #F97316; }
.merge-title {
  font-size: 0.78rem;
  font-weight: 800;
  color: #1E293B;
  margin-bottom: 9px;
}
.merge-path {
  font-size: 0.74rem;
  color: #334155;
  line-height: 1.75;
}
.merge-path code {
  color: #1E3A5F;
  background: #EFF6FF;
  border: 1px solid #DBEAFE;
  border-radius: 5px;
  padding: 1px 5px;
}
.merge-desc {
  margin-top: 9px;
  font-size: 0.72rem;
  color: #64748B;
  line-height: 1.55;
}
.critical-note {
  margin-top: 14px;
  padding: 13px 15px;
  background: #FFF7ED;
  border: 1px solid #FED7AA;
  border-left: 4px solid #F97316;
  border-radius: 10px;
  font-size: 0.78rem;
  color: #92400E;
  line-height: 1.6;
}
@media (max-width: 1100px) {
  .merge-grid { grid-template-columns: 1fr; }
}

/* ── İNFO KUTUSU ─────────────────────────────────────── */
.info-box {
  background: #EFF6FF; border: 1px solid #BFDBFE;
  border-radius: 10px; padding: 14px 16px;
  font-size: 0.8rem; color: #1E40AF; line-height: 1.65;
  margin: 20px 0 18px 0;
}
.info-box strong { color: #1D4ED8; }
.info-box em { color: #1D4ED8; font-style: normal; font-weight: 600; }

/* ── YORUM KARTI ─────────────────────────────────────── */
.yorum-card {
  background: white; border: 1px solid #E2E8F0;
  border-radius: 10px; padding: 14px 16px;
  margin-bottom: 10px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.yorum-text { font-size: 0.85rem; color: #334155; line-height: 1.65; }
.yorum-meta { font-size: 0.68rem; color: #94A3B8; margin-top: 7px; }

/* ── PILL ─────────────────────────────────────────────── */
.pill {
  background: white; border: 1px solid #E2E8F0;
  border-radius: 9px; padding: 12px 14px;
  margin-bottom: 9px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.03);
}
.pill-label { font-size: 0.6rem; color: #94A3B8; text-transform: uppercase;
              letter-spacing: 0.06em; margin-bottom: 3px; }
.pill-val   { font-size: 1.05rem; font-weight: 700; line-height: 1.2; }
.pill-sub   { font-size: 0.68rem; color: #94A3B8; margin-top: 4px; }

/* ── SİDEBAR NAV BUTONLARI ──────────────────────────── */
section[data-testid="stSidebar"] div.stButton > button {
  width: 100% !important; text-align: left !important;
  border-radius: 8px !important; font-size: 0.8rem !important;
  font-weight: 500 !important; padding: 9px 14px !important;
  border: 1px solid transparent !important;
  background: transparent !important; color: #475569 !important;
  box-shadow: none !important;
  margin-bottom: 2px !important;
  transition: all 0.12s ease !important;
}
section[data-testid="stSidebar"] div.stButton > button p {
  color: #475569 !important; text-align: left !important;
}
section[data-testid="stSidebar"] div.stButton > button:hover {
  background: #FFF7ED !important; color: #C2410C !important;
  border-color: #FED7AA !important;
}
section[data-testid="stSidebar"] div.stButton > button:hover p {
  color: #C2410C !important;
}
/* Sidebar arka planı */
section[data-testid="stSidebar"] {
  background: white !important;
  border-right: 1px solid #E2E8F0 !important;
}
section[data-testid="stSidebar"] > div {
  padding-top: 16px !important;
  padding-left: 12px !important;
  padding-right: 12px !important;
}

hr { border-color: #E2E8F0 !important; margin: 16px 0 !important; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# GRAFİK TEMA
# =============================================================================
TURUNCU  = "#F97316"; LACIVERT = "#1E3A5F"; MAVI = "#3B82F6"
YESIL    = "#16A34A"; KIRMIZI  = "#DC2626"; SARI = "#FBBF24"

DUYGU_RENK = {"positive":YESIL,"neutral":"#94A3B8","negative":KIRMIZI}
SEG_RENK   = {"Şampiyon":TURUNCU,"Sadık":YESIL,"Yeni":MAVI,
              "Potansiyel":SARI,"Risk Altında":"#EA580C","Kayıp":KIRMIZI}

TEMEL = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="white",
    font=dict(family="Inter",color="#64748B",size=11),
    margin=dict(l=10,r=10,t=8,b=10),
)
EKSEN = dict(gridcolor="#F1F5F9",linecolor="#E2E8F0",
             tickfont=dict(color="#94A3B8",size=10),zeroline=False)

def tema(fig, h=300, baslik="", **ekstra):
    """Grafiklere açık tema + tutarlı stil uygular. title_font TEMEL içinde yok."""
    d = {}; d.update(TEMEL); d["height"] = h
    if baslik:
        d["title"] = baslik
        d["title_font"] = dict(family="Inter",color="#1E293B",size=12)
        d["margin"] = dict(l=10,r=10,t=40,b=10)
    if "legend" not in ekstra:
        d["legend"] = dict(bgcolor="rgba(255,255,255,0.95)",
                           bordercolor="#E2E8F0",borderwidth=1,
                           font=dict(color="#334155",size=10))
    d.update(ekstra)
    fig.update_layout(**d)
    fig.update_xaxes(**EKSEN)
    fig.update_yaxes(**EKSEN)
    return fig

def html_tablo(df):
    """Küçük özet tablolarını dashboard tasarımına uygun HTML tabloya çevirir."""
    return df.to_html(index=False, classes="eda-html-table", border=0, escape=False)


# =============================================================================
# VERİ YÜKLEME
# =============================================================================
def yukle():
    try:
        funnel  = pd.read_csv("outputs/funnel.csv")
        duygu   = pd.read_csv("outputs/duygu.csv")
        urunler = pd.read_csv("outputs/urun_skorlar.csv")
        top50   = pd.read_csv("outputs/top50.csv")
        rfm     = pd.read_csv("outputs/rfm.csv")
        mkat    = pd.read_csv("outputs/musteri_kat.csv")
        eda     = pd.read_csv("outputs/eda_ozet.csv")
        mst     = pd.read_csv("data/customers.csv", parse_dates=["signup_date"])
        sip     = pd.read_csv("data/orders.csv",    parse_dates=["order_time"])
        kal     = pd.read_csv("data/order_items.csv")
        try:
            davranis = pd.read_csv("outputs/musteri_urun_davranis.csv")
            kmet = pd.read_csv("outputs/kisisel_model_met.csv")
        except FileNotFoundError:
            davranis = pd.DataFrame()
            kmet = pd.DataFrame()
        return funnel, duygu, urunler, top50, rfm, mkat, eda, mst, sip, kal, davranis, kmet
    except FileNotFoundError as e:
        st.error(f"❌ {e}")
        st.info("💡 Önce çalıştırın: `python pipeline.py`")
        st.stop()

(funnel_df, duygu_df, urun_df, top50_df,
 rfm_df, mkat_df, eda_df, musteri_df, siparis_df, kalem_df,
 davranis_df, kisisel_met_df) = yukle()

ADIM_AD = {"page_view":"👀 Sayfa Görüntüleme","add_to_cart":"🛒 Sepete Ekleme",
           "checkout":"💳 Ödeme Başlatma","purchase":"✅ Satın Alma"}
OZELLIK_AD = {"sepet_orani":"Sepete Ekleme Oranı","ort_rating":"Müşteri Puanı",
              "ort_yildiz":"Duygu Skoru","margin_usd":"Ürün Marjı",
              "price_usd":"Fiyat","page_view":"Sayfa Görüntüleme",
              "yorum_sayisi":"Yorum Sayısı"}


@st.cache_resource
def kisisel_model_yukle():
    """Kişiye özel satın alma olasılığı modelini yükler."""
    model_path = Path("models/kisisel_satin_alma_model.pkl")
    if not model_path.exists():
        return None
    return load(model_path)


def kisisel_satin_alma_tahmini(customer_id, adet=8):
    """
    Seçilen müşteri için satın alma olasılığı yüksek ürünleri döndürür.

    Model satırı: müşteri + ürün çifti.
    Çıktı: her ürün için satın alma olasılığı.
    """
    paket = kisisel_model_yukle()
    if paket is None:
        return pd.DataFrame()

    model = paket["model"]
    ozellikler = paket["ozellikler"]

    # Müşterinin daha önce satın aldığı ürünler öneriden çıkarılır.
    musteri_siparisleri = siparis_df.loc[
        siparis_df["customer_id"] == customer_id,
        "order_id",
    ].tolist()
    alinan_urunler = kalem_df.loc[
        kalem_df["order_id"].isin(musteri_siparisleri),
        "product_id",
    ].unique()

    adaylar = urun_df[~urun_df["product_id"].isin(alinan_urunler)].copy()
    adaylar["customer_id"] = customer_id

    # Müşterinin bu ürünle geçmiş etkileşimi varsa eklenir.
    if davranis_df.empty:
        adaylar["page_view"] = 0
        adaylar["add_to_cart"] = 0
    else:
        gecmis = davranis_df[davranis_df["customer_id"] == customer_id][
            ["product_id", "page_view", "add_to_cart"]
        ]
        adaylar = adaylar.drop(columns=["page_view", "add_to_cart"], errors="ignore")
        adaylar = adaylar.merge(gecmis, on="product_id", how="left")
        adaylar[["page_view", "add_to_cart"]] = adaylar[["page_view", "add_to_cart"]].fillna(0)

    # RFM müşteri özellikleri.
    rfm_satir = rfm_df[rfm_df["customer_id"] == customer_id]
    if rfm_satir.empty:
        adaylar["recency"] = rfm_df["recency"].max() + 365
        adaylar["frequency"] = 0
        adaylar["monetary"] = 0
        adaylar["R"] = 0
        adaylar["F"] = 0
        adaylar["M"] = 0
        adaylar["age"] = 0
    else:
        for kolon in ["recency", "frequency", "monetary", "R", "F", "M", "age"]:
            adaylar[kolon] = rfm_satir.iloc[0].get(kolon, 0)

    # Müşterinin kategori ilgisi satın alma geçmişinden değil,
    # görüntüleme ve sepete ekleme davranışından hesaplanır.
    if davranis_df.empty:
        adaylar["kategori_ilgi_orani"] = 0.0
    else:
        davranis_kat = davranis_df[davranis_df["customer_id"] == customer_id].merge(
            urun_df[["product_id", "category"]],
            on="product_id",
            how="left",
        )
        davranis_kat["kategori_davranis_puani"] = (
            davranis_kat["page_view"] + davranis_kat["add_to_cart"] * 3
        )
        tercih = (
            davranis_kat.groupby("category")["kategori_davranis_puani"]
            .sum()
            .reset_index()
        )
        toplam_puan = tercih["kategori_davranis_puani"].sum()
        adaylar = adaylar.merge(tercih, on="category", how="left")
        adaylar["kategori_davranis_puani"] = adaylar["kategori_davranis_puani"].fillna(0)
        adaylar["kategori_ilgi_orani"] = np.where(
            toplam_puan > 0,
            adaylar["kategori_davranis_puani"] / toplam_puan,
            0.0,
        )

    for kolon in ozellikler:
        if kolon not in adaylar.columns:
            adaylar[kolon] = 0
        adaylar[kolon] = adaylar[kolon].fillna(0)

    adaylar["satin_alma_olasiligi"] = model.predict_proba(adaylar[ozellikler])[:, 1]

    sonuc = (
        adaylar.sort_values("satin_alma_olasiligi", ascending=False)
        .head(adet)
        .copy()
    )
    return sonuc

# =============================================================================
# BANNER
# =============================================================================
# Kişiye özel satın alma tahmini metriklerini banner için önceden yükle
try:
    _met_df  = pd.read_csv("outputs/kisisel_model_met.csv")
    _auc_str = f"{float(_met_df['auc'].iloc[0]):.3f}"
    _dogr_str = f"% {float(_met_df['dogruluk'].iloc[0])*100:.1f}"
except Exception:
    _auc_str  = "—"
    _dogr_str = "—"

st.markdown(f"""
<div class="banner">
  <div class="banner-inner">
    <div class="banner-left">
      <div class="brand-row">
        <span class="brand-name">Shop<span class="brand-accent">Lens</span></span>
        <span class="brand-ver">v2.0</span>
      </div>
      <div class="banner-tagline">E-Ticaret Davranış Analizi ve Akıllı Ürün Öneri Sistemi</div>
      <div class="banner-problem">
        <strong>Problem:</strong> Çok ürünlü e-ticaret sitelerinde doğru ürünü doğru müşteriye göstermek zordur.
        Yalnızca “en çok satan” listeleri kişisel ilgi ve satın alma davranışını yeterince yansıtmaz.<br>
        <strong>Çözüm:</strong> Görüntüleme, sepete ekleme, sipariş, yorum ve RFM verileri birleştirilerek
        hem genel ürün skoru hem de kişiye özel satın alma tahmini üretilir.
      </div>
      <div class="banner-chips">
        <span class="chip">İkili Sınıflandırma</span>
        <span class="chip">Random Forest</span>
        <span class="chip">VADER Duygu Analizi</span>
        <span class="chip">RFM Segmentasyon</span>
        <span class="chip">Clickstream + Satış</span>
        <span class="chip">1M+ Gözlem</span>
      </div>
    </div>
    <div class="banner-stats">
      <div class="bstat">
        <span class="bstat-num">1.0M</span>
        <span class="bstat-lbl">Gözlem</span>
      </div>
      <div class="bstat-div"></div>
      <div class="bstat">
        <span class="bstat-num">7</span>
        <span class="bstat-lbl">Veri Seti</span>
      </div>
      <div class="bstat-div"></div>
      <div class="bstat">
        <span class="bstat-num accent">{_auc_str}</span>
        <span class="bstat-lbl">Kişisel AUC</span>
      </div>
      <div class="bstat-div"></div>
      <div class="bstat">
        <span class="bstat-num">20K</span>
        <span class="bstat-lbl">Toplam Müşteri</span>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# LAYOUT: SOL NAV + SAĞ İÇERİK
# =============================================================================
SAYFALAR = [
    ("📊", "Özet"),
    ("🔍", "EDA"),
    ("📈", "Analiz"),
    ("🎯", "Genel Öneriler"),
    ("🤖", "Model"),
    ("💬", "Duygu"),
    ("👤", "Müşteri"),
    ("⚙️", "İstatistik"),
]

sayfa = st.session_state.get("s", "Özet")
if sayfa == "Öneriler":
    sayfa = "Genel Öneriler"
    st.session_state["s"] = sayfa

# ── Sidebar navigasyon — her zaman solda sabit, boşluk sorunu yok ──────────
with st.sidebar:
    st.markdown("""
    <div style="padding:4px 0 14px">
      <div style="font-size:1.05rem;font-weight:800;color:#1E3A5F;letter-spacing:-0.02em">
        🛒 ShopLens
      </div>
      <div style="font-size:0.58rem;color:#94A3B8;text-transform:uppercase;
                  letter-spacing:0.1em;margin-top:3px">E-Ticaret Analiz Platformu</div>
    </div>
    <hr style="border-color:#F1F5F9;margin:0 0 14px">
    <div style="font-size:0.58rem;font-weight:700;color:#94A3B8;
                text-transform:uppercase;letter-spacing:0.1em;
                margin-bottom:10px">Navigasyon</div>
    """, unsafe_allow_html=True)
    for icon, isim in SAYFALAR:
        if st.button(f"{icon}  {isim}", key=f"nav_{isim}", use_container_width=True):
            st.session_state["s"] = isim
            sayfa = isim
    st.markdown("""
    <div style="position:fixed;bottom:20px;left:0;width:245px;
                padding:12px 16px;font-size:0.6rem;color:#CBD5E1;
                border-top:1px solid #F1F5F9;background:#F8FAFC">
      Python · Streamlit · VADER<br>scikit-learn · Plotly · Pandas
    </div>
    """, unsafe_allow_html=True)

# ── Ana içerik ──────────────────────────────────────────────────────────────
st.markdown('<div style="padding:22px 16px 40px;">', unsafe_allow_html=True)

# =========================================================================
# SAYFA 1 — ÖZET
# =========================================================================
if sayfa == "Özet":
    st.markdown('''<div class="sec-head">📊 Genel Bakış <span class="sec-tag">Özet</span></div>''',
                unsafe_allow_html=True)

    pos_n  = (duygu_df["duygu"]=="positive").sum()
    rec_n  = int(urun_df["onerilir"].sum())
    ort_sk = urun_df["oneri_skoru"].mean()
    ort_rt = urun_df["ort_rating"].mean()
    page_view_sayisi = float(funnel_df.loc[funnel_df["adim"] == "page_view", "event_sayisi"].iloc[0])
    purchase_sayisi = float(funnel_df.loc[funnel_df["adim"] == "purchase", "event_sayisi"].iloc[0])
    genel_donusum = purchase_sayisi / page_view_sayisi if page_view_sayisi else 0

    k1,k2,k3,k4,k5 = st.columns(5)
    for kol, lbl, val, sub, cls in [
        (k1,"Toplam Ürün",f"{len(urun_df):,}",f"↗ {rec_n} önerilir","orange"),
        (k2,"Yorum Analizi",f"{len(duygu_df):,}",f"↗ {pos_n:,} pozitif","blue"),
        (k3,"Ort. Skor",f"{ort_sk:.3f}",f"Maks: {urun_df['oneri_skoru'].max():.3f}","green"),
        (k4,"Ort. Rating",f"{ort_rt:.2f}","/5 puan","amber"),
        (k5,"Genel Dönüşüm",f"{genel_donusum:.1%}","purchase/page_view","navy"),
    ]:
        kol.markdown(f'''
        <div class="kpi {cls}">
          <div class="kpi-label">{lbl}</div>
          <div class="kpi-num">{val}</div>
          <div class="kpi-badge {"up" if "↗" in sub else "neu"}">{sub}</div>
        </div>''', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('''<div class="sec-head">📊 Dönüşüm Hunisi <span class="sec-tag">Adım Adım</span></div>''',
                unsafe_allow_html=True)

    hk, pk = st.columns([3, 1])
    rk_h = [LACIVERT, MAVI, TURUNCU, YESIL]
    with hk:
        fig = go.Figure()
        for i, satir in funnel_df.iterrows():
            isim = ADIM_AD.get(satir["adim"], satir["adim"])
            fig.add_trace(go.Bar(
                y=[isim], x=[satir["event_sayisi"]], orientation="h",
                marker_color=rk_h[i], showlegend=False,
                text=f"  {satir['event_sayisi']:,}",
                textposition="outside", textfont=dict(color="#334155",size=11),
            ))
        tema(fig, h=225, bargap=0.38,
             xaxis=dict(**EKSEN,title="Event Sayısı", range=[0, int(funnel_df["event_sayisi"].max() * 1.12)]),
             yaxis=dict(**EKSEN,autorange="reversed"))
        fig.update_traces(cliponaxis=False)
        fig.update_layout(showlegend=False, margin=dict(l=10, r=95, t=8, b=10))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.caption("📌 Ödeme başlatanların %74.8'i satın alıyor — bu geçiş oldukça sağlıklı. Asıl kayıp görüntülemeden sepete eklemeye geçişte: her 100 kullanıcıdan yalnızca 26'sı sepete ekliyor.")

    with pk:
        for idx,(_, satir) in enumerate(funnel_df.iterrows()):
            c = rk_h[idx]; p = satir["onceki_pct"]; t = satir["toplam_pct"]
            isim = ADIM_AD.get(satir["adim"],satir["adim"])
            st.markdown(f'''
            <div class="pill" style="border-left:3px solid {c}">
              <div class="pill-label">{isim}</div>
              <div class="pill-val" style="color:{c}">%{p:.1f}</div>
              <div class="pill-sub">öncekinden · %{t:.1f} toplamdan</div>
            </div>''', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('''<div class="sec-head">💬 Duygu & Kategori</div>''',
                unsafe_allow_html=True)
    dk, kk = st.columns(2)
    with dk:
        sayim = duygu_df["duygu"].value_counts()
        pos_p = sayim.get("positive",0)/sayim.sum()*100
        fig = go.Figure(go.Pie(
            labels=[l.capitalize() for l in sayim.index], values=sayim.values, hole=0.58,
            marker=dict(colors=[DUYGU_RENK.get(l,"#999") for l in sayim.index],
                        line=dict(color="white",width=3)),
        ))
        fig.add_annotation(text=f"<b>{pos_p:.0f}%</b><br><span style='font-size:9px'>pozitif</span>",
                           x=0.5,y=0.5,showarrow=False,font=dict(size=16,color=YESIL))
        tema(fig, h=250, showlegend=True,
             legend=dict(orientation="h",y=-0.1,bgcolor="rgba(0,0,0,0)",
                         font=dict(color="#334155",size=10)))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("📌 10.780 yorumun %70.7'si pozitif — müşteri memnuniyeti yüksek. Bu oran, öneri formülümüzün %18'ini oluşturan duygu skorunu besliyor.")
    with kk:
        kat = urun_df.groupby("category")["oneri_skoru"].mean().reset_index().sort_values("oneri_skoru")
        fig = px.bar(kat, x="oneri_skoru", y="category", orientation="h",
                     color="oneri_skoru",
                     color_continuous_scale=[[0,"#DBEAFE"],[0.5,MAVI],[1,LACIVERT]],
                     text=kat["oneri_skoru"].apply(lambda x: f"{x:.3f}"))
        fig.update_traces(textposition="outside",textfont=dict(color="#334155",size=10))
        fig.update_coloraxes(showscale=False)
        tema(fig, h=250, baslik="Kategori Ort. Skor",
             xaxis=dict(**EKSEN,title="Ort. Skor"), yaxis=dict(**EKSEN,title=""))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("📌 Kitap ve oyuncak kategorileri en yüksek ortalama öneri skoruna sahip. Elektronik kategorisi belirgin biçimde geride — fiyat-satış dengesi daha zayıf.")

# =========================================================================
# SAYFA 2 — EDA
# =========================================================================
elif sayfa == "EDA":
    st.markdown('''<div class="sec-head">🔍 Keşifsel Veri Analizi <span class="sec-tag">EDA</span></div>''',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
      <strong>Keşifsel Veri Analizi (EDA)</strong> nedir?<br>
      Modeli geliştirmeden önce veriyi <em>anlamak</em> için yapılan ilk keşif aşamasıdır.
      Hangi değişkenler anlamlı? Verinin dağılımı nasıl? Outlier var mı?
      Bu sorulara cevap aranır. EDA bulguları doğrudan feature engineering kararlarını etkiler.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # EDA özet tablosu
    st.markdown('''<div class="sec-head">📋 Veri Özeti</div>''', unsafe_allow_html=True)
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    a1, a2 = st.columns(2, gap="large")
    with a1:
        tablo = pd.DataFrame({
            "Tablo": ["events.csv","products.csv","reviews.csv","sessions.csv",
                      "order_items.csv","customers.csv","orders.csv"],
            "Satır": ["760.958","1.197","10.780","120.000","59.163","20.000","33.580"],
            "Sütun": ["10","6","6","6","5","7","10"],
            "Bağlantı": ["session_id","product_id","product_id, order_id",
                          "session_id, customer_id","order_id","customer_id","customer_id, order_id"],
        })
        st.markdown(
            '<div class="card table-card"><div class="table-title">Veri Seti Boyutları</div>'
            f'<div class="table-body">{html_tablo(tablo)}</div></div>',
            unsafe_allow_html=True
        )

    with a2:
        null_df = pd.DataFrame({
            "Sütun": ["events.product_id","events.qty","events.cart_size",
                      "events.payment","events.discount_pct","events.amount_usd"],
            "Eksik Sayısı": ["78.489","617.832","716.049","727.378","727.378","727.378"],
            "Oran": ["%10.3","%81.2","%94.1","%95.6","%95.6","%95.6"],
            "Açıklama": ["checkout+purchase=boş (beklenen)","Sadece purchase dolu",
                          "Sadece add_to_cart dolu","Sadece purchase dolu",
                          "Sadece purchase dolu","Sadece purchase dolu"],
        })
        st.markdown(
            '<div class="card table-card"><div class="table-title">Eksik Değer Analizi</div>'
            f'<div class="table-body">{html_tablo(null_df)}</div></div>',
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('''<div class="sec-head">📈 EDA Grafikleri</div>''', unsafe_allow_html=True)

    b1, b2 = st.columns(2)
    event_sirasi = ["page_view", "add_to_cart", "checkout", "purchase"]
    ev_say = (
        funnel_df.set_index("adim")["event_sayisi"]
        .reindex(event_sirasi)
        .dropna()
        .astype(int)
    )
    with b1:
        fig = px.bar(x=ev_say.index, y=ev_say.values,
                     color=ev_say.values,
                     color_continuous_scale=[[0,MAVI],[1,LACIVERT]],
                     text=[f"{v:,}" for v in ev_say.values],
                     labels={"x":"Event Tipi","y":"Sayı"})
        fig.update_traces(textposition="outside", textfont=dict(color="#334155", size=11))
        fig.update_coloraxes(showscale=False)
        # Barların üstündeki sayıların kesilmemesi için y ekseni biraz geniş tutuldu
        tema(fig, h=300, baslik="Event Tipi Dağılımı",
             xaxis=dict(**EKSEN, title=""),
             yaxis=dict(**EKSEN, title="Kayıt Sayısı",
                        range=[0, int(ev_say.max() * 1.18)]))
        st.plotly_chart(fig, use_container_width=True)
        pv = int(ev_say.get("page_view", 0))
        cart = int(ev_say.get("add_to_cart", 0))
        cart_ratio = (cart / pv) if pv else 0
        st.caption(
            f"📌 Sayfa görüntüleme ({pv:,}) açık ara en sık eylem. "
            f"Sepete ekleme oranı %{cart_ratio * 100:.1f}; kullanıcıların önemli bir kısmı "
            "ürünü görüyor ancak sepete ekleme adımına geçmiyor. Asıl darboğaz burada."
        )
    with b2:
        try:
            events_tmp = pd.read_csv("data/events.csv", parse_dates=["timestamp"])
            events_tmp["ay"] = events_tmp["timestamp"].dt.to_period("M").astype(str)
            aylik = events_tmp.groupby("ay").size().reset_index(name="n").tail(18)
            if aylik.empty:
                st.info("Aylık trend grafiği için yeterli tarih bilgisi bulunamadı.")
            else:
                fig = go.Figure(go.Scatter(x=aylik["ay"], y=aylik["n"],
                    mode="lines+markers", line=dict(color=TURUNCU, width=2.5),
                    marker=dict(size=5, color=TURUNCU),
                    fill="tozeroy", fillcolor="rgba(249,115,22,0.07)"))
                tema(fig, h=300, baslik="Aylık Etkileşim Trendi (Son 18 Ay)",
                     xaxis=dict(**EKSEN, title="", tickangle=-30),
                     yaxis=dict(**EKSEN, title="Etkileşim Sayısı"))
                st.plotly_chart(fig, use_container_width=True)
                st.caption("📌 Son 18 aylık hareket, trafik hacminin dönemsel olarak nasıl değiştiğini gösterir. Bu grafik kampanya, sezon ve veri kesim dönemi etkisini okumak için kullanılır.")
        except Exception as hata:
            st.warning(f"Aylık trend grafiği oluşturulamadı: {hata}")

    c1, c2 = st.columns(2)
    with c1:
        urun_tmp = pd.read_csv("data/products.csv")
        fig = px.histogram(urun_tmp, x="price_usd", nbins=25, color_discrete_sequence=[MAVI])
        fig.add_vline(x=urun_tmp["price_usd"].median(), line_dash="dash",
                      line_color=TURUNCU, line_width=2,
                      annotation_text=f"Medyan: ${urun_tmp['price_usd'].median():.0f}",
                      annotation_font_color=TURUNCU)
        tema(fig, h=280, baslik="Ürün Fiyat Dağılımı",
             xaxis=dict(**EKSEN, title="Fiyat (USD)"),
             yaxis=dict(**EKSEN, title="Ürün Sayısı"))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("📌 Fiyatlar 3.50 USD ile 596.62 USD arasında. Dağılım sağa çarpık: ürünlerin büyük çoğunluğu 100 USD'nin altında, medyan 77 USD. Düşük fiyatlı ürünler daha geniş kitleye hitap ediyor.")
    with c2:
        rev_tmp = pd.read_csv("data/reviews.csv")
        rat_say = rev_tmp["rating"].value_counts().sort_index()
        clrs = [KIRMIZI, "#F97316", SARI, MAVI, YESIL]
        fig = go.Figure(go.Bar(
            x=rat_say.index, y=rat_say.values,
            marker_color=clrs[:len(rat_say)],
            text=[f"{v:,}" for v in rat_say.values],
            textposition="outside",
            textfont=dict(color="#334155", size=11),
        ))
        # Barların üstündeki sayıların kesilmemesi için y range geniş
        tema(fig, h=280, baslik="Müşteri Rating Dağılımı",
             xaxis=dict(**EKSEN, title="Puan (1–5)",
                        tickvals=[1,2,3,4,5], ticktext=["1★","2★","3★","4★","5★"]),
             yaxis=dict(**EKSEN, title="Yorum Sayısı", range=[0, 4700]))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("📌 Yorumların %70.7'si 4–5 yıldız — genel müşteri memnuniyeti yüksek. Yalnızca %10.9 negatif yorum var; bu duygu analizi modelimizi güçlü bir pozitif sinyal temeline oturtuyor.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('''<div class="sec-head">🔗 Tablolar Nasıl Birleştirildi?</div>''', unsafe_allow_html=True)
    st.markdown("""<div class="card">
<div style="font-size:0.85rem;font-weight:800;color:#1E293B;margin-bottom:6px">
7 ayrı tablo tek bir analiz hikayesinde birleştirildi
</div>
<div style="font-size:0.76rem;color:#64748B;line-height:1.6">
Bu bölümde her tablo aynı anda birbirine eklenmedi. Önce müşteri davranışı, gerçek satış ve yorum bilgisi ayrı ayrı anlamlandırıldı; sonra model için gerekli seviyede birleştirildi.
</div>
<div class="merge-grid">
  <div class="merge-card blue">
    <div class="merge-title">Müşteri davranışı</div>
    <div class="merge-path"><code>events</code> → <code>sessions</code> → <code>customers</code></div>
    <div class="merge-desc">Events tablosunda müşteri kimliği doğrudan yoktur. Bu yüzden oturum bilgisi üzerinden müşteriye ulaşılır.</div>
  </div>
  <div class="merge-card green">
    <div class="merge-title">Gerçek satış verisi</div>
    <div class="merge-path"><code>orders</code> → <code>order_items</code> → <code>products</code></div>
    <div class="merge-desc">Satılan ürün adedi, ürün bazlı gelir ve sipariş kalemleri bu bağlantıdan hesaplanır.</div>
  </div>
  <div class="merge-card orange">
    <div class="merge-title">Yorum ve memnuniyet</div>
    <div class="merge-path"><code>reviews</code> → <code>products</code></div>
    <div class="merge-desc">Rating ortalaması, yorum sayısı ve duygu analizi sonucu ürün seviyesine eklenir.</div>
  </div>
</div>
	<div class="critical-note">
	<strong>Kritik not:</strong> Checkout ve purchase eventlerinde ürün bilgisi beklenen şekilde boş olabilir. Bu nedenle ürün bazlı satış adedi events tablosundan değil, <strong>order_items.quantity</strong> alanından hesaplanır. Funnel ise event sayılarıyla oturum davranışını gösterir.
	</div>
	</div>""", unsafe_allow_html=True)

# =========================================================================
# SAYFA 3 — ANALİZ
# =========================================================================
elif sayfa == "Analiz":
    st.markdown('''<div class="sec-head">📈 Detaylı Ürün Analizi</div>''', unsafe_allow_html=True)

    k1, k2 = st.columns(2)
    with k1:
        fig = px.scatter(urun_df, x="price_usd", y="satis_orani",
                         color="oneri_skoru", size="satis_adedi", size_max=20,
                         hover_name="name",
                         color_continuous_scale=[[0,"#DBEAFE"],[1,LACIVERT]],
                         labels={"price_usd":"Fiyat","satis_orani":"Satış Oranı"})
        tema(fig, h=320, baslik="Fiyat vs Satış Oranı",
             xaxis=dict(**EKSEN,title="Fiyat (USD)"),
             yaxis=dict(**EKSEN,title="Satış Oranı"))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("📌 Düşük ve orta fiyatlı ürünlerde satış oranı daha yoğun görünüyor. Bu grafik fiyat arttıkça dönüşüm davranışının nasıl değiştiğini okumak için kullanılır.")
    with k2:
        fig = px.scatter(urun_df, x="ort_rating", y="oneri_skoru",
                         color="gelir", size="yorum_sayisi", size_max=20,
                         hover_name="name",
                         color_continuous_scale=[[0,"#FED7AA"],[1,TURUNCU]],
                         labels={"ort_rating":"Rating","oneri_skoru":"Öneri Skoru"})
        tema(fig, h=320, baslik="Rating vs Öneri Skoru",
             xaxis=dict(**EKSEN,title="Rating"),
             yaxis=dict(**EKSEN,title="Öneri Skoru"))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("📌 Rating tek başına yeterli değildir; yüksek öneri skoru için satış, duygu, sepet ve kârlılık sinyalleri birlikte değerlendirilir.")

    k3, k4 = st.columns(2)
    with k3:
        fig = px.scatter(urun_df, x="page_view", y="satis_adedi",
                         color="oneri_skoru", hover_name="name",
                         color_continuous_scale=[[0,"#DBEAFE"],[1,LACIVERT]],
                         labels={"page_view":"Görüntüleme","satis_adedi":"Satış"})
        tema(fig, h=300, baslik="Görüntüleme vs Satış",
             xaxis=dict(**EKSEN,title="Sayfa Görüntüleme"),
             yaxis=dict(**EKSEN,title="Satış Adedi"))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("📌 Çok görüntülenen ürünlerin tamamı yüksek satış üretmiyor. Bu ayrım, yalnızca trafik değil dönüşüm kalitesinin de takip edilmesi gerektiğini gösterir.")
    with k4:
        fig = px.scatter(urun_df, x="sepet_orani", y="satis_orani",
                         color="oneri_skoru", hover_name="name",
                         color_continuous_scale=[[0,"#FED7AA"],[1,TURUNCU]],
                         labels={"sepet_orani":"Sepet Oranı","satis_orani":"Satış Oranı"})
        tema(fig, h=300, baslik="Sepete Ekleme Oranı vs Satış Oranı",
             xaxis=dict(**EKSEN,title="Sepete Ekleme Oranı"),
             yaxis=dict(**EKSEN,title="Satış Oranı"))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("📌 Sepete ekleme oranı yükseldikçe satış oranı da çoğunlukla artıyor. Modelde bu değişkenin güçlü çıkması dönüşüm hunisi bulgusuyla uyumlu.")

# =========================================================================
# SAYFA 4 — ÖNERİLER
# =========================================================================
elif sayfa == "Genel Öneriler":
    st.markdown('''<div class="sec-head">🏆 Genel Ürün Önerileri <span class="sec-tag">Top 50</span></div>''',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
      <strong>Bu bölüm kişiye özel değildir.</strong><br>
      Ürünler; satış oranı, yorum kalitesi, rating, kârlılık ve öneri skoru gibi genel performans
      sinyallerine göre sıralanır. Müşteri sekmesindeki öneri motoru ise belirli bir müşteri için
      satın alma olasılığı üretir.
    </div>
    """, unsafe_allow_html=True)

    f1, f2, f3 = st.columns([2,2,1])
    with f1: n = st.slider("Kaç ürün?",5,50,15)
    with f2:
        katlar = ["Tümü"] + sorted(top50_df["category"].dropna().unique().tolist())
        sec_k = st.selectbox("Kategori",katlar)
    with f3:
        sira = st.selectbox("Sıralama",["Skor ↓","Rating ↓","Satış % ↓"])

    fil = top50_df.copy()
    if sec_k != "Tümü": fil = fil[fil["category"]==sec_k]
    siralama_k = {"Skor ↓":"oneri_skoru","Rating ↓":"ort_rating","Satış % ↓":"satis_orani"}
    grafik_baslik = {"Skor ↓":"Öneri Skoru","Rating ↓":"Rating","Satış % ↓":"Satış Oranı"}
    grafik_format = {"Skor ↓":lambda x: f"  {x:.3f}",
                     "Rating ↓":lambda x: f"  {x:.2f}",
                     "Satış % ↓":lambda x: f"  {x:.1%}"}
    fil = fil.sort_values(
        [siralama_k[sira], "oneri_skoru", "name"],
        ascending=[False, False, True]
    ).head(n).copy()
    fil["grafik_urun"] = fil["name"].str[:42]
    grafik_sirasi = fil["grafik_urun"].tolist()[::-1]
    grafik_deger = siralama_k[sira]

    fig = go.Figure(go.Bar(
        y=fil["grafik_urun"], x=fil[grafik_deger], orientation="h",
        marker=dict(color=fil[grafik_deger],
                    colorscale=[[0,"#FFF7ED"],[0.5,"#FED7AA"],[1,TURUNCU]],
                    showscale=True,
                    colorbar=dict(title=grafik_baslik[sira],thickness=10,
                                  tickfont=dict(color="#64748B",size=9))),
        text=fil[grafik_deger].apply(grafik_format[sira]),
        textposition="outside",textfont=dict(color="#334155",size=10),
    ))
    tema(fig, h=max(260,len(fil)*26), xaxis=dict(**EKSEN,title=grafik_baslik[sira]))
    fig.update_yaxes(categoryorder="array", categoryarray=grafik_sirasi, **EKSEN)
    st.plotly_chart(fig, use_container_width=True)

    tbl = fil[["name","category","price_usd","ort_rating",
               "pozitif_oran","satis_orani","oneri_skoru"]].copy()
    tbl.columns = ["Ürün","Kategori","Fiyat","Rating","Pozitif %","Satış %","Skor"]
    tbl["Fiyat"] = tbl["Fiyat"].apply(lambda x: f"${x:.2f}")
    tbl["Rating"] = tbl["Rating"].apply(lambda x: f"⭐ {x:.2f}")
    tbl["Pozitif %"] = tbl["Pozitif %"].apply(lambda x: f"{x:.1%}")
    tbl["Satış %"] = tbl["Satış %"].apply(lambda x: f"{x:.1%}")
    tbl["Skor"] = tbl["Skor"].apply(lambda x: f"{x:.4f}")
    st.dataframe(tbl, use_container_width=True, hide_index=True, height=320)

# =========================================================================
# SAYFA 5 — MODEL
# =========================================================================
elif sayfa == "Model":
    st.markdown('''<div class="sec-head">🤖 Model Değerlendirmesi <span class="sec-tag">Sınıflandırma</span></div>''',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
      <strong>Bu model şu soruyu yanıtlar: "Bu ürün müşterilere önerilmeli mi?"</strong><br><br>
      Cevap iki sınıftan biridir: <strong>Evet (önerilir)</strong> veya <strong>Hayır (önerilmez)</strong>.
      Bu nedenle problem bir regresyon değil, <em>ikili sınıflandırma</em> problemidir.<br><br>
      Kullanılan algoritma <strong>Random Forest Classifier</strong>'dır. Modelde 200 karar ağacı kullanılır;
      her ağaç ayrı karar verir, sonuç çoğunluk oyu ile belirlenir. Bu yapı tek karar ağacına göre daha kararlı sonuç üretir.<br><br>
      <strong>Neden "satış oranı" modele dahil edilmedi?</strong> Çünkü hedef etiket, satış oranının da içinde bulunduğu
      ağırlıklı <strong>öneri skoru</strong> üzerinden oluşturuldu. Satış oranını ayrıca modele vermek,
      modele cevabın bir parçasını göstermek olurdu. Bu durum <em>veri sızıntısı</em> yaratır ve test sonuçlarını olduğundan iyi gösterebilir.
    </div>
    """, unsafe_allow_html=True)

    try:
        cm      = np.load("outputs/cm.npy")
        fpr     = np.load("outputs/roc_fpr.npy")
        tpr     = np.load("outputs/roc_tpr.npy")
        fi      = pd.read_csv("outputs/ozellik_onemi.csv")
        met     = pd.read_csv("outputs/model_met.csv")
        auc_val = float(met["auc"].iloc[0])
        dogr    = float(met["dogruluk"].iloc[0])
    except FileNotFoundError:
        st.warning("Model çıktısı yok. `python pipeline.py` çalıştırın.")
        st.stop()

    tn,fp,fn,tp = cm.ravel()
    prec = tp/(tp+fp) if (tp+fp)>0 else 0
    rec  = tp/(tp+fn) if (tp+fn)>0 else 0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0

    m1,m2,m3,m4 = st.columns(4)
    for kol,l,v,s,cls in [
        (m1,"Doğruluk",f"{dogr:.1%}","Test seti","green"),
        (m2,"ROC AUC",f"{auc_val:.3f}","Mükemmel ayrım","orange"),
        (m3,"Precision",f"{prec:.1%}","Önerilir sınıfı","blue"),
        (m4,"F1 Skoru",f"{f1:.3f}","Denge metriği","amber"),
    ]:
        kol.markdown(f'''
        <div class="kpi {cls}">
          <div class="kpi-label">{l}</div>
          <div class="kpi-num">{v}</div>
          <div class="kpi-badge neu">{s}</div>
        </div>''', unsafe_allow_html=True)

    st.markdown("<br>",unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Heatmap(
            z=[[int(tn),int(fp)],[int(fn),int(tp)]],
            x=["Önerilmez","Önerilir"],y=["Önerilmez","Önerilir"],
            text=[[f"TN\n{tn}",f"FP\n{fp}"],[f"FN\n{fn}",f"TP\n{tp}"]],
            texttemplate="%{text}",textfont=dict(size=14,color="white"),
            colorscale=[[0,"#DBEAFE"],[0.5,MAVI],[1,LACIVERT]],showscale=False,
        ))
        tema(fig, h=290, baslik="Confusion Matrix",
             xaxis=dict(**EKSEN,title="Tahmin"),
             yaxis=dict(**EKSEN,title="Gerçek",autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("📌 Confusion Matrix modelin doğru ve yanlış kararlarını gösterir. Bu testte model 185 ürünü doğru şekilde önerilmez, 69 ürünü doğru şekilde önerilir olarak sınıflandırmıştır; 25 yanlış pozitif ve 21 yanlış negatif karar vardır.")
    with c2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr,y=tpr,mode="lines",
            line=dict(color=TURUNCU,width=2.5),
            name=f"ROC (AUC={auc_val:.3f})",
            fill="tozeroy",fillcolor="rgba(249,115,22,0.07)"))
        fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",
            line=dict(color="#CBD5E1",width=1.5,dash="dash"),name="Rastgele"))
        tema(fig, h=290, baslik="ROC Eğrisi",
             xaxis=dict(**EKSEN,title="Yanlış Pozitif"),
             yaxis=dict(**EKSEN,title="Doğru Pozitif"),
             showlegend=True,
             legend=dict(x=0.55,y=0.08,bgcolor="rgba(255,255,255,0.95)",
                         bordercolor="#E2E8F0",borderwidth=1,
                         font=dict(color="#334155",size=10)))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("📌 ROC eğrisi modelin önerilir ve önerilmez sınıflarını ayırma gücünü gösterir. AUC değerinin 0.913 olması, modelin rastgele tahminden çok daha güçlü bir ayrım yaptığını gösterir.")

    st.markdown('''<div class="sec-head" style="margin-top:8px">🔍 Özellik Önemi</div>''',
                unsafe_allow_html=True)
    fi["ad"] = fi["ozellik"].map(OZELLIK_AD).fillna(fi["ozellik"])
    fi_s = fi.sort_values("onem")
    rk_fi = [TURUNCU if i==len(fi_s)-1 else MAVI for i in range(len(fi_s))]
    fig = go.Figure(go.Bar(y=fi_s["ad"],x=fi_s["onem"],orientation="h",
        marker_color=rk_fi,
        text=fi_s["onem"].apply(lambda x: f"  %{x*100:.1f}"),
        textposition="outside",textfont=dict(color="#334155",size=11)))
    tema(fig, h=295, xaxis=dict(**EKSEN,title="Göreceli Önem",range=[0,0.28]),yaxis=dict(**EKSEN))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("📌 Sepete ekleme oranı (%21.7) açık ara en belirleyici faktör — dönüşüm hunisi bulgusunu doğruluyor. Duygu skoru üçüncü sırada: NLP analizinin modele somut katkısı var.")

# =========================================================================
# SAYFA 6 — DUYGU
# =========================================================================
elif sayfa == "Duygu":
    st.markdown('''<div class="sec-head">💬 Duygu Analizi <span class="sec-tag">VADER + Rating Hibrit</span></div>''',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
      <strong>Yöntem: VADER + Rating Hibrit</strong> — VADER sözlük tabanlı duygu analizi yapar.
      Ancak bu veri setinde "Okay overall" (compound=0.226) VADER'ı pozitif sayar, rating=3 olduğundan aslında nötrdür.
      Hibrit kural: <em>compound &gt; 0.3 VE rating ≥ 4 → pozitif, compound &lt; -0.1 VEYA rating ≤ 2 → negatif, diğerleri → nötr</em>
    </div>
    """, unsafe_allow_html=True)

    toplam=len(duygu_df); pos=(duygu_df["duygu"]=="positive").sum()
    neg=(duygu_df["duygu"]=="negative").sum(); not_=(duygu_df["duygu"]=="neutral").sum()

    d1,d2,d3 = st.columns(3)
    for kol,l,v,s,cls in [
        (d1,"😊 Pozitif",f"{pos:,}",f"↗ %{pos/toplam*100:.1f}","green"),
        (d2,"😐 Nötr",f"{not_:,}",f"%{not_/toplam*100:.1f}","blue"),
        (d3,"😠 Negatif",f"{neg:,}",f"↘ %{neg/toplam*100:.1f}","red"),
    ]:
        kol.markdown(f'''
        <div class="kpi {cls}">
          <div class="kpi-label">{l}</div>
          <div class="kpi-num">{v}</div>
          <div class="kpi-badge {"up" if "↗" in s else "down" if "↘" in s else "neu"}">{s}</div>
        </div>''', unsafe_allow_html=True)

    st.markdown("<br>",unsafe_allow_html=True)
    dk2, hk2 = st.columns(2)
    with dk2:
        sayim=duygu_df["duygu"].value_counts()
        fig=go.Figure(go.Pie(labels=[l.capitalize() for l in sayim.index],
            values=sayim.values,hole=0.56,
            marker=dict(colors=[DUYGU_RENK.get(l,"#999") for l in sayim.index],
                        line=dict(color="white",width=3))))
        tema(fig,h=260,baslik="Duygu Dağılımı",showlegend=True,
             legend=dict(orientation="h",y=-0.12,bgcolor="rgba(0,0,0,0)",
                         font=dict(color="#334155",size=10)))
        st.plotly_chart(fig,use_container_width=True)
        st.caption("📌 Yorumların büyük kısmı pozitif sınıfta toplanıyor. Bu durum ürün memnuniyetinin genel olarak güçlü olduğunu ve duygu skorunun öneri sistemine anlamlı katkı verdiğini gösterir.")
    with hk2:
        fig=px.histogram(duygu_df,x="guven",nbins=20,color="duygu",
                         color_discrete_map=DUYGU_RENK,barmode="overlay",opacity=0.8,
                         labels={"guven":"Güven Skoru","duygu":"Duygu"})
        tema(fig,h=260,baslik="Güven Skoru Dağılımı",
             xaxis=dict(**EKSEN,title="Güven Skoru"),
             yaxis=dict(**EKSEN,title="Yorum Sayısı"))
        st.plotly_chart(fig,use_container_width=True)
        st.caption("📌 Güven skorları duygu kararının ne kadar güçlü verildiğini gösterir. Yüksek güven, yorum metni ve rating bilgisinin aynı yönde sinyal verdiği durumları temsil eder.")

    st.markdown('''<div class="sec-head" style="margin-top:8px">⭐ Yıldız Dağılımı</div>''',
                unsafe_allow_html=True)
    yildiz=duygu_df["yildiz"].value_counts().sort_index()
    fig=px.bar(x=yildiz.index,y=yildiz.values,
        color=yildiz.values.tolist(),
        color_continuous_scale=[[0,KIRMIZI],[0.5,SARI],[1,YESIL]],
        text=[f"{v:,}" for v in yildiz.values],labels={"x":"Yıldız","y":"Yorum"})
    fig.update_traces(textposition="outside",textfont=dict(color="#334155",size=12))
    fig.update_coloraxes(showscale=False)
    tema(fig,h=240,xaxis=dict(**EKSEN,tickvals=[1,2,3,4,5],
        ticktext=["1★","2★★","3★★★","4★★★★","5★★★★★"]),
        yaxis=dict(**EKSEN,title="Yorum Sayısı"))
    st.plotly_chart(fig,use_container_width=True)
    st.caption("📌 4 ve 5 yıldızlı yorumların baskın olması müşteri memnuniyetinin yüksek olduğunu destekliyor. Düşük yıldızlı yorumlar ise iyileştirme yapılabilecek ürünleri yakalamak için kullanılır.")

    st.markdown('''<div class="sec-head" style="margin-top:8px">💬 Örnek Yorumlar</div>''',
                unsafe_allow_html=True)
    sec=st.selectbox("Duygu türü",["positive","negative","neutral"],
        format_func=lambda x:{"positive":"😊 Pozitif","negative":"😠 Negatif","neutral":"😐 Nötr"}[x])
    for _,r in duygu_df[duygu_df["duygu"]==sec].head(6).iterrows():
        c=DUYGU_RENK[sec]
        st.markdown(f'''
        <div class="yorum-card" style="border-left:3px solid {c}">
          <div class="yorum-text">{str(r["review_text"])[:220]}</div>
          <div class="yorum-meta">
            Güven: <span style="color:{c};font-weight:700">{r["guven"]:.1%}</span>
            &nbsp;·&nbsp; VADER: {r["compound"]:.3f}
            &nbsp;·&nbsp; Rating: {r["rating"]}★
          </div>
        </div>''', unsafe_allow_html=True)

# =========================================================================
# SAYFA 7 — MÜŞTERİ
# =========================================================================
elif sayfa == "Müşteri":
    st.markdown('''<div class="sec-head">👤 Müşteri Segmentasyonu & Kişisel Öneri</div>''',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
      <strong>RFM analizi yalnızca en az bir siparişi olan müşteriler için hesaplanmıştır.</strong><br>
      Veri setinde <strong>20.000 kayıtlı müşteri</strong> bulunurken, <strong>16.268 müşterinin</strong>
      sipariş geçmişi vardır. Sipariş geçmişi olmayan <strong>3.732 müşteri</strong> segmentasyona
      dahil edilmemiştir.
    </div>
    """, unsafe_allow_html=True)

    sc=rfm_df["segment"].value_counts()
    m1,m2,m3,m4=st.columns(4)
    for kol,l,v,s,cls in [
        (m1,"🏆 Şampiyon",f"{int(sc.get('Şampiyon',0)):,}","En değerliler","orange"),
        (m2,"💚 Sadık",f"{int(sc.get('Sadık',0)):,}","Düzenli alışveriş","green"),
        (m3,"⚠ Risk",f"{int(sc.get('Risk Altında',0)):,}","Geri kazan","amber"),
        (m4,"💔 Kayıp",f"{int(sc.get('Kayıp',0)):,}","Uzun süre yok","red"),
    ]:
        kol.markdown(f'''
        <div class="kpi {cls}">
          <div class="kpi-label">{l}</div>
          <div class="kpi-num">{v}</div>
          <div class="kpi-badge neu">{s}</div>
        </div>''', unsafe_allow_html=True)

    st.markdown("<br>",unsafe_allow_html=True)
    r1,r2=st.columns(2)
    with r1:
        fig=go.Figure(go.Bar(x=sc.values,y=sc.index,orientation="h",
            marker_color=[SEG_RENK.get(s,"#999") for s in sc.index],
            text=[f" {v:,}" for v in sc.values],
            textposition="outside",textfont=dict(color="#334155",size=10)))
        tema(fig,h=290,baslik="RFM Segment Dağılımı",
             xaxis=dict(**EKSEN,title="Müşteri Sayısı",range=[0,5200]),
             yaxis=dict(**EKSEN,autorange="reversed"))
        st.plotly_chart(fig,use_container_width=True)
        st.caption("📌 Sadık segment en kalabalık (4.388), ancak Şampiyon grubu (2.923) kişi başına en yüksek harcama değerine sahip. Risk Altında grubu ise geri kazanılabilecek aktif gelir kaynağı.")
    with r2:
        ornek=rfm_df.sample(min(650,len(rfm_df)),random_state=42)
        fig=px.scatter(ornek,x="recency",y="monetary",color="segment",
            size="frequency",size_max=10,opacity=0.58,color_discrete_map=SEG_RENK,
            labels={"recency":"Recency (gün)","monetary":"Harcama (USD)","segment":"Segment"},
            hover_data={"frequency":True})
        fig.update_traces(marker=dict(line=dict(width=0.35,color="white")))
        tema(fig,h=290,baslik="RFM Dağılım Haritası (Örneklem)",
             xaxis=dict(**EKSEN,title="Recency (gün)"),
             yaxis=dict(**EKSEN,title="Monetary (USD)"),showlegend=True,
             legend=dict(font=dict(color="#334155",size=9),
                          bgcolor="rgba(255,255,255,0.95)",
                          bordercolor="#E2E8F0",borderwidth=1))
        st.plotly_chart(fig,use_container_width=True)
        st.caption("📌 Noktaların iç içe görünmesi normaldir; RFM bir kümeleme algoritması değil, recency-frequency-monetary skorlarına dayalı kural tabanlı segmentlemedir. Grafikte renkler segmenti, nokta büyüklüğü sipariş sıklığını gösterir.")

    seg_o=(rfm_df.groupby("segment")
           .agg(Musteri=("customer_id","count"),
                Ort_Recency=("recency","mean"),
                Ort_Siparis=("frequency","mean"),
                Ort_Harcama=("monetary","mean"),
                Toplam=("monetary","sum"))
           .round(1).reset_index().sort_values("Toplam",ascending=False))
    seg_o["Ort_Harcama"]=seg_o["Ort_Harcama"].apply(lambda x:f"${x:,.0f}")
    seg_o["Toplam"]=seg_o["Toplam"].apply(lambda x:f"${x:,.0f}")
    st.dataframe(seg_o,use_container_width=True,hide_index=True)

    st.markdown("<hr>",unsafe_allow_html=True)
    st.markdown('''<div class="sec-head">🎯 Kişiye Özel Satın Alma Tahmini</div>''',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
      <strong>Nasıl Çalışır?</strong><br>
      Seçilen müşteri için her aday ürün ayrı ayrı değerlendirilir.
      Model, <em>müşteri + ürün</em> özelliklerinden satın alma olasılığı üretir.
      Daha önce satın alınan ürünler çıkarılır ve en yüksek olasılıklı ürünler önerilir.
      Bu bölüm, Genel Öneriler sekmesinden farklı olarak kişiye özeldir.
    </div>
    """, unsafe_allow_html=True)

    ark, n_kol, b_kol = st.columns([4, 1, 0.9])
    with ark:
        girdi = st.text_input(
            "Müşteri adı veya ID",
            placeholder="Örn: Jennifer Salinas veya 1",
            key="musteri_ara"
        )
    with n_kol:
        n_on = st.selectbox("Öneri sayısı", [3, 5, 8, 10], index=1)
    with b_kol:
        st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)
        ara_btn = st.button("🔍 Öner", use_container_width=True)

    # Enter tuşu VEYA buton tıklaması ile çalışır
    if girdi:
        try: cid=int(girdi); satir=musteri_df[musteri_df["customer_id"]==cid]
        except: satir=musteri_df[musteri_df["name"].str.lower().str.contains(girdi.lower(),na=False)]
        if satir.empty: st.warning("Bulunamadı.")
        else:
            m=satir.iloc[0]; cid=int(m["customer_id"])
            rfm_s=rfm_df[rfm_df["customer_id"]==cid]
            seg=rfm_s["segment"].iloc[0] if not rfm_s.empty else "?"
            rec=int(rfm_s["recency"].iloc[0]) if not rfm_s.empty else 0
            freq=int(rfm_s["frequency"].iloc[0]) if not rfm_s.empty else 0
            mon=float(rfm_s["monetary"].iloc[0]) if not rfm_s.empty else 0
            sc2=SEG_RENK.get(seg,"#64748B")
            st.markdown(f'''
            <div style="display:flex;gap:10px;margin:14px 0">
              <div class="pill" style="flex:2;border-left:3px solid {sc2}">
                <div class="pill-label">Müşteri</div>
                <div class="pill-val">{m["name"]}</div>
                <div class="pill-sub">ID:{cid} · {m.get("country","?")} · Yaş:{int(m.get("age",0))}</div>
              </div>
              <div class="pill" style="flex:1;border-left:3px solid {sc2}">
                <div class="pill-label">Segment</div>
                <div class="pill-val" style="color:{sc2}">{seg}</div>
              </div>
              <div class="pill" style="flex:1;border-left:3px solid #94A3B8">
                <div class="pill-label">Son Alışveriş</div>
                <div class="pill-val">{rec} gün</div>
              </div>
              <div class="pill" style="flex:1;border-left:3px solid #94A3B8">
                <div class="pill-label">Harcama</div>
                <div class="pill-val">${mon:,.0f}</div>
              </div>
            </div>''', unsafe_allow_html=True)

            tercih=mkat_df[mkat_df["customer_id"]==cid].sort_values("quantity",ascending=False)
            top_kat=tercih["category"].tolist()[:2]
            if not top_kat: st.info("Satın alma geçmişi yok.")
            else:
                oner = kisisel_satin_alma_tahmini(cid, adet=n_on)
                if oner.empty:
                    st.warning("Kişiye özel satın alma modeli bulunamadı. Önce `python pipeline.py` çalıştırın.")
                    st.stop()
                pk2,ok2=st.columns([1,3])
                with pk2:
                    st.markdown('<p style="font-size:0.72rem;color:#94A3B8;font-weight:700;text-transform:uppercase">Geçmiş Kategori Satın Alımları</p>', unsafe_allow_html=True)
                    tot_a=tercih["quantity"].sum()
                    for _,tr in tercih.head(5).iterrows():
                        p2=tr["quantity"]/tot_a*100
                        st.markdown(f'''
                        <div style="margin-bottom:8px">
                          <div style="display:flex;justify-content:space-between;margin-bottom:3px">
                            <span style="font-size:0.78rem;color:#334155">{tr["category"]}</span>
                            <span style="font-size:0.68rem;color:#94A3B8">{int(tr["quantity"])} adet</span>
                          </div>
                          <div style="background:#F1F5F9;border-radius:3px;height:4px">
                            <div style="background:{TURUNCU};width:{p2:.0f}%;height:4px;border-radius:3px"></div>
                          </div>
                        </div>''', unsafe_allow_html=True)
                with ok2:
                    baslik = "Kişiye Özel Satın Alma Tahmini"
                    if not kisisel_met_df.empty:
                        baslik += f" · AUC {float(kisisel_met_df['auc'].iloc[0]):.3f}"
                    st.markdown(f'<p style="font-size:0.72rem;color:#94A3B8;font-weight:700;text-transform:uppercase">{baslik}</p>', unsafe_allow_html=True)
                    for sira,(_, u) in enumerate(oner.iterrows(),1):
                        ol = float(u.get("satin_alma_olasiligi",0))
                        sp=int(min(1, ol)*100)
                        st.markdown(f'''
                        <div class="yorum-card" style="margin-bottom:8px;border-left:3px solid {TURUNCU}">
                          <div style="display:flex;align-items:center;justify-content:space-between">
                            <div><span style="font-size:0.68rem;color:#94A3B8">#{sira}</span>
                                 <span style="font-size:0.88rem;color:#1E293B;font-weight:600;margin-left:8px">{u["name"]}</span></div>
                            <span style="font-size:0.75rem;color:{TURUNCU};font-weight:700">%{ol*100:.1f}</span>
                          </div>
                          <div style="margin-top:8px;display:flex;align-items:center;gap:12px">
                            <span style="font-size:0.72rem;color:#64748B">{u["category"]}</span>
                            <span style="font-size:0.72rem;color:#64748B">⭐ {u.get("ort_rating",0):.2f}</span>
                            <span style="font-size:0.72rem;color:#64748B">Skor {u.get("oneri_skoru",0):.3f}</span>
                            <div style="flex:1;background:#F1F5F9;border-radius:3px;height:4px;max-width:80px">
                              <div style="background:{TURUNCU};width:{sp}%;height:4px;border-radius:3px"></div>
                            </div>
                            <span style="font-size:0.7rem;color:{TURUNCU};font-weight:700">Satın alma olasılığı</span>
                          </div>
                        </div>''', unsafe_allow_html=True)
    else:
        sec_seg=st.selectbox("Segment seç",list(SEG_RENK.keys()))
        liste=(rfm_df[rfm_df["segment"]==sec_seg].sort_values("monetary",ascending=False).head(10))
        mev=[c for c in ["customer_id","name","country","age","recency","frequency","monetary","segment"] if c in liste.columns]
        g=liste[mev].copy()
        if "monetary" in g.columns: g["monetary"]=g["monetary"].apply(lambda x:f"${x:,.0f}")
        g.columns=[{"customer_id":"ID","name":"Ad","country":"Ülke","age":"Yaş",
                     "recency":"Recency","frequency":"Sipariş","monetary":"Harcama",
                     "segment":"Segment"}.get(c,c) for c in g.columns]
        st.dataframe(g,use_container_width=True,hide_index=True)

# =========================================================================
# SAYFA 8 — İSTATİSTİK
# =========================================================================
elif sayfa == "İstatistik":
    st.markdown('''<div class="sec-head">⚙️ Sistem İstatistikleri</div>''',
                unsafe_allow_html=True)

    kat=(urun_df.groupby("category")
         .agg(urun=("product_id","count"),ort_sk=("oneri_skoru","mean"),
              ort_rat=("ort_rating","mean"),satis=("satis_adedi","sum"),
              gelir=("gelir","sum"))
         .round(3).reset_index())

    bk,rk2=st.columns(2)
    with bk:
        fig=px.bar(kat.sort_values("ort_sk"),x="ort_sk",y="category",orientation="h",
            color="ort_sk",
            color_continuous_scale=[[0,"#DBEAFE"],[0.5,MAVI],[1,LACIVERT]],
            text=kat.sort_values("ort_sk")["ort_sk"].apply(lambda x:f"{x:.3f}"),
            labels={"ort_sk":"Ort.Skor","category":""})
        fig.update_traces(textposition="outside",textfont=dict(color="#334155",size=10))
        fig.update_coloraxes(showscale=False)
        tema(fig,h=320,baslik="Kategori Başına Ort. Skor",
             xaxis=dict(**EKSEN,title="Ortalama Skor"),yaxis=dict(**EKSEN))
        st.plotly_chart(fig,use_container_width=True)
        st.caption("📌 Kategori ortalama skorları, hangi ürün gruplarının genel öneri performansında öne çıktığını gösterir. Skoru düşük kategoriler fiyat, rating veya dönüşüm sinyalleri açısından daha zayıf kalmış olabilir.")

    with rk2:
        # Radar — title_font sadece burada, TEMEL dict içinde değil
        rd=kat.copy()
        sc3=MinMaxScaler()
        rd[["ort_sk","ort_rat","gelir"]]=sc3.fit_transform(rd[["ort_sk","ort_rat","gelir"]])
        pal=[TURUNCU,YESIL,SARI,KIRMIZI,MAVI,"#8B5CF6","#EC4899"]
        fig=go.Figure()
        for i,(_,s) in enumerate(rd.iterrows()):
            fig.add_trace(go.Scatterpolar(
                r=[s["ort_sk"],s["ort_rat"],s["gelir"],s["ort_sk"]],
                theta=["Skor","Rating","Gelir","Skor"],
                fill="toself",opacity=0.3,name=s["category"],
                line=dict(color=pal[i%len(pal)],width=1.8)))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", height=320,
            margin=dict(l=10,r=10,t=42,b=10),
            title="Kategori Karşılaştırma Radarı",
            title_font=dict(family="Inter",color="#1E293B",size=12),
            polar=dict(bgcolor="white",
                       radialaxis=dict(visible=True,color="#E2E8F0",
                                       tickfont=dict(color="#94A3B8")),
                       angularaxis=dict(color="#E2E8F0",
                                        tickfont=dict(color="#334155"))),
            legend=dict(font=dict(color="#334155",size=9),
                        bgcolor="rgba(255,255,255,0.95)",
                        bordercolor="#E2E8F0",borderwidth=1),
        )
        st.plotly_chart(fig,use_container_width=True)
        st.caption("📌 Radar grafik kategorileri aynı ölçeğe getirerek karşılaştırır. Skor, rating ve gelir dengesi birlikte yüksek olan kategoriler daha güçlü ticari potansiyel taşır.")

    esik=urun_df["oneri_skoru"].quantile(0.70)
    fig=px.histogram(urun_df,x="oneri_skoru",nbins=35,
                     color_discrete_sequence=[MAVI],opacity=0.8)
    fig.add_vline(x=esik,line_dash="dash",line_color=TURUNCU,line_width=2,
                  annotation_text=f"Eşik: {esik:.3f}",
                  annotation_font=dict(color=TURUNCU,size=11))
    tema(fig,h=240,xaxis=dict(**EKSEN,title="Öneri Skoru"),
         yaxis=dict(**EKSEN,title="Ürün Sayısı"))
    st.plotly_chart(fig,use_container_width=True)
    st.caption("📌 Turuncu kesikli çizgi önerilir eşiğini gösterir. Bu eşiğin üzerindeki ürünler genel öneri listesine aday olur; eşik altında kalanlar daha zayıf performans sinyali taşır.")

    try:
        auc_val=float(pd.read_csv("outputs/model_met.csv")["auc"].iloc[0])
    except: auc_val=0.0
    ozet=pd.DataFrame({
        "Metrik":["Toplam Ürün","Önerilen","Eşik","Yorum","Pozitif","Nötr","Negatif",
                  "Ort. Rating","Ort. Skor","AUC","Müşteri","Şampiyon"],
        "Değer":[f"{len(urun_df):,}",f"{int(urun_df['onerilir'].sum()):,}",
                 f"{esik:.4f}",f"{len(duygu_df):,}",
                 f"{(duygu_df['duygu']=='positive').sum():,}",
                 f"{(duygu_df['duygu']=='neutral').sum():,}",
                 f"{(duygu_df['duygu']=='negative').sum():,}",
                 f"{urun_df['ort_rating'].mean():.2f}/5",
                 f"{urun_df['oneri_skoru'].mean():.4f}",
                 f"{auc_val:.3f}",f"{len(rfm_df):,}",
                 f"{int(rfm_df['segment'].eq('Şampiyon').sum()):,}"],
    })
    st.dataframe(ozet,use_container_width=True,hide_index=True)

st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align:center;padding:16px 0;margin-top:8px;
            border-top:1px solid #E2E8F0;background:#F8FAFC">
  <span style="font-size:0.8rem;font-weight:700;color:#1E3A5F">🛒 ShopLens</span>
  <span style="color:#CBD5E1;margin:0 8px">·</span>
  <span style="font-size:0.7rem;color:#94A3B8">E-Ticaret Ürün Önerme & Müşteri Analiz Platformu</span>
  <br><span style="font-size:0.62rem;color:#CBD5E1">
    Python · Streamlit · Plotly · scikit-learn · VADER · Pandas
  </span>
</div>
""", unsafe_allow_html=True)
