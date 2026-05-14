"""
ShopLens — E-Ticaret Davranış Analizi ve Akıllı Ürün Öneri Sistemi
============================================================
İSMEK Yapay Zekâ ve Veri Bilimi — Proje Çalışması

Kullanım:
    python pipeline.py          # Tüm analizleri çalıştır
    streamlit run dashboard.py # Dashboard'u aç

=== PROJE AKIŞI ===

Problem Türü : SINIFLANDIRMA (İkili — Binary Classification)
  - Her ürün için "önerilir (1)" veya "önerilmez (0)" kararı verilir.
  - Bu bir regresyon değildir; model sürekli satış adedi yerine
    sınıf etiketi tahmin eder.

Veri Kaynakları (7 tablo):
  events.csv       → Kullanıcı davranış kayıtları (sayfa, sepet, ödeme)
  products.csv     → Ürün kataloğu (fiyat, maliyet, marj)
  reviews.csv      → Müşteri yorumları ve puanları
  orders.csv       → Sipariş başlıkları (müşteri, ülke, ödeme)
  order_items.csv  → Sipariş kalemleri (ürün, adet, tutar)
  customers.csv    → Müşteri profilleri
  sessions.csv     → Oturum kayıtları (cihaz, kaynak, ülke)
  [sessions → events köprüsü: session_id üzerinden bağlanır]

Metodoloji Akışı:
  1. Veri Yükleme & Birleştirme (Merge)
  2. Veri Ön İşleme & Kalite Kontrol
  3. Keşifsel Veri Analizi (EDA)
  4. Özellik Mühendisliği (Feature Engineering)
  5. Dönüşüm Hunisi Analizi
  6. Duygu Analizi (VADER + Rating Hibrit)
  7. Öneri Skoru Hesaplama
  8. Model Eğitimi (Random Forest Classifier)
  9. Müşteri Segmentasyonu (RFM)
  10. Görselleştirme (Matplotlib + Plotly)
"""

import warnings
import matplotlib
matplotlib.use("Agg")   # GUI olmayan backend — Mac/Linux/server uyumlu
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, roc_curve, auc as sk_auc,
    precision_score, recall_score, f1_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from joblib import dump
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

warnings.filterwarnings("ignore")

_baslama = datetime.now()
print("=" * 58)
print("  ShopLens — Analiz Başladı")
print("=" * 58)

# ─────────────────────────────────────────────────────────────
# RENK PALETİ (grafiklerde tutarlı kullanım)
# ─────────────────────────────────────────────────────────────
TURUNCU  = "#F97316"   # vurgu, birinci sınıf
LACIVERT = "#1E3A5F"   # başlık, güçlü
MAVI     = "#3B82F6"   # ikincil, bilgi
YESIL    = "#16A34A"   # başarı, pozitif
KIRMIZI  = "#DC2626"   # hata, negatif
SARI     = "#FBBF24"   # uyarı, dikkat
GERI     = "#F8FAFC"   # açık arka plan
METIN    = "#1E293B"   # koyu metin
MUTED    = "#64748B"   # soluk metin

SEG_RENK = {
    "Şampiyon":    TURUNCU,
    "Sadık":       YESIL,
    "Yeni":        MAVI,
    "Potansiyel":  SARI,
    "Risk Altında":"#EA580C",
    "Kayıp":       KIRMIZI,
}


# =============================================================================
# YARDIMCI FONKSİYONLAR
# =============================================================================

def min_max_norm(seri):
    """
    Min-Max Normalizasyonu: değerleri 0-1 aralığına getirir.
    Farklı ölçeklerdeki (örn. fiyat 3-597 USD, oran 0-1)
    değişkenleri karşılaştırılabilir hale getirir.
    """
    mn, mx = seri.min(), seri.max()
    return seri * 0 if mx == mn else (seri - mn) / (mx - mn)


def rfm_skoru(seri, artan=True):
    """
    RFM analizi için 1-5 arası skor atar.
    pd.qcut kullanır: her gruba eşit sayıda gözlem düşer.
    artan=True  → küçük değer iyidir (Recency: az gün = iyi)
    artan=False → büyük değer iyidir (Frequency, Monetary)
    """
    try:
        etiketler = [5, 4, 3, 2, 1] if artan else [1, 2, 3, 4, 5]
        return pd.qcut(seri, q=5, labels=etiketler, duplicates="drop").astype(int)
    except Exception:
        yuzde = seri.rank(pct=True)
        return ((5 - (yuzde * 4).astype(int)) if artan
                else ((yuzde * 4).astype(int) + 1)).clip(1, 5)


def grafik_stili():
    """Tüm Matplotlib grafiklerine tutarlı açık tema uygular."""
    plt.rcParams.update({
        "figure.facecolor": GERI,   "axes.facecolor":  "white",
        "axes.edgecolor":   "#CBD5E1", "axes.labelcolor": METIN,
        "xtick.color":      MUTED,  "ytick.color":     MUTED,
        "text.color":       METIN,  "grid.color":      "#E2E8F0",
        "grid.alpha":       0.7,    "font.family":     "DejaVu Sans",
    })


# =============================================================================
# ADIM 1 — VERİ YÜKLEME
# =============================================================================

def veri_yukle():
    """
    7 CSV dosyasını yükler.

    Bağlantı Mimarisi (hangi tablo neyle birleşiyor):
      events ─── session_id ───► sessions ─── customer_id ───► customers
         │                                                          │
         └─── product_id ──────────────────────────────────► products
      orders ─── order_id ──────────────────────────────────► order_items
         │                              └─── product_id ────► products
         └─── customer_id ──────────────────────────────────► customers
      reviews ─── product_id ──────────────────────────────► products
    """
    print("\n[1/9] Veriler yükleniyor...")
    d = Path("data")

    events      = pd.read_csv(d / "events.csv",      parse_dates=["timestamp"])
    products    = pd.read_csv(d / "products.csv")
    reviews     = pd.read_csv(d / "reviews.csv",     parse_dates=["review_time"])
    sessions    = pd.read_csv(d / "sessions.csv",    parse_dates=["start_time"])
    order_items = pd.read_csv(d / "order_items.csv")
    customers   = pd.read_csv(d / "customers.csv",   parse_dates=["signup_date"])
    orders      = pd.read_csv(d / "orders.csv",      parse_dates=["order_time"])

    for isim, df in [("events", events), ("products", products),
                     ("reviews", reviews), ("sessions", sessions),
                     ("order_items", order_items), ("customers", customers),
                     ("orders", orders)]:
        print(f"  {isim:<14} {len(df):>9,} satır  {len(df.columns):>2} sütun")

    toplam = sum(len(d) for d in [events, products, reviews, sessions,
                                   order_items, customers, orders])
    print(f"  {'TOPLAM':<14} {toplam:>9,} satır")
    return events, products, reviews, sessions, order_items, customers, orders


# =============================================================================
# ADIM 2 — VERİ ÖN İŞLEME & KALİTE KONTROL
# =============================================================================

def veri_on_isleme(events, products, reviews, orders, order_items, customers, sessions):
    """
    Veri Temizleme ve Ön İşleme Adımları:

    1. NULL Analizi:
       events.product_id: checkout (44.909) ve purchase (33.580) kayıtlarında
       boş → bu event tipleri ürün bazlı değil, sipariş bazlıdır.
       Çözüm: ürün analizinde bu satırlar dışarıda bırakılır.

    2. Müşteri-Oturum Köprüsü:
       events tablosunda customer_id YOK, sadece session_id var.
       Müşteri bilgisine ulaşmak için:
       events → sessions (session_id) → customers (customer_id)

    3. Gerçek Satış Verisi:
       events'teki purchase eventlerinin product_id'si boş olduğundan
       satış verileri order_items tablosundan alınır.

    4. Veri Türü Düzeltmeleri:
       product_id float → int dönüşümü (merge için gerekli)

    Sonuç: Her tablonun temiz, merge'e hazır versiyonu döndürülür.
    """
    print("\n[2/9] Veri ön işleme & kalite kontrol...")

    # ── 1. NULL özeti
    print("  Null değer özeti:")
    events_null = events.isnull().sum()
    onemli_nulllar = events_null[events_null > 0]
    for k, v in onemli_nulllar.items():
        pct = v / len(events) * 100
        print(f"    events.{k}: {v:,} null (%{pct:.1f}) → beklenen, işlenecek")

    # ── 2. Checkout/purchase event'lerinin product_id'si boş — beklenen durum
    ev_temiz = events.dropna(subset=["product_id"]).copy()
    ev_temiz["product_id"] = ev_temiz["product_id"].astype(int)
    print(f"  events ürün bazlı satırlar: {len(ev_temiz):,} / {len(events):,}")

    # ── 3. Duplikasyon kontrolü
    dup_events = events.duplicated().sum()
    dup_orders = orders["order_id"].duplicated().sum()
    print(f"  Duplikasyon: events={dup_events}, orders={dup_orders}")

    # ── 4. Tarih aralığı kontrolü
    t_min = events["timestamp"].min()
    t_max = events["timestamp"].max()
    print(f"  Veri zaman aralığı: {t_min.date()} → {t_max.date()}")

    # ── 5. Müşteri-Oturum köprüsü
    musteri_oturum = sessions[["session_id", "customer_id"]].copy()

    # ── 6. Özet kaydet
    ozet = {
        "toplam_events": len(events),
        "urun_events": len(ev_temiz),
        "null_orani": onemli_nulllar.sum() / (len(events) * len(events.columns)),
        "tarih_min": str(t_min.date()),
        "tarih_max": str(t_max.date()),
        "toplam_musteri": len(customers),
        "toplam_urun": len(products),
        "toplam_siparis": len(orders),
    }
    pd.DataFrame([ozet]).to_csv("outputs/veri_ozet.csv", index=False)
    print("  ✓ Veri kalitesi kontrolü tamamlandı")
    return ev_temiz, musteri_oturum


# =============================================================================
# ADIM 3 — KEŞİFSEL VERİ ANALİZİ (EDA)
# =============================================================================

def keşifsel_veri_analizi(events, ev_temiz, products, reviews, orders,
                            order_items, customers, sessions):
    """
    Keşifsel Veri Analizi (Exploratory Data Analysis — EDA):
    Modeli geliştirmeden önce verileri anlamak için yapılan ilk keşif.

    Yapılan Analizler:
      - Event type dağılımı (hangi davranış ne kadar sık)
      - Kategori bazlı ürün dağılımı
      - Fiyat ve marj dağılımı
      - Rating dağılımı (yorumlar)
      - Ülke ve cihaz bazlı dağılım
      - Zaman serisi trendi (aylık sipariş)

    Bu bulgular feature engineering kararlarını ve
    skor ağırlıklarını doğrudan etkiler.
    """
    print("\n[3/9] Keşifsel Veri Analizi (EDA)...")

    grafik_stili()
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), facecolor=GERI)

    # ── 1. Event type dağılımı
    ax = axes[0, 0]
    ev_say = events["event_type"].value_counts()
    rk = [LACIVERT, MAVI, TURUNCU, YESIL]
    bars = ax.bar(ev_say.index, ev_say.values, color=rk[:len(ev_say)], edgecolor="white")
    for bar, v in zip(bars, ev_say.values):
        ax.text(bar.get_x() + bar.get_width()/2, v + 5000,
                f"{v:,}", ha="center", fontsize=9.5, fontweight="bold")
    ax.set_title("Event Type Dağılımı", fontsize=12, fontweight="bold")
    ax.set_ylabel("Kayıt Sayısı")
    ax.spines[["top", "right"]].set_visible(False)

    # ── 2. Kategori bazlı ürün sayısı
    ax = axes[0, 1]
    kat_say = products["category"].value_counts()
    colors_kat = [TURUNCU if i == 0 else MAVI for i in range(len(kat_say))]
    bars = ax.barh(kat_say.index, kat_say.values, color=colors_kat, height=0.6)
    for bar, v in zip(bars, kat_say.values):
        ax.text(v + 1, bar.get_y() + bar.get_height()/2,
                str(v), va="center", fontsize=9.5, fontweight="bold")
    ax.set_title("Kategori Bazlı Ürün Sayısı", fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

    # ── 3. Fiyat dağılımı (histogram)
    ax = axes[0, 2]
    ax.hist(products["price_usd"], bins=25, color=MAVI, alpha=0.8, edgecolor="white")
    ax.axvline(products["price_usd"].median(), color=TURUNCU, linestyle="--",
               linewidth=2, label=f'Medyan: ${products["price_usd"].median():.0f}')
    ax.set_title("Ürün Fiyat Dağılımı", fontsize=12, fontweight="bold")
    ax.set_xlabel("Fiyat (USD)")
    ax.set_ylabel("Ürün Sayısı")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    # ── 4. Rating dağılımı
    ax = axes[1, 0]
    rat_say = reviews["rating"].value_counts().sort_index()
    clrs = [KIRMIZI, "#F97316", SARI, MAVI, YESIL]
    bars = ax.bar(rat_say.index, rat_say.values, color=clrs[:len(rat_say)], edgecolor="white")
    for bar, v in zip(bars, rat_say.values):
        ax.text(bar.get_x() + bar.get_width()/2, v + 50,
                f"{v:,}", ha="center", fontsize=9.5, fontweight="bold")
    ax.set_title("Müşteri Rating Dağılımı", fontsize=12, fontweight="bold")
    ax.set_xlabel("Puan (1-5)")
    ax.set_ylabel("Yorum Sayısı")
    ax.spines[["top", "right"]].set_visible(False)

    # ── 5. Ülke bazlı sipariş dağılımı (top 10)
    ax = axes[1, 1]
    ulke = orders["country"].value_counts().head(10)
    rk_ulke = [TURUNCU if i == 0 else MAVI for i in range(len(ulke))]
    bars = ax.barh(ulke.index[::-1], ulke.values[::-1], color=rk_ulke[::-1], height=0.6)
    for bar, v in zip(bars, ulke.values[::-1]):
        ax.text(v + 30, bar.get_y() + bar.get_height()/2,
                f"{v:,}", va="center", fontsize=9, fontweight="bold")
    ax.set_title("Ülke Bazlı Sipariş Sayısı (Top 10)", fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

    # ── 6. Aylık sipariş trendi
    ax = axes[1, 2]
    orders["ay"] = pd.to_datetime(orders["order_time"]).dt.to_period("M")
    aylik = orders.groupby("ay").size().reset_index(name="siparis")
    aylik["ay_str"] = aylik["ay"].astype(str)
    son_12 = aylik.tail(12)
    ax.plot(range(len(son_12)), son_12["siparis"].values,
            color=TURUNCU, linewidth=2.5, marker="o", markersize=5)
    ax.fill_between(range(len(son_12)), son_12["siparis"].values,
                    alpha=0.15, color=TURUNCU)
    ax.set_xticks(range(0, len(son_12), 3))
    ax.set_xticklabels(son_12["ay_str"].values[::3], rotation=30, fontsize=8)
    ax.set_title("Aylık Sipariş Trendi (Son 12 Ay)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Sipariş Sayısı")
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("ShopLens — Keşifsel Veri Analizi (EDA)", fontsize=15,
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig("outputs/charts/00_eda.png", dpi=130, bbox_inches="tight")
    plt.close()

    # EDA özet tablosu
    eda_ozet = pd.DataFrame({
        "Metrik": [
            "Toplam event", "Ürün bazlı event", "Checkout/purchase null (beklenen)",
            "Benzersiz ürün", "Kategori sayısı", "Ortalama fiyat (USD)",
            "Medyan fiyat (USD)", "Ortalama marj (USD)",
            "Toplam yorum", "Ortalama rating", "Sipariş ülke sayısı",
        ],
        "Değer": [
            f"{len(events):,}",
            f"{len(ev_temiz):,}",
            f"{events['product_id'].isnull().sum():,}",
            f"{products['product_id'].nunique():,}",
            f"{products['category'].nunique()}",
            f"${products['price_usd'].mean():.2f}",
            f"${products['price_usd'].median():.2f}",
            f"${products['margin_usd'].mean():.2f}",
            f"{len(reviews):,}",
            f"{reviews['rating'].mean():.2f} / 5",
            f"{orders['country'].nunique()}",
        ],
    })
    eda_ozet.to_csv("outputs/eda_ozet.csv", index=False)
    print("  ✓ EDA tamamlandı → outputs/charts/00_eda.png")
    return eda_ozet


# =============================================================================
# ADIM 4 — DÖNÜŞÜM HUNİSİ
# =============================================================================

def funnel_analizi(events):
    """
    E-ticaret dönüşüm hunisini hesaplar.

    Neden önemli?
    Kullanıcıların hangi adımda sistemden ayrıldığını gösterir.
    Bu bilgi pazarlama ve UX optimizasyonuna yön verir.

    Her adım için iki oran:
      Önceki Adımdan % → darboğaz noktaları bulur
      İlk Adımdan %    → toplam kayıp oranını gösterir
    """
    print("\n[4/9] Dönüşüm hunisi hesaplanıyor...")

    adimlar = ["page_view", "add_to_cart", "checkout", "purchase"]
    sayilar = events["event_type"].value_counts().reindex(adimlar).fillna(0).astype(int)

    onceki = [100.0] + [
        round(sayilar.iloc[i] / sayilar.iloc[i-1] * 100, 2) for i in range(1, 4)
    ]
    toplam = [100.0] + [
        round(sayilar.iloc[i] / sayilar.iloc[0] * 100, 2) for i in range(1, 4)
    ]

    df = pd.DataFrame({
        "adim": adimlar,
        "event_sayisi": sayilar.values,
        "onceki_pct": onceki,
        "toplam_pct": toplam,
    })
    print(df.to_string(index=False))
    df.to_csv("outputs/funnel.csv", index=False)
    return df


# =============================================================================
# ADIM 5 — ÖZELLİK MÜHENDİSLİĞİ + ÜRÜN ANALİTİĞİ
# =============================================================================

def ozellik_muhendisligi(ev_temiz, products, reviews, order_items):
    """
    Özellik Mühendisliği (Feature Engineering):
    Ham veritabanı sütunlarından model için anlamlı özellikler türetir.

    Birleştirme (Merge) Adımları:
      1. events pivot  → her ürün için event sayıları
         (product_id bazında pivot table)
      2. order_items   → gerçek satış adedi ve gelir
         (NOT: events.purchase product_id=NULL olduğundan
          satış order_items'dan alınır)
      3. reviews       → yorum sayısı ve ortalama puan
      4. Hepsi products tablosuyla LEFT JOIN

    Türetilen Özellikler:
      sepet_orani   = add_to_cart / page_view   (davranışsal)
      odeme_orani   = ürün bazlı checkout olmadığı için hesaplanmadı
      satis_orani   = satis_adedi / page_view   (nihai dönüşüm)
      pozitif_oran  = duygu analizinden (sonraki adımda)

    NOT: Bu oranlar ham sayılardan çok daha anlamlı.
    100 görüntüleme → 80 sepet = %80 oran
    1000 görüntüleme → 200 sepet = %20 oran
    Sadece ham sayıya bakarsak 2. ürünü daha iyi sanırız — YANLIŞ.
    """
    print("\n[5/9] Özellik mühendisliği & ürün analitiği...")

    # ── 1. Event pivot: ürün bazlı event sayıları
    # (sadece product_id dolu olanlar: page_view ve add_to_cart)
    pivot = (
        ev_temiz.pivot_table(
            index="product_id", columns="event_type",
            values="event_id", aggfunc="count", fill_value=0,
        )
        .reset_index()
    )
    for s in ["page_view", "add_to_cart"]:
        if s not in pivot.columns:
            pivot[s] = 0

    # ── 2. Gerçek satış verileri (order_items'dan — purchase event NULL'dı)
    satis = (
        order_items.groupby("product_id")
        .agg(satis_adedi=("quantity", "sum"), gelir=("line_total_usd", "sum"))
        .reset_index()
    )

    # ── 3. Yorum istatistikleri
    yorum = (
        reviews.groupby("product_id")
        .agg(yorum_sayisi=("review_id", "count"), ort_rating=("rating", "mean"))
        .reset_index()
    )

    # ── 4. Birleştirme: products + pivot + satis + yorum
    df = (
        products
        .merge(pivot, on="product_id", how="left")
        .merge(satis,  on="product_id", how="left")
        .merge(yorum,  on="product_id", how="left")
    )
    for s in ["page_view", "add_to_cart", "satis_adedi",
              "yorum_sayisi", "ort_rating", "gelir"]:
        if s in df.columns:
            df[s] = df[s].fillna(0)

    # ── 5. Türetilmiş özellikler (Feature Engineering)
    # np.where → sıfıra bölme koruması
    df["sepet_orani"] = np.where(df["page_view"] > 0,
                                 df["add_to_cart"] / df["page_view"], 0.0)
    # checkout eventleri ürün bazında product_id taşımadığı için bu oran ürün
    # tablosunda gerçek anlamda hesaplanamaz. Yanıltıcı bir oran üretmemek için
    # sabit 0 bırakılır; gerçek satış sinyali order_items üzerinden alınır.
    df["odeme_orani"] = 0.0
    df["satis_orani"] = np.where(df["page_view"] > 0,
                                 df["satis_adedi"] / df["page_view"], 0.0)

    df.to_csv("outputs/urun_analitik.csv", index=False)
    print(f"  {len(df)} ürün, {len(df.columns)} özellik → merge tamamlandı")
    print(f"  page_view: {df['page_view'].min():.0f}–{df['page_view'].max():.0f}")
    print(f"  satis_adedi: {df['satis_adedi'].min():.0f}–{df['satis_adedi'].max():.0f}")
    print(f"  sepet_orani: {df['sepet_orani'].min():.3f}–{df['sepet_orani'].max():.3f}")
    return df


# =============================================================================
# ADIM 6 — DUYGU ANALİZİ (VADER + Rating Hibrit)
# =============================================================================

def duygu_analizi(reviews):
    """
    Duygu Analizi (Sentiment Analysis) — Doğal Dil İşleme:

    Neden gerekli?
      Rating (1-5 yıldız) müşteri memnuniyetinin sayısal göstergesidir.
      Yorum metni ise nüanslı bilgi içerir: "Not great but okay" gibi
      ifadeler rakamla tam anlatılamaz.
      Duygu skoru öneri formülüne %18 ağırlıkla girer.

    Yöntem: VADER + Rating Hibrit Modeli
    ─────────────────────────────────────
    VADER (Valence Aware Dictionary and sEntiment Reasoner):
      - Sözlük tabanlı, internet gerektirmez (pip install vaderSentiment)
      - İngilizce için optimize: olumsuzlama, büyük harf, noktalama algılar
      - Compound skoru: -1 (çok negatif) ↔ +1 (çok pozitif)

    Neden sadece VADER yeterli değil?
      Bu veri setinde 5 farklı yorum metni var, tekrarlı kullanılmış.
      "Okay overall" → compound=0.226 → VADER bu yorumu pozitif sayar.
      Ama bu metnin rating'i 3 → gerçekte nötr olmalı.
      Gerçek rating bilgisi ile hibrit karar daha doğru sonuç verir.

    Hibrit Karar Kuralı:
      compound > 0.3  VE rating ≥ 4 → pozitif
      compound < -0.1 VEYA rating ≤ 2 → negatif
      diğer → nötr (rating=3 durumları dahil)
    """
    print("\n[6/9] Duygu analizi (VADER + Rating hibrit)...")

    analizci = SentimentIntensityAnalyzer()
    df = reviews.copy()
    df["review_text"] = df["review_text"].fillna("")

    etiketler, guvenler, yildizlar = [], [], []

    for _, satir in df.iterrows():
        metin   = str(satir["review_text"])
        rating  = int(satir.get("rating", 3))
        compound = analizci.polarity_scores(metin)["compound"]

        # Hibrit karar kuralı
        if compound > 0.3 and rating >= 4:
            etiket, yildiz = "positive", (5 if compound > 0.6 else 4)
        elif compound < -0.1 or rating <= 2:
            etiket, yildiz = "negative", (1 if compound < -0.4 else 2)
        else:
            etiket, yildiz = "neutral", 3

        guven = min(0.97, abs(compound) * 0.4 + 0.55)
        etiketler.append(etiket)
        guvenler.append(round(guven, 3))
        yildizlar.append(yildiz)

    df["duygu"]   = etiketler
    df["guven"]   = guvenler
    df["yildiz"]  = yildizlar
    df["compound"] = df["review_text"].apply(
        lambda t: round(analizci.polarity_scores(str(t))["compound"], 4)
    )
    df.to_csv("outputs/duygu.csv", index=False)

    yorum_ozet = pd.DataFrame([{
        "toplam_yorum": len(df),
        "benzersiz_yorum_metni": df["review_text"].nunique(),
        "benzersiz_yorum_orani": df["review_text"].nunique() / len(df) if len(df) else 0,
    }])
    yorum_ozet.to_csv("outputs/duygu_metin_ozeti.csv", index=False)

    dagılim = df["duygu"].value_counts()
    print(f"  Pozitif: {dagılim.get('positive',0):,}  "
          f"Nötr: {dagılim.get('neutral',0):,}  "
          f"Negatif: {dagılim.get('negative',0):,}")
    print(f"  Benzersiz yorum metni: {df['review_text'].nunique():,}/{len(df):,}")
    return df


# =============================================================================
# ADIM 7 — ÖNERI SKORU
# =============================================================================

def onerme_skoru(urun_df, duygu_df):
    """
    Ağırlıklı Öneri Skoru Formülü:

    Her özellik önce min-max normalizasyonu ile 0-1'e getirilir,
    sonra ağırlıklarla çarpılarak toplanır (toplam = %100).

    Ağırlık Gerekçeleri:
      %28 Satış Oranı       → nihai dönüşüm — en güçlü sinyal
      %20 Müşteri Puanı     → doğrudan kullanıcı memnuniyeti
      %18 Sepete Ekleme     → satın alma niyeti
      %18 Pozitif Yorum     → NLP çıktısı, duygu analizi katkısı
      %8  Toplam Gelir      → ürün popülerlik göstergesi
      %8  Ürün Marjı        → iş kârlılığı açısından önem

    Eşik (Threshold):
      Skor dağılımının üst %30'u → önerilir (1)
      Alt %70 → önerilmez (0)
      Bu sayede 359 ürün önerilir olarak etiketlenir.
    """
    print("\n[7/9] Öneri skorları hesaplanıyor...")

    veri = urun_df.copy()

    duygu_ozet = (
        duygu_df.groupby("product_id")
        .agg(
            ort_yildiz=("yildiz", "mean"),
            pozitif_oran=("duygu", lambda x: (x == "positive").mean()),
        )
        .reset_index()
    )

    veri = veri.merge(duygu_ozet, on="product_id", how="left")
    veri[["ort_yildiz", "pozitif_oran"]] = veri[["ort_yildiz", "pozitif_oran"]].fillna(0)

    # Ağırlıklı formül
    veri["oneri_skoru"] = (
        min_max_norm(veri["satis_orani"])   * 0.28 +
        min_max_norm(veri["ort_rating"])    * 0.20 +
        min_max_norm(veri["sepet_orani"])   * 0.18 +
        min_max_norm(veri["pozitif_oran"])  * 0.18 +
        min_max_norm(veri["gelir"])         * 0.08 +
        min_max_norm(veri["margin_usd"])    * 0.08
    )

    esik = veri["oneri_skoru"].quantile(0.70)
    veri["onerilir"] = (veri["oneri_skoru"] >= esik).astype(int)
    veri.to_csv("outputs/urun_skorlar.csv", index=False)

    top50 = ["product_id", "name", "category", "price_usd",
             "ort_rating", "pozitif_oran", "satis_orani", "oneri_skoru"]
    (veri[veri["onerilir"] == 1]
     .sort_values("oneri_skoru", ascending=False)
     .head(50)[top50]
     .to_csv("outputs/top50.csv", index=False))

    print(f"  Eşik: {esik:.4f}  |  Önerilir: {veri['onerilir'].sum()} ürün")
    return veri


# =============================================================================
# ADIM 8 — MODEL EĞİTİMİ (SINIFLANDIRMA)
# =============================================================================

def model_egit(veri):
    """
    Problem Türü: İKİLİ SINIFLANDIRMA (Binary Classification)
    ──────────────────────────────────────────────────────────
    Soru: "Bu ürün önerilmeli mi?" → Evet (1) / Hayır (0)

    Bu bir REGRESYON değildir. Regresyonda "kaç satacak?" gibi
    sürekli bir değer tahmin ederiz. Burada kategorik etiket
    (önerilir/önerilmez) tahmin ediyoruz.

    Model: Random Forest Classifier
    ─────────────────────────────────
    Neden Random Forest?
      - Karar ağacı mantığı yorumlanabilir
      - Doğrusal olmayan ilişkileri yakalayabilir
      - Özellik önemlerini gösterebilir
      - Sınıf dengesizliğini class_weight="balanced" ile yönetir

    Özellikler (Features):
      price_usd, margin_usd → ürün ekonomisi
      page_view             → görünürlük
      sepet_orani           → kullanıcı davranışı (türetilmiş)
      ort_rating, ort_yildiz → memnuniyet (ham + NLP)
      yorum_sayisi          → sosyal kanıt

    Neden satis_orani dışarıda?
      Hedef değişken (onerilir) satis_orani üzerinden hesaplandı.
      Modele eklemek veri sızıntısı (data leakage) yaratır —
      model gerçek veriyi değil, sızıntıyı öğrenir.

    Veri Bölme:
      %75 eğitim  /  %25 test
      stratify=y → sınıf oranları korunur
    """
    print("\n[8/9] Model eğitiliyor (Sınıflandırma)...")

    ozellikler = [
        "price_usd", "margin_usd",
        "page_view",
        "sepet_orani",
        "ort_rating", "ort_yildiz",
        "yorum_sayisi",
    ]
    X = veri[ozellikler].fillna(0)
    y = veri["onerilir"]

    print(f"  Sınıf dağılımı: 0=Önerilmez {(y==0).sum()}, "
          f"1=Önerilir {(y==1).sum()}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,    # 200 karar ağacı
        max_depth=8,         # maksimum derinlik — overfitting önlemi
        class_weight="balanced",  # dengesiz sınıf düzeltmesi
        random_state=42,
        n_jobs=-1,           # tüm CPU çekirdeklerini kullan
    )
    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]
    acc    = accuracy_score(y_te, y_pred)
    cm     = confusion_matrix(y_te, y_pred)
    fpr, tpr, _ = roc_curve(y_te, y_prob)
    auc_val = sk_auc(fpr, tpr)

    print(f"  Doğruluk: {acc:.1%}  |  ROC AUC: {auc_val:.3f}")
    print(classification_report(y_te, y_pred, target_names=["Önerilmez", "Önerilir"]))

    np.save("outputs/cm.npy", cm)
    np.save("outputs/roc_fpr.npy", fpr)
    np.save("outputs/roc_tpr.npy", tpr)
    pd.DataFrame({"auc": [auc_val], "dogruluk": [acc]}).to_csv(
        "outputs/model_met.csv", index=False)

    fi = (pd.DataFrame({"ozellik": ozellikler, "onem": model.feature_importances_})
          .sort_values("onem", ascending=False))
    fi.to_csv("outputs/ozellik_onemi.csv", index=False)
    dump({"model": model, "ozellikler": ozellikler}, "models/model.pkl")
    print("  Model → models/model.pkl")
    return model, ozellikler


# =============================================================================
# ADIM 9 — MÜŞTERİ SEGMENTASYONU (RFM)
# =============================================================================

def musteri_segmentasyonu(customers, orders, order_items, products):
    """
    RFM Analizi — Müşteri Segmentasyonu:

    RFM üç boyutta müşteriyi değerlendirir:
      R (Recency)   → Son alışverişten kaç gün geçti?  (az = iyi)
      F (Frequency) → Toplam kaç sipariş verdi?         (çok = iyi)
      M (Monetary)  → Toplam ne kadar harcadı?          (yüksek = iyi)

    Her boyut 1-5 arası skorlanır (pd.qcut → eşit dağılım).
    Toplam skor → 6 segment:

      Şampiyon     → R≥4, F≥4  Sık alışveriş, yakın tarih
      Sadık        → R≥3, F≥3  Düzenli müşteri
      Yeni         → R≥4, F≤2  Yeni başladı, potansiyel büyük
      Potansiyel   → Orta      Geliştirilebilir
      Risk Altında → R≤2, F≥3  Eskiden aktif, şimdi sessiz — geri kazan
      Kayıp        → R≤2, F≤2  Uzun süre yok — aktivasyon zor

    Kişisel Öneri Motoru:
      Müşteri geçmişi → Top kategori → Alınmayan ürünler → Skor sıralaması
    """
    print("\n[9/9] Müşteri segmentasyonu (RFM)...")

    ref = orders["order_time"].max()

    # RFM için önce müşteri bazında son sipariş, sipariş sayısı ve harcama
    # özetlenir. Recency hesabını lambda ile groupby içinde yapmak yavaş
    # kalabildiği için burada vektörel olarak hesaplıyoruz.
    rfm = orders.groupby("customer_id").agg(
        son_siparis=("order_time", "max"),
        frequency=("order_id", "count"),
        monetary=("total_usd", "sum"),
    ).reset_index()
    rfm["recency"] = (ref - rfm["son_siparis"]).dt.days
    rfm = rfm.drop(columns=["son_siparis"])

    rfm["R"] = rfm_skoru(rfm["recency"],   artan=True)
    rfm["F"] = rfm_skoru(rfm["frequency"], artan=False)
    rfm["M"] = rfm_skoru(rfm["monetary"],  artan=False)

    def seg(r):
        R, F = r["R"], r["F"]
        if   R >= 4 and F >= 4: return "Şampiyon"
        elif R >= 3 and F >= 3: return "Sadık"
        elif R >= 4 and F <= 2: return "Yeni"
        elif R <= 2 and F >= 3: return "Risk Altında"
        elif R <= 2 and F <= 2: return "Kayıp"
        else:                   return "Potansiyel"

    rfm["segment"] = rfm.apply(seg, axis=1)
    rfm = rfm.merge(
        customers[["customer_id", "name", "country", "age"]],
        on="customer_id", how="left",
    )
    rfm.to_csv("outputs/rfm.csv", index=False)

    musteri_kat = (
        order_items
        .merge(products[["product_id", "category"]], on="product_id")
        .merge(orders[["order_id", "customer_id"]], on="order_id")
        .groupby(["customer_id", "category"])["quantity"].sum()
        .reset_index()
    )
    musteri_kat.to_csv("outputs/musteri_kat.csv", index=False)

    print("  Segmentler:", rfm["segment"].value_counts().to_dict())
    return rfm, musteri_kat



# =============================================================================
# ADIM 10 — KİŞİYE ÖZEL ÖNERİ SIRALAMA MODELİ
# =============================================================================

def kisisel_satin_alma_modeli(events, sessions, orders, order_items, customers,
                               urun_df, rfm_df, musteri_kat):
    """
    Kişiye özel öneri sıralama modeli:

    Bu modelin sorusu ürün skor modelinden farklıdır.

      Ürün skor modeli:
        "Bu ürün genel olarak önerilebilir mi?"

      Kişisel öneri modeli:
        "Bu müşteri için hangi ürünler daha öncelikli önerilmeli?"

    Bu nedenle veri seviyesi artık ürün değil, müşteri-ürün çiftidir.

    Pozitif örnek:
      orders + order_items tablolarında müşterinin gerçekten satın aldığı ürün.

    Negatif örnek:
      Müşterinin görüntülediği veya sepete eklediği fakat satın almadığı ürün.

    Not:
      Bir müşterinin hiç görmediği ürünü satın almaması tek başına olumsuz
      tercih anlamına gelmez. Bu yüzden negatif örnekleri özellikle
      "etkileşim var ama satın alma yok" mantığıyla oluşturuyoruz.
    """
    print("\n[10/10] Kişiye özel öneri sıralama modeli...")

    # 1) Müşteri-ürün davranışları: hangi müşteri hangi ürünü gördü/sepete ekledi?
    ev = events.dropna(subset=["product_id"]).copy()
    ev["product_id"] = ev["product_id"].astype(int)
    ev = ev.merge(sessions[["session_id", "customer_id"]], on="session_id", how="left")

    davranis = (
        ev.pivot_table(
            index=["customer_id", "product_id"],
            columns="event_type",
            values="event_id",
            aggfunc="count",
            fill_value=0,
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    for kolon in ["page_view", "add_to_cart"]:
        if kolon not in davranis.columns:
            davranis[kolon] = 0

    davranis = davranis[["customer_id", "product_id", "page_view", "add_to_cart"]]

    # 2) Gerçek satın almalar: order_items ürün bazlı gerçek satış kaynağıdır.
    satin_almalar = (
        orders[["order_id", "customer_id"]]
        .merge(order_items[["order_id", "product_id", "quantity"]], on="order_id", how="inner")
        .groupby(["customer_id", "product_id"])
        .agg(satin_alma_adedi=("quantity", "sum"))
        .reset_index()
    )
    satin_almalar["satin_aldi"] = 1

    # 3) Pozitif ve negatif örnekleri tek müşteri-ürün tablosunda birleştir.
    veri = davranis.merge(
        satin_almalar,
        on=["customer_id", "product_id"],
        how="outer",
    )
    veri[["page_view", "add_to_cart", "satin_alma_adedi"]] = veri[
        ["page_view", "add_to_cart", "satin_alma_adedi"]
    ].fillna(0)
    veri["satin_aldi"] = veri["satin_aldi"].fillna(0).astype(int)

    # Yalnızca anlamlı negatifleri tut: en az görüntüleme veya sepete ekleme olmalı.
    veri = veri[
        (veri["satin_aldi"] == 1)
        | (veri["page_view"] > 0)
        | (veri["add_to_cart"] > 0)
    ].copy()

    # 4) Ürün özelliklerini ekle.
    urun_ozellikleri = urun_df[
        [
            "product_id", "category", "price_usd", "margin_usd", "ort_rating",
            "pozitif_oran", "oneri_skoru", "sepet_orani",
        ]
    ].copy()
    veri = veri.merge(urun_ozellikleri, on="product_id", how="left")

    # 5) Müşteri özelliklerini ekle.
    musteri_ozellikleri = rfm_df[
        ["customer_id", "recency", "frequency", "monetary", "R", "F", "M", "segment", "age"]
    ].copy()
    veri = veri.merge(musteri_ozellikleri, on="customer_id", how="left")

    # 6) Müşterinin kategori ilgisi:
    # Satın alma etiketine çok yaklaşmamak için kategori ilgisi satıştan değil,
    # görüntüleme ve sepete ekleme davranışından hesaplanır.
    davranis_kat = davranis.merge(
        urun_ozellikleri[["product_id", "category"]],
        on="product_id",
        how="left",
    )
    davranis_kat["kategori_davranis_puani"] = (
        davranis_kat["page_view"] + davranis_kat["add_to_cart"] * 3
    )
    kategori_ilgi = (
        davranis_kat.groupby(["customer_id", "category"])["kategori_davranis_puani"]
        .sum()
        .reset_index()
    )
    kategori_toplam = (
        kategori_ilgi.groupby("customer_id")["kategori_davranis_puani"]
        .sum()
        .reset_index(name="toplam_kategori_davranis_puani")
    )
    veri = veri.merge(kategori_ilgi, on=["customer_id", "category"], how="left")
    veri = veri.merge(kategori_toplam, on="customer_id", how="left")
    veri[["kategori_davranis_puani", "toplam_kategori_davranis_puani"]] = veri[
        ["kategori_davranis_puani", "toplam_kategori_davranis_puani"]
    ].fillna(0)
    veri["kategori_ilgi_orani"] = np.where(
        veri["toplam_kategori_davranis_puani"] > 0,
        veri["kategori_davranis_puani"] / veri["toplam_kategori_davranis_puani"],
        0.0,
    )

    # Eksik müşteri bilgileri: alışveriş geçmişi olmayanlarda nötr/0 değerler.
    veri["recency"] = veri["recency"].fillna(veri["recency"].max() + 365)
    for kolon in [
        "frequency", "monetary", "R", "F", "M", "age", "price_usd",
        "margin_usd", "ort_rating", "pozitif_oran",
    ]:
        veri[kolon] = veri[kolon].fillna(0)

    # Not:
    # oneri_skoru ve ürün bazlı sepet_orani genel ürün performansını özetler.
    # Kişisel modelde bunları kullanmak, ürünün geçmiş satış/sepet başarısını
    # müşteri-ürün tahminine fazla yakından taşır. Bu yüzden kişisel model daha
    # temiz bir kurgu için müşteri davranışı, RFM ve temel ürün sinyalleriyle eğitilir.
    ozellikler = [
        "page_view", "add_to_cart",
        "price_usd", "margin_usd", "ort_rating", "pozitif_oran",
        "recency", "frequency", "monetary", "R", "F", "M", "age",
        "kategori_ilgi_orani",
    ]

    X = veri[ozellikler]
    y = veri["satin_aldi"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=180,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_tr, y_tr)

    tahmin = model.predict(X_te)
    olasilik = model.predict_proba(X_te)[:, 1]
    cm = confusion_matrix(y_te, tahmin)
    fpr, tpr, _ = roc_curve(y_te, olasilik)
    auc_val = sk_auc(fpr, tpr)
    acc = accuracy_score(y_te, tahmin)

    print(f"  Kişisel model veri satırı: {len(veri):,}")
    print(f"  Pozitif satın alma: {(y==1).sum():,}  |  Negatif örnek: {(y==0).sum():,}")
    print(f"  Doğruluk: {acc:.1%}  |  ROC AUC: {auc_val:.3f}")

    veri.to_csv("outputs/kisisel_model_verisi.csv", index=False)
    pd.DataFrame({"auc": [auc_val], "dogruluk": [acc]}).to_csv(
        "outputs/kisisel_model_met.csv", index=False
    )
    np.save("outputs/kisisel_cm.npy", cm)
    np.save("outputs/kisisel_roc_fpr.npy", fpr)
    np.save("outputs/kisisel_roc_tpr.npy", tpr)

    fi = (
        pd.DataFrame({"ozellik": ozellikler, "onem": model.feature_importances_})
        .sort_values("onem", ascending=False)
    )
    fi.to_csv("outputs/kisisel_ozellik_onemi.csv", index=False)

    # add_to_cart kişisel model için çok güçlü bir davranış sinyalidir.
    # Bu yüzden ayrıca "bu alan çıkarılırsa model ne kadar değişiyor?" kontrolü yapılır.
    def model_metrikleri(model_adi, senaryo, egitilmis_model, x_test, y_test,
                         add_to_cart_dahil, aciklama):
        tahmin_lokal = egitilmis_model.predict(x_test)
        olasilik_lokal = egitilmis_model.predict_proba(x_test)[:, 1]
        tn_l, fp_l, fn_l, tp_l = confusion_matrix(
            y_test, tahmin_lokal, labels=[0, 1]
        ).ravel()
        fpr_l, tpr_l, _ = roc_curve(y_test, olasilik_lokal)

        return {
            "model": model_adi,
            "senaryo": senaryo,
            "add_to_cart_dahil": add_to_cart_dahil,
            "auc": sk_auc(fpr_l, tpr_l),
            "dogruluk": accuracy_score(y_test, tahmin_lokal),
            "precision": precision_score(y_test, tahmin_lokal, zero_division=0),
            "recall": recall_score(y_test, tahmin_lokal, zero_division=0),
            "f1": f1_score(y_test, tahmin_lokal, zero_division=0),
            "tn": tn_l,
            "fp": fp_l,
            "fn": fn_l,
            "tp": tp_l,
            "aciklama": aciklama,
        }

    ozellikler_erken = [kolon for kolon in ozellikler if kolon != "add_to_cart"]

    rf_erken = RandomForestClassifier(
        n_estimators=180,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf_erken.fit(X_tr[ozellikler_erken], y_tr)

    lojistik_tam = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    )
    lojistik_tam.fit(X_tr, y_tr)

    lojistik_erken = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    )
    lojistik_erken.fit(X_tr[ozellikler_erken], y_tr)

    karsilastirma = pd.DataFrame([
        model_metrikleri(
            "Random Forest",
            "Davranış sinyalleri dahil",
            model,
            X_te,
            y_te,
            True,
            "Ana kişisel öneri modeli. Sepete ekleme sinyalini de kullanır.",
        ),
        model_metrikleri(
            "Random Forest",
            "add_to_cart hariç kontrol",
            rf_erken,
            X_te[ozellikler_erken],
            y_te,
            False,
            "Sepete ekleme etkisini görmek için add_to_cart alanı çıkarılmıştır.",
        ),
        model_metrikleri(
            "Lojistik Regresyon",
            "Davranış sinyalleri dahil",
            lojistik_tam,
            X_te,
            y_te,
            True,
            "Karşılaştırma için daha basit ve okunabilir bir doğrusal model.",
        ),
        model_metrikleri(
            "Lojistik Regresyon",
            "add_to_cart hariç kontrol",
            lojistik_erken,
            X_te[ozellikler_erken],
            y_te,
            False,
            "Basit modelde add_to_cart çıkarıldığında performansın nasıl değiştiğini gösterir.",
        ),
    ])
    karsilastirma.to_csv("outputs/kisisel_model_karsilastirma.csv", index=False)

    rf_tam = karsilastirma[
        (karsilastirma["model"] == "Random Forest")
        & (karsilastirma["add_to_cart_dahil"])
    ].iloc[0]
    rf_kontrol = karsilastirma[
        (karsilastirma["model"] == "Random Forest")
        & (~karsilastirma["add_to_cart_dahil"])
    ].iloc[0]
    add_to_cart_onemi = float(
        fi.loc[fi["ozellik"] == "add_to_cart", "onem"].iloc[0]
    ) if "add_to_cart" in set(fi["ozellik"]) else 0.0

    pd.DataFrame([
        {
            "metrik": "add_to_cart_ozellik_onemi",
            "deger": add_to_cart_onemi,
            "aciklama": "Ana kişisel modelde add_to_cart alanının göreceli özellik önemi.",
        },
        {
            "metrik": "auc_farki_tam_eksi_kontrol",
            "deger": float(rf_tam["auc"] - rf_kontrol["auc"]),
            "aciklama": "Random Forest modelinde add_to_cart dahilken AUC farkı.",
        },
        {
            "metrik": "recall_farki_tam_eksi_kontrol",
            "deger": float(rf_tam["recall"] - rf_kontrol["recall"]),
            "aciklama": "Random Forest modelinde add_to_cart dahilken recall farkı.",
        },
        {
            "metrik": "precision_farki_tam_eksi_kontrol",
            "deger": float(rf_tam["precision"] - rf_kontrol["precision"]),
            "aciklama": "Random Forest modelinde add_to_cart dahilken precision farkı.",
        },
    ]).to_csv("outputs/kisisel_add_to_cart_kontrol.csv", index=False)

    print("  Kişisel model karşılaştırması kaydedildi.")
    print(
        "  add_to_cart önem kontrolü: "
        f"{add_to_cart_onemi:.1%} | "
        f"AUC farkı: {float(rf_tam['auc'] - rf_kontrol['auc']):.3f}"
    )

    # Dashboard hızlı çalışsın diye müşteri-ürün davranış özeti ayrıca kaydedilir.
    davranis.to_csv("outputs/musteri_urun_davranis.csv", index=False)

    dump(
        {
            "model": model,
            "ozellikler": ozellikler,
            "model_tipi": "customer_product_purchase_probability",
            "hedef": "satin_aldi",
        },
        "models/kisisel_satin_alma_model.pkl",
    )

    return model


# =============================================================================
# GRAFİKLER
# =============================================================================

OZELLIK_AD = {
    "sepet_orani":  "Sepete Ekleme Oranı",
    "ort_rating":   "Müşteri Puanı",
    "ort_yildiz":   "Duygu Skoru",
    "margin_usd":   "Ürün Marjı",
    "price_usd":    "Fiyat",
    "page_view":    "Sayfa Görüntüleme",
    "yorum_sayisi": "Yorum Sayısı",
}


def grafikleri_olustur(funnel_df, duygu_df, urun_df, rfm_df):
    """10 + 1 (EDA dahil toplam 11) grafik üretir."""
    print("\n     Grafikler oluşturuluyor...")
    grafik_stili()

    # ── 1. Dönüşüm Hunisi ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 5), facecolor=GERI)
    isimler = ["Sayfa Görüntüleme", "Sepete Ekleme", "Ödeme Başlatma", "Satın Alma"]
    rk_h    = [LACIVERT, MAVI, TURUNCU, YESIL]
    bars = ax.barh(isimler[::-1], funnel_df["event_sayisi"].values[::-1],
                   color=rk_h[::-1], height=0.52)
    mx = funnel_df["event_sayisi"].max()
    for i, bar in enumerate(bars):
        v = funnel_df["event_sayisi"].iloc[3-i]
        p = funnel_df["onceki_pct"].iloc[3-i]
        ax.text(v + mx * 0.022, bar.get_y() + bar.get_height()/2,
                f"{v:,}  (%{p:.1f})", va="center", fontsize=10.5, fontweight="bold")
    ax.set_xlim(0, mx * 1.38)
    ax.set_xlabel("Event Sayısı", fontsize=11)
    ax.set_title("Dönüşüm Hunisi", fontsize=13, fontweight="bold", pad=12)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig("outputs/charts/01_funnel.png", dpi=130, bbox_inches="tight")
    plt.close()
    print("  ✓ 01_funnel.png")

    # ── 2. Duygu Dağılımı ────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor=GERI)
    sayim = duygu_df["duygu"].value_counts()
    rkd   = {"positive": YESIL, "neutral": "#94A3B8", "negative": KIRMIZI}
    rk_l  = [rkd.get(l, "#999") for l in sayim.index]
    wedges, texts, pcts = ax1.pie(sayim.values,
        labels=[l.capitalize() for l in sayim.index], autopct="%1.1f%%",
        colors=rk_l, startangle=90, pctdistance=0.78,
        wedgeprops=dict(linewidth=2, edgecolor="white"))
    for p in pcts: p.set_fontweight("bold"); p.set_fontsize(10)
    ax1.set_title("Duygu Dağılımı", fontsize=12, fontweight="bold")
    bars2 = ax2.bar(sayim.index, sayim.values, color=rk_l, edgecolor="white", linewidth=1.5)
    for bar in bars2:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h + 60,
                 f"{int(h):,}", ha="center", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Yorum Sayısı"); ax2.spines[["top","right"]].set_visible(False)
    ax2.set_title("Yorum Sayıları", fontsize=12, fontweight="bold")
    fig.suptitle("VADER Duygu Analizi — 10.780 Müşteri Yorumu",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("outputs/charts/02_duygu.png", dpi=130, bbox_inches="tight")
    plt.close()
    print("  ✓ 02_duygu.png")

    # ── 3. Top 10 Ürün ───────────────────────────────────────────────────────
    top10 = (urun_df[urun_df["onerilir"] == 1]
             .sort_values("oneri_skoru", ascending=False).head(10))
    fig, ax = plt.subplots(figsize=(11, 6), facecolor=GERI)
    rk10 = [TURUNCU] + [MAVI] * 2 + [LACIVERT] * 7
    bars = ax.barh(range(len(top10)), top10["oneri_skoru"].values[::-1],
                   color=rk10[::-1], height=0.58)
    for bar, val in zip(bars, top10["oneri_skoru"].values[::-1]):
        ax.text(val + 0.004, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=9.5, fontweight="bold")
    ax.set_yticks(range(len(top10)))
    ax.set_yticklabels(
        [f"{10-i}. {n[:38]}" for i, n in enumerate(top10["name"].values[::-1])],
        fontsize=9.5)
    ax.set_xlabel("Öneri Skoru")
    ax.set_title("Top 10 Önerilen Ürün", fontsize=13, fontweight="bold", pad=12)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig("outputs/charts/03_top_urunler.png", dpi=130, bbox_inches="tight")
    plt.close()
    print("  ✓ 03_top_urunler.png")

    # ── 4. Skor Dağılımı ─────────────────────────────────────────────────────
    esik = urun_df["oneri_skoru"].quantile(0.70)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor=GERI)
    ax1.hist(urun_df["oneri_skoru"], bins=30, color=MAVI, alpha=0.8, edgecolor="white")
    ax1.axvline(esik, color=TURUNCU, linestyle="--", linewidth=2,
                label=f"Eşik: {esik:.3f}")
    ax1.set_xlabel("Öneri Skoru"); ax1.set_ylabel("Ürün Sayısı")
    ax1.set_title("Skor Histogramı", fontsize=12, fontweight="bold")
    ax1.legend(); ax1.spines[["top","right"]].set_visible(False)
    bp = ax2.boxplot(
        [urun_df[urun_df["onerilir"]==0]["oneri_skoru"],
         urun_df[urun_df["onerilir"]==1]["oneri_skoru"]],
        labels=["Önerilmez","Önerilir"], patch_artist=True, notch=True)
    bp["boxes"][0].set_facecolor("#FCA5A5"); bp["boxes"][1].set_facecolor("#86EFAC")
    ax2.set_ylabel("Öneri Skoru")
    ax2.set_title("Segment Karşılaştırması", fontsize=12, fontweight="bold")
    ax2.spines[["top","right"]].set_visible(False)
    fig.suptitle("Öneri Skoru Dağılımı", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("outputs/charts/04_skor.png", dpi=130, bbox_inches="tight")
    plt.close()
    print("  ✓ 04_skor.png")

    # ── 5. Kategori Geliri ───────────────────────────────────────────────────
    kat = urun_df.groupby("category")["gelir"].sum().sort_values()
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=GERI)
    rk5 = [TURUNCU if i == len(kat)-1 else MAVI for i in range(len(kat))]
    bars = ax.barh(kat.index, kat.values, color=rk5, height=0.58)
    for bar, val in zip(bars, kat.values):
        ax.text(val + 10000, bar.get_y() + bar.get_height()/2,
                f"${val/1000:.0f}K", va="center", fontsize=10, fontweight="bold")
    ax.set_xlabel("Toplam Gelir (USD)")
    ax.set_xlim(0, kat.max() * 1.22)
    ax.set_title("Kategori Bazlı Toplam Gelir", fontsize=13, fontweight="bold", pad=12)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig("outputs/charts/05_gelir.png", dpi=130, bbox_inches="tight")
    plt.close()
    print("  ✓ 05_gelir.png")

    # ── 6–8. Model Grafikleri ─────────────────────────────────────────────────
    for grafik, dosya, label in [
        ("cm", "06_cm.png", None),
        ("roc", "07_roc.png", None),
        ("fi", "08_onem.png", None),
    ]:
        try:
            if grafik == "cm":
                cm = np.load("outputs/cm.npy")
                acc = float(pd.read_csv("outputs/model_met.csv")["dogruluk"].iloc[0])
                fig, ax = plt.subplots(figsize=(6, 5), facecolor=GERI)
                im = ax.imshow(cm, cmap="Blues")
                plt.colorbar(im, ax=ax)
                for i in range(2):
                    for j in range(2):
                        c = "white" if cm[i,j] > cm.max()/2 else METIN
                        lbl = [["TN","FP"],["FN","TP"]][i][j]
                        ax.text(j, i, f"{lbl}\n{cm[i,j]:,}", ha="center",
                                va="center", fontsize=13, fontweight="bold", color=c)
                ax.set_xticks([0,1]); ax.set_yticks([0,1])
                ax.set_xticklabels(["Önerilmez","Önerilir"])
                ax.set_yticklabels(["Önerilmez","Önerilir"])
                ax.set_xlabel("Tahmin Edilen")
                ax.set_ylabel("Gerçek Değer")
                ax.set_title(f"Confusion Matrix | Doğruluk: {acc:.1%}",
                             fontsize=12, fontweight="bold", pad=12)
            elif grafik == "roc":
                fpr = np.load("outputs/roc_fpr.npy")
                tpr = np.load("outputs/roc_tpr.npy")
                auc_val = float(pd.read_csv("outputs/model_met.csv")["auc"].iloc[0])
                fig, ax = plt.subplots(figsize=(7, 5), facecolor=GERI)
                ax.plot(fpr, tpr, color=TURUNCU, lw=2.5, label=f"ROC (AUC={auc_val:.3f})")
                ax.fill_between(fpr, tpr, alpha=0.1, color=TURUNCU)
                ax.plot([0,1],[0,1], color="#94A3B8", lw=1.5, linestyle="--")
                ax.set_xlabel("Yanlış Pozitif Oranı")
                ax.set_ylabel("Doğru Pozitif Oranı")
                ax.set_title("ROC Eğrisi", fontsize=13, fontweight="bold", pad=12)
                ax.legend(fontsize=11)
                ax.spines[["top","right"]].set_visible(False)
            else:
                fi = pd.read_csv("outputs/ozellik_onemi.csv")
                fi["ad"] = fi["ozellik"].map(OZELLIK_AD).fillna(fi["ozellik"])
                fi = fi.sort_values("onem", ascending=True)
                rk_fi = [TURUNCU if i == len(fi)-1 else MAVI for i in range(len(fi))]
                fig, ax = plt.subplots(figsize=(9, 5), facecolor=GERI)
                bars = ax.barh(fi["ad"], fi["onem"], color=rk_fi, height=0.58)
                for bar, val in zip(bars, fi["onem"]):
                    ax.text(val + 0.003, bar.get_y() + bar.get_height()/2,
                            f"%{val*100:.1f}", va="center", fontsize=10, fontweight="bold")
                ax.set_xlabel("Göreceli Önem")
                ax.set_xlim(0, fi["onem"].max() * 1.28)
                ax.set_title("Özellik Önemi", fontsize=13, fontweight="bold", pad=12)
                ax.spines[["top","right"]].set_visible(False)
            plt.tight_layout()
            plt.savefig(f"outputs/charts/{dosya}", dpi=130, bbox_inches="tight")
            plt.close()
            print(f"  ✓ {dosya}")
        except FileNotFoundError:
            print(f"  ⚠ {dosya} atlandı")

    # ── 9. RFM Segment Dağılımı ──────────────────────────────────────────────
    if rfm_df is not None:
        say = rfm_df["segment"].value_counts()
        fig, ax = plt.subplots(figsize=(9, 5), facecolor=GERI)
        rk9 = [SEG_RENK.get(s, "#999") for s in say.index]
        bars = ax.barh(say.index, say.values, color=rk9, height=0.58)
        for bar, val in zip(bars, say.values):
            ax.text(val + 60, bar.get_y() + bar.get_height()/2,
                    f"{val:,}", va="center", fontsize=11, fontweight="bold")
        ax.set_xlabel("Müşteri Sayısı")
        ax.set_xlim(0, say.max() * 1.2)
        ax.set_title("RFM Müşteri Segmentleri", fontsize=13, fontweight="bold", pad=12)
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        plt.savefig("outputs/charts/09_rfm.png", dpi=130, bbox_inches="tight")
        plt.close()
        print("  ✓ 09_rfm.png")

        # ── 10. RFM Scatter ──────────────────────────────────────────────────
        ornek = rfm_df.sample(min(800, len(rfm_df)), random_state=42)
        fig, ax = plt.subplots(figsize=(10, 7), facecolor=GERI)
        for seg, grp in ornek.groupby("segment"):
            ax.scatter(grp["recency"], grp["monetary"],
                       s=grp["frequency"]*12, c=SEG_RENK.get(seg,"#999"),
                       alpha=0.55, edgecolors="white", linewidth=0.3, label=seg)
        ax.set_xlabel("Recency — Son Alışverişten Geçen Gün")
        ax.set_ylabel("Monetary — Toplam Harcama (USD)")
        ax.set_title("RFM Dağılım Haritası  (boyut = sipariş sıklığı)",
                     fontsize=12, fontweight="bold", pad=12)
        ax.legend(title="Segment", fontsize=9, framealpha=0.7)
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        plt.savefig("outputs/charts/10_rfm_scatter.png", dpi=130, bbox_inches="tight")
        plt.close()
        print("  ✓ 10_rfm_scatter.png")

    print("  Grafikler → outputs/charts/")


# =============================================================================
# ANA FONKSİYON
# =============================================================================

def main():
    for d in ["outputs", "outputs/charts", "models"]:
        Path(d).mkdir(exist_ok=True, parents=True)

    # Pipeline
    events, products, reviews, sessions, order_items, customers, orders = veri_yukle()
    ev_temiz, musteri_oturum = veri_on_isleme(events, products, reviews,
                                               orders, order_items, customers, sessions)
    eda_ozet    = keşifsel_veri_analizi(events, ev_temiz, products, reviews,
                                        orders, order_items, customers, sessions)
    funnel_df   = funnel_analizi(events)
    urun_ham    = ozellik_muhendisligi(ev_temiz, products, reviews, order_items)
    duygu_df    = duygu_analizi(reviews)
    urun_df     = onerme_skoru(urun_ham, duygu_df)
    model_egit(urun_df)
    rfm_df, musteri_kat_df = musteri_segmentasyonu(customers, orders, order_items, products)
    kisisel_satin_alma_modeli(
        events, sessions, orders, order_items, customers,
        urun_df, rfm_df, musteri_kat_df,
    )
    grafikleri_olustur(funnel_df, duygu_df, urun_df, rfm_df)

    sure = (datetime.now() - _baslama).total_seconds()
    print(f"""
{"=" * 58}
  Tamamlandı! ({sure:.0f} saniye)
{"=" * 58}

outputs/
  veri_ozet.csv    eda_ozet.csv     funnel.csv
  urun_analitik.csv  duygu.csv      urun_skorlar.csv
  top50.csv        model_met.csv    ozellik_onemi.csv
  kisisel_model_met.csv  kisisel_model_verisi.csv
  rfm.csv          musteri_kat.csv
  charts/ (11 grafik — EDA dahil)

models/model.pkl

  streamlit run dashboard.py
""")


if __name__ == "__main__":
    main()
