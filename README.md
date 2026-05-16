# ShopLens — E-Ticaret Davranış Analizi ve Akıllı Ürün Öneri Sistemi

**Hazırlayan:** Özge Güneş  
**Ders:** İSMEK Yapay Zekâ ve Veri Bilimi  
**Tarih:** Mayıs 2026

## Projenin Amacı

Bu projede bir e-ticaret platformunda müşteri-ürün etkileşimlerini analiz ettim. Amacım sadece en çok satan ürünleri listelemek değil, ürün görüntüleme, sepete ekleme, sipariş, ürün bilgisi ve yorum verilerini birleştirerek belirli bir müşteri için ürünleri öneri önceliğine göre sıralayan bir sistem kurmaktı.

Dashboard tarafında hem genel ürün performansı hem de seçilen müşteri için kişisel ürün önerileri tek ekranda görünür.

## Canlı Dashboard

https://shoplens.streamlit.app/

## Veri Seti

**Kaynak:** Kaggle — [E-commerce Transactions + Clickstream](https://www.kaggle.com/datasets/wafaaelhusseini/e-commerce-transactions-clickstream)

Sentetik bir veri seti. Müşteri, ürün, oturum, clickstream eventleri, sipariş, sipariş kalemi ve yorum tablolarını birlikte içerdiği için e-ticaret analitiği için uygun.

| Tablo | Satır | Kullanım |
|---|---:|---|
| events.csv | 760.958 | Sayfa görüntüleme, sepete ekleme, checkout, purchase |
| sessions.csv | 120.000 | Eventleri müşteriye bağlamak |
| customers.csv | 20.000 | Müşteri profili |
| orders.csv | 33.580 | Sipariş başlığı |
| order_items.csv | 59.163 | Ürün bazlı gerçek satış adedi ve gelir |
| products.csv | 1.197 | Ürün bilgisi (fiyat, kategori, marj) |
| reviews.csv | 10.780 | Rating ve yorum metinleri |

## Klasör Yapısı

```text
data/                    Ham CSV dosyaları
outputs/                 Analiz çıktıları (CSV, model metrikleri)
outputs/charts/          Sunum ve dashboard grafikleri
models/                  Eğitilmiş model dosyaları
pipeline.py              Veri hazırlama, analiz ve modelleme
dashboard.py             Streamlit dashboard
requirements.txt         Bağımlılıklar
ShopLens_Sunum_Final.pptx  Sunum
```

## Çalıştırma

```bash
pip install -r requirements.txt
python pipeline.py          # Tüm analizleri ve modelleri yeniden üretir
streamlit run dashboard.py  # Dashboard'u açar (http://localhost:8501)
```

## Veri Birleştirme Mantığı

Bütün tabloları tek seferde birleştirmek yerine iki analitik tablo oluşturdum çünkü her tablo farklı seviyede bilgi tutuyor:

1. **Ürün bazlı tablo** — `outputs/urun_analitik.csv`  
   Her satır bir ürün. Görüntüleme, sepete ekleme, satış adedi, gelir, yorum sayısı, rating ve duygu oranı bu seviyede.

2. **Kişisel model tablosu** — `outputs/kisisel_model_verisi.csv`  
   Her satır bir müşteri-ürün eşleşmesi. Modelin sorusu: "Bu müşteri için hangi ürünler daha öncelikli önerilmeli?"

Ürün sinyalleri `product_id` üzerinden kişisel model tablosuna eklendi. Ancak hedeften doğrudan türeyen `oneri_skoru` ve ürün bazlı `satis_orani`, kişisel modelde özellik olarak kullanılmadı (veri sızıntısı önlemi).

## Eksik Değerlerle Nasıl Baş Ettim?

Eksik değerleri doğrudan silmedim. Önce eksikliğin yapısal mı yoksa gerçek bilgi eksikliği mi olduğunu kontrol ettim.

| Durum | Karar |
|---|---|
| `events.product_id` checkout/purchase'ta boş | Yapısal eksik. Satış adedi `order_items.quantity`'den alındı |
| Üründe yorum yok | yorum_sayisi=0, pozitif_oran=0 |
| Üründe görüntüleme yok | İlgili event sayıları 0 |
| Oran paydası 0 | `np.where` ile koşullu hesaplama |
| Sipariş geçmişi olmayan müşteri | RFM segmentasyonu dışı |

## Duygu Analizi

VADER kullandım, ama metin skoruna ek olarak rating bilgisini de hibrit kuralın içine kattım. Sebebi: veri setindeki yorumlar kısa ve tekrarlı, sadece metin skoruna güvenmek yetersiz kalıyor.

```python
if compound > 0.3 and rating >= 4:
    etiket = "positive"
elif compound < -0.1 or rating <= 2:
    etiket = "negative"
else:
    etiket = "neutral"
```

**Sonuç:** 7.620 pozitif (%70.7) · 1.980 nötr (%18.4) · 1.180 negatif (%10.9)

## Genel Ürün Öneri Skoru

Bir iş kuralı olarak tasarlandı. Ürünün genel performansını gösteren yardımcı sinyal — kişisel öneri yerine geçmez.

Ağırlıklı formül:

```text
oneri_skoru = 
    %28 satış oranı
  + %20 ortalama rating
  + %18 sepete ekleme oranı
  + %18 pozitif yorum oranı
  + %8  gelir
  + %8  marj
```

Her değişken min-max normalize edilir. Skorun üst %30'luk dilimi → `onerilir = 1` (**359 ürün**).

## Modelleme

İki ayrı ikili sınıflandırma modeli var. İki model farklı soruyu sorduğu için sonuçlar ayrı yorumlanmalı.

| Model | Veri seviyesi | Soru | Kullanıldığı yer |
|---|---|---|---|
| Genel ürün modeli | Ürün (1.197 satır) | Bu ürün genel olarak önerilmeli mi? | Genel öneri listesi |
| Kişiye özel öneri sıralama modeli | Müşteri × ürün (529.593 satır) | Bu müşteri için hangi ürünler öncelikli? | Kişisel ürün önerisi |

Her ikisinde de **Random Forest Classifier** kullandım. Karar ağacı mantığı açıklanabilir; ayrıca özellik önemlerini görebiliyorum.

### Model Performansları

| Model | AUC | Doğruluk | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Genel ürün modeli | 0.913 | %84.7 | %73.4 | %75.6 | 0.750 |
| Kişiye özel model | 0.954 | %88.3 | %48.5 | %99.99 | 0.653 |

### Veri Sızıntısı Önlemi

- **Genel ürün modelinde:** Hedef değişken (`onerilir`) öneri skorundan üretildiği için, öneri skorunun girdisi olan `satis_orani` modele özellik olarak verilmedi.
- **Kişisel modelde:** `satin_aldi` hedefi sipariş verisinden üretildi. Hedefi doğrudan ele veren `oneri_skoru` ve ürün bazlı `sepet_orani` gibi alanlar modele konmadı. Eğitim ve test bölünmesi `stratify` ile sınıf oranları korunarak yapıldı.

### Kişisel Model Hakkında Önemli Bir Not

Kişisel modelde **recall %99.99 ve precision %48.5**. Recall yüksek olduğu için test verisinde satın alma yapan müşteri-ürün çiftleri neredeyse hiç kaçırılmıyor. Precision daha düşük olduğu için aday liste geniş tutuluyor.

Bu nedenle modeli **kesin satış tahmini** değil, müşteriye gösterilecek ürünleri **öncelik sırasına koyan bir aday sıralayıcı** olarak yorumladım. Dashboard'da çıktılar "öneri önceliği" başlığıyla gösteriliyor.

#### Sepete Ekleme Sinyali Kontrolü

Modelin en güçlü özelliği olan `add_to_cart` çıkarıldığında metriklerin nasıl değiştiğini test ettim:

| Senaryo | AUC | Recall | Precision |
|---|---:|---:|---:|
| add_to_cart **dahil** | 0.954 | %99.99 | %48.5 |
| add_to_cart **hariç** | 0.752 | %85.6 | %18.5 |

`add_to_cart` özelliğinin önem skoru **%80.7**. Model satın alma sinyalini büyük ölçüde sepete ekleme davranışından öğreniyor — bu beklenen bir durum (sepete ekleme, satın almanın en güçlü erken göstergelerinden biri). Veri sızıntısı değil, ama bu yüzden model **kesin tahmin yerine sıralama amacıyla** kullanılmalı. Gerçek mağaza ortamında zaman bazlı doğrulama ve A/B test ile yeniden sınanması gerekir.

## Müşteri Segmentasyonu (RFM)

Sipariş geçmişi olan **16.268 müşteri** segmentlendi. 3.732 müşterinin sipariş geçmişi olmadığı için RFM'e dahil edilmedi.

| Segment | Kural | Müşteri |
|---|---|---:|
| Şampiyon | R≥4 ve F≥4 | 2.923 |
| Sadık | R≥3 ve F≥3 | 4.388 |
| Yeni | R≥4 ve F≤2 | 1.442 |
| Potansiyel | Orta R/F | 1.014 |
| Risk Altında | R≤2 ve F≥3 | 2.734 |
| Kayıp | R≤2 ve F≤2 | 3.767 |

## Dashboard

Streamlit ile hazırlandı. Ana sekmeler:

1. **Yönetici Özeti** — KPI'lar, dönüşüm hunisi
2. **Veri Keşfi** — Tablo boyutları, eksik değer analizi, EDA grafikleri
3. **Ürün Davranışı** — Fiyat × satış, rating × skor scatter analizleri
4. **Genel Ürün Önerileri** — Filtrelenebilir top ürün listesi
5. **Modelleme** — İki modelin metrik karşılaştırması, CM, ROC, özellik önemi
6. **Yorum Duygusu** — VADER + rating sonuçları, örnek yorumlar
7. **Kişisel Ürün Önerisi** — Müşteri seç → kişisel öneri sıralaması
8. **Sistem Özeti** — Genel KPI dökümü ve kategori karşılaştırma

## Geliştirme Önerileri

- Gerçek mağaza verisiyle zaman bazlı doğrulama ve A/B test
- RFM yerine KMeans tabanlı segmentasyon denemesi
- Çok dilli yorumlar için VADER yerine BERT tabanlı model
- Soğuk başlangıç (yeni müşteri) için içerik bazlı öneri katmanı
- Model izleme paneli ve veri kayması (drift) kontrolü

## Final Teslim Dosyaları

```text
README.md
requirements.txt
pipeline.py
dashboard.py
ShopLens_Sunum_Final.pptx
data/
outputs/
models/
```
