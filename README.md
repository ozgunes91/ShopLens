# ShopLens — E-Ticaret Davranış Analizi ve Akıllı Ürün Öneri Sistemi

**Hazırlayan:** Özge Güneş  
**Ders / Proje:** İSMEK Yapay Zekâ ve Veri Bilimi  
**Konu:** E-ticaret davranış analizi, ürün öneri skoru, duygu analizi, müşteri segmentasyonu ve kişiye özel satın alma tahmini

## Projenin Amacı

Bu projede bir e-ticaret platformunda müşterilerin ürünlerle olan etkileşimlerini analiz ettim. Amacım yalnızca en çok satan ürünleri listelemek değildi. Ürün görüntüleme, sepete ekleme, sipariş, ürün bilgisi, müşteri bilgisi ve yorum verilerini bir araya getirerek belirli bir müşterinin hangi ürünü satın alma olasılığının daha yüksek olduğunu tahmin eden bir öneri sistemi kurdum.

Dashboard tarafında hem genel ürün performansını hem de seçilen müşteri için kişisel ürün önerilerini tek ekranda takip edilebilir hale getirdim.

## Veri Seti Kaynağı

Projede Kaggle üzerindeki **E-commerce Transactions + Clickstream** veri setini kullandım.

Kaynak: https://www.kaggle.com/datasets/wafaaelhusseini/e-commerce-transactions-clickstream

Veri seti sentetik olarak üretilmiştir. Buna rağmen müşteri, ürün, oturum, clickstream eventleri, sipariş, sipariş kalemi ve yorum tablolarını birlikte içerdiği için e-ticaret analitiği projesi için uygundur.

## Klasör Yapısı

```text
data/                         Ham veri dosyaları
outputs/                      Analiz çıktıları, skor tabloları ve model metrikleri
outputs/charts/               Dashboard ve sunumda kullanılan grafikler
models/                       Eğitilmiş model dosyaları
pipeline.py                    Veri hazırlama, analiz ve modelleme pipeline'ı
dashboard.py                  Streamlit dashboard
ShopLens_Sunum_Final.pptx     Final proje sunumu
requirements.txt              Gerekli Python kütüphaneleri
```

## Çalıştırma

Gerekli kütüphaneleri yüklemek için:

```bash
pip install -r requirements.txt
```

Analizleri, skor tablolarını, grafikleri ve model dosyalarını yeniden üretmek için:

```bash
python pipeline.py
```

Dashboard'u açmak için:

```bash
streamlit run dashboard.py
```

Yerel dashboard adresi:

```text
http://localhost:8501
```

## Kullandığım Veri Tabloları

Projede 7 ana tablo kullandım:

| Tablo | Kullanım amacı |
|---|---|
| `events.csv` | Sayfa görüntüleme, sepete ekleme, checkout ve purchase eventleri |
| `sessions.csv` | Eventleri müşteri ile eşleştirmek |
| `customers.csv` | Müşteri profil bilgileri |
| `products.csv` | Ürün adı, kategori, fiyat, maliyet ve marj bilgileri |
| `orders.csv` | Sipariş başlığı, müşteri ve sipariş zamanı |
| `order_items.csv` | Ürün bazlı gerçek satış adedi ve gelir |
| `reviews.csv` | Rating ve yorum metinleri |

## Veri Birleştirme Mantığı

Bütün tabloları tek seferde büyük bir tabloya çevirmedim. Çünkü her tablo farklı seviyede bilgi tutuyor. Bunun yerine iki ana analitik tablo oluşturdum:

1. **Ürün bazlı analitik tablo:** `outputs/urun_analitik.csv`  
   Her satır bir ürünü temsil eder. Ürün görüntüleme, sepete ekleme, satış adedi, gelir, yorum sayısı ve rating gibi alanlar bu seviyede özetlenir.

2. **Kişisel model tablosu:** `outputs/kisisel_model_verisi.csv`  
   Her satır bir müşteri-ürün eşleşmesini temsil eder. Modelin sorusu şudur: “Bu müşteri bu ürünü satın alır mı?”

Ürün bazlı tabloda hesaplanan `oneri_skoru`, `pozitif_oran`, `sepet_orani`, `ort_rating` gibi ürün sinyalleri daha sonra `product_id` üzerinden kişisel model tablosuna eklendi.

## Eksik Verilerle Nasıl Baş Ettim?

Eksik değerleri doğrudan silmedim. Önce eksikliğin veri yapısından mı, yoksa gerçekten eksik bilgiden mi kaynaklandığını kontrol ettim.

| Durum | Uygulanan işlem |
|---|---|
| `events.product_id` checkout ve purchase satırlarında boş | Ürün bazlı satış için `events` değil, `order_items.quantity` kullanıldı |
| Üründe yorum yok | Yorum sayısı ve duygu oranları uygun 0/nötr değerlerle dolduruldu |
| Üründe görüntüleme veya sepete ekleme yok | İlgili event sayıları 0 kabul edildi |
| Oran hesaplarında payda 0 olabilir | `np.where` ile sıfıra bölme engellendi |
| Sipariş geçmişi olmayan müşteri | RFM segmentasyonuna dahil edilmedi |

Bu yaklaşım modelin hatalı veya uydurma sinyaller öğrenmesini engelledi.

## Analiz ve Modelleme Adımları

1. Veriler yüklendi ve temel kalite kontrolleri yapıldı.
2. Eventler ürün seviyesinde özetlendi.
3. Gerçek satış adedi ve gelir `order_items.quantity` ve `line_total_usd` alanlarından hesaplandı.
4. Yorumlar VADER + rating hibrit yöntemiyle pozitif, nötr ve negatif olarak etiketlendi.
5. Ürünler için ağırlıklı `oneri_skoru` hesaplandı.
6. Skor dağılımının üst %30'undaki ürünler `onerilir=1` olarak işaretlendi.
7. Müşteriler RFM yöntemiyle segmentlere ayrıldı.
8. Müşteri-ürün seviyesinde kişiye özel satın alma tahmin modeli kuruldu.
9. Model sonuçları dashboard ve sunumda yorumlanabilir grafiklerle gösterildi.

## Duygu Analizi

Duygu analizi için VADER kullandım. Ancak yalnızca metin skoruna göre karar vermedim; yorumdaki rating bilgisini de dahil ettim. Bunun nedeni, veri setindeki yorumların kısa ve tekrarlı olmasıdır.

Kullandığım temel karar mantığı:

```python
if compound > 0.3 and rating >= 4:
    etiket = "positive"
elif compound < -0.1 or rating <= 2:
    etiket = "negative"
else:
    etiket = "neutral"
```

## Öneri Skoru

Genel ürün öneri skoru bir iş kuralı olarak tasarlandı. Bu skor kişisel önerinin kendisi değildir; ürünün genel performansını gösteren yardımcı bir sinyaldir.

Skorda kullanılan başlıca alanlar:

- Satış oranı
- Ortalama rating
- Sepete ekleme oranı
- Pozitif yorum oranı
- Gelir
- Marj

## Modelleme

Bu projedeki ana problem **ikili sınıflandırma** problemidir. Model satış adedi gibi sürekli bir değer tahmin etmez. Bunun yerine müşteri-ürün çifti için satın alma durumunu tahmin eder:

```text
1 = satın aldı
0 = satın almadı
```

Kişisel satın alma tahmini için Random Forest kullandım. Bu model, karar ağaçları mantığıyla çalıştığı için sonuçları yorumlamak kolaydır. Ayrıca doğrusal olmayan ilişkileri yakalayabilir ve özellik önemlerini gösterebilir.

Kişisel model sonuçları:

```text
AUC: 0.955
Doğruluk: 0.883
```

Bu sonuçlar modelin satın alma olasılığı yüksek ve düşük müşteri-ürün çiftlerini ayırabildiğini gösterir. Gerçek bir iş ortamında bu tür bir modelin ayrıca canlı A/B test ile doğrulanması gerekir.

## Dashboard

Dashboard Streamlit ile hazırlandı. Ana bölümler:

- Genel özet ve dönüşüm hunisi
- Keşifsel veri analizi
- Genel ürün önerileri
- Model değerlendirmesi
- Duygu analizi
- Müşteri segmentasyonu ve kişisel öneri
- İstatistiksel analizler

Canlıya alma alanı:

```text
Streamlit linki: ________________________________
```

## Final Dosyaları

Teslim için ana dosyalar:

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

## Canlı Dashboard Yayını

Dashboard Streamlit Community Cloud üzerinden yayınlanacak şekilde hazırlanmıştır.

Yayın ayarları:

```text
Repository root: ShopLens
Main file path: dashboard.py
Python dependencies: requirements.txt
Streamlit config: .streamlit/config.toml
```

Canlı link oluşturulduktan sonra sunumdaki `Canlı Dashboard` alanına eklenebilir.

