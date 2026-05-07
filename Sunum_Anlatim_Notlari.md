# ShopLens Sunum Anlatım Notları

Bu dosyayı sunuma hazırlanırken kısa prova notu olarak kullanıyorum. Slaytlarda yazan her şeyi birebir okumak yerine, ana fikri kendi cümlelerimle anlatmayı tercih ettim.

## 1. Açılış

Bu projede bir e-ticaret veri seti üzerinden ürün öneri sistemi kurdum. Amacım sadece en çok satan ürünleri göstermek değildi. Kullanıcının davranış geçmişini, ürün performansını ve yorumlardan gelen duygu bilgisini birlikte değerlendirerek kişiye özel satın alma olasılığı üretmek istedim.

Kısa anlatım:

> ShopLens, müşteri davranışı, ürün performansı ve yorum duygu bilgisini birleştirerek seçilen müşteri için satın alma olasılığı yüksek ürünleri öneren bir e-ticaret analiz sistemidir.

## 2. Problem

E-ticaret sitelerinde ürün sayısı arttıkça kullanıcıya doğru ürünü göstermek zorlaşıyor. Her müşteriye aynı ürünleri önermek kişiselleştirme sağlamıyor. Bu projede bu problemi veri analizi ve makine öğrenmesiyle ele aldım.

Söylemek istediğim ana cümle:

> Benim hedefim, ürünleri genel popülerliğe göre değil, müşterinin davranışına göre sıralayan bir öneri yapısı kurmaktı.

## 3. Veri Seti

Kaggle'daki E-commerce Transactions + Clickstream veri setini kullandım. Veri seti sentetik ama proje için uygun; çünkü gerçek bir e-ticaret sisteminde karşılaşılabilecek farklı tabloları birlikte içeriyor.

Kullandığım tablolar:

- `events.csv`: görüntüleme, sepete ekleme, checkout ve purchase eventleri
- `sessions.csv`: eventleri müşteriyle eşleştirmek için oturum bilgileri
- `customers.csv`: müşteri profilleri
- `products.csv`: ürün bilgileri
- `orders.csv`: sipariş başlıkları
- `order_items.csv`: ürün bazlı satış adedi ve gelir
- `reviews.csv`: rating ve yorum metinleri

Burada özellikle şunu vurgulayacağım:

> Ham veriyi doğrudan modele vermedim. Önce iş problemine uygun iki analitik tablo oluşturdum.

## 4. Veri Birleştirme Hikâyesi

Bütün tabloları tek seferde birleştirmek doğru olmazdı; çünkü her tablo farklı seviyede bilgi tutuyor. `events` event seviyesinde, `orders` sipariş seviyesinde, `order_items` sipariş kalemi seviyesinde, `products` ürün seviyesinde, `customers` müşteri seviyesinde.

Bu yüzden iki ana veri seti oluşturdum:

1. `urun_analitik.csv`: Her satır bir ürünü temsil ediyor.
2. `kisisel_model_verisi.csv`: Her satır bir müşteri-ürün eşleşmesini temsil ediyor.

Kısa anlatım:

> Ürün bazlı tablo ürünün genel performansını ölçüyor. Kişisel model tablosu ise belirli bir müşterinin belirli bir ürünü satın alma olasılığını tahmin etmek için kullanılıyor.

## 5. Eksik Değerler

Eksik değerleri rastgele silmedim. Önce neden eksik olduklarını yorumladım.

Örneğin `events` tablosunda checkout ve purchase satırlarında `product_id` boş. Bu bir veri hatası gibi görünse de aslında bu eventler ürün seviyesinde değil, oturum veya sipariş seviyesinde tutulmuş. Bu yüzden ürün bazlı gerçek satış hesabında `events.purchase` yerine `order_items.quantity` alanını kullandım.

Sunumda kullanacağım net cümle:

> Ürün bazlı satış adedini `events` tablosundan değil, sipariş kalemlerini tutan `order_items.quantity` alanından hesapladım.

## 6. Keşifsel Veri Analizi

EDA aşamasında veri setinin genel yapısını, event dağılımını, dönüşüm hunisini, yorum dağılımını ve ürün performanslarını inceledim.

Funnel tarafındaki ana bulgu:

> En büyük kayıp sayfa görüntülemeden sepete eklemeye geçişte oluşuyor. Checkout aşamasına gelen kullanıcıların satın alma oranı daha yüksek.

Bu bulgu kişisel öneri sistemini destekliyor; çünkü doğru ürün gösterilirse sepete ekleme oranı artabilir.

## 7. Duygu Analizi

Duygu analizi için VADER + rating hibrit yaklaşımı kullandım. Yalnızca metin skoruna göre karar vermedim; rating bilgisini de dahil ettim.

Bunun nedeni şu:

> Veri setindeki yorumlar kısa ve tekrarlı olduğu için sadece metin skoru bazı yorumları olduğundan daha pozitif gösterebilir. Rating bilgisini ekleyerek daha dengeli bir karar verdim.

Kullandığım sınıflar:

- Pozitif
- Nötr
- Negatif

## 8. Ürün Öneri Skoru

Ürün öneri skoru genel ürün performansını ölçmek için oluşturduğum ağırlıklı bir skordur. Bu skor kişisel öneri değildir; kişisel modelde ürüne ait yardımcı bir sinyal olarak kullanılır.

Skorda kullandığım sinyaller:

- Satış oranı
- Ortalama rating
- Sepete ekleme oranı
- Pozitif yorum oranı
- Gelir
- Marj

Kısa anlatım:

> Önce ürünlerin genel gücünü ölçtüm, sonra bu ürün sinyallerini kişisel satın alma modeline taşıdım.

## 9. Kişiye Özel Satın Alma Modeli

Asıl model müşteri-ürün seviyesinde kuruldu. Her satır bir müşteri ve bir ürün eşleşmesini gösteriyor.

Hedef değişken:

```text
satin_aldi
```

Anlamı:

```text
1 = müşteri ürünü satın aldı
0 = müşteri ürünü satın almadı
```

Negatif örnekleri oluştururken hiç görülmeyen ürünleri doğrudan “satın almadı” saymadım. Daha anlamlı olması için müşterinin görüntülediği veya sepete eklediği ama satın almadığı ürünleri negatif örnek olarak kullandım.

Önemli cümle:

> Modelin sorduğu soru “Bu ürün genel olarak iyi mi?” değil, “Bu müşteri bu ürünü satın alır mı?” sorusudur.

## 10. Neden Random Forest?

Bu problem bir regresyon problemi değil, sınıflandırma problemidir. Çünkü model sayısal satış adedi tahmin etmiyor; satın aldı veya satın almadı kararını tahmin ediyor.

Random Forest kullanmamın nedenleri:

- Karar ağacı mantığı anlaşılırdır.
- Doğrusal olmayan ilişkileri yakalayabilir.
- Sınıf dengesizliğini `class_weight="balanced"` ile yönetebilir.
- Özellik önemlerini gösterdiği için yorumlanabilir.

## 11. Model Sonuçları

Kişisel satın alma modelinde AUC değeri 0.955 çıktı. Bu değer modelin satın alma olasılığı yüksek ve düşük müşteri-ürün çiftlerini ayırabildiğini gösteriyor.

Bunu abartmadan şöyle anlatacağım:

> Sonuçlar proje verisi üzerinde güçlü görünüyor. Gerçek bir e-ticaret sisteminde bu modelin canlı A/B test ile ayrıca doğrulanması gerekir.

## 12. RFM Segmentasyonu

RFM analiziyle sipariş geçmişi olan müşterileri segmentlere ayırdım.

RFM anlamı:

- Recency: Son alışverişten kaç gün geçti?
- Frequency: Kaç sipariş verdi?
- Monetary: Toplam ne kadar harcadı?

Burada dikkat etmem gereken nokta:

> RFM bir kümeleme modeli değil, iş kurallarına dayalı müşteri skorlamasıdır.

## 13. Dashboard

Dashboard projenin çalışan arayüzüdür. Burada genel ürün performansı, dönüşüm hunisi, model sonuçları, duygu analizi ve kişisel öneriler birlikte görülebilir.

Müşteri sekmesinde seçilen müşteri için model her aday ürüne satın alma olasılığı verir ve en yüksek olasılıklı ürünler listelenir.

## 14. Değerlendirme Kriterleri

Projede veri temini, ön işleme, EDA, modelleme, görselleştirme, kod kalitesi ve raporlama başlıklarının her biri için karşılığı olan bir çıktı hazırladım.

Kısa anlatım:

> Projede sadece model eğitmedim; veriyi hazırladım, analiz ettim, yorumladım, dashboard ile görselleştirdim ve sunumla raporladım.

## 15. Kapanış

Kapanışta şu cümleyi kullanabilirim:

> Bu projede ham e-ticaret verisini analiz ederek ürün performansını, müşteri davranışını ve yorum duygu bilgisini bir araya getirdim. Son aşamada müşteri-ürün bazlı satın alma olasılığı tahminiyle kişiye özel ürün öneri sistemi oluşturdum.
