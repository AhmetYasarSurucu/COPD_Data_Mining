1.	VERİ KÜMESİ HAKKINDA
1.1 VERİ KÜMESİNDEKİ DEĞİŞKENLER VE ÖZELLİKLERİ

Demografik Bilgiler
•	Hasta Numarası (hasta_no): Hasta kimlik numarası (anonimleştirilmiş).
•	Yaş (yas): Hastanın yaşı.
•	Cinsiyet (cinsiyet): Hastanın cinsiyeti (1: Erkek, 2: Kadın).
•	Eğitim Düzeyi (egitim_duzeyi): Hastanın eğitim düzeyi.
•	Meslek (meslek): Hastanın mesleği.

Yaşam Tarzı ve Alışkanlıklar
•	Sigara Kullanımı Durumu (sigara_kullanimi):
•	1: Hiç kullanmamış,
•	2: Kullanmış ama bırakmış,
•	3: Aktif kullanıyor.
•	Sigara İçme Süresi (sigara_birakan_ne_kadar_gun_icmis): Sigara bırakanların toplam sigara içme süresi (gün).
•	Sigara İçme Adeti (sigara_birakan_gunde_kac_adet_icmis): Sigara bırakanların günde ortalama içtikleri sigara sayısı.
•	Sigara Bırakma Süresi (ne_zaman_birakmis_gun): Sigara bırakılmasından bu yana geçen süre (gün).

Tıbbi Durumlar ve Aile Öyküsü
•	Ailede KOAH veya Astım Tanısı (ailede_koah_veya_astim_tanili_hasta_var_mi):
•	1: Evet,
•	0: Hayır.
•	Hasta Aile Bireyleri (varsa_kimde_anne/baba/kardeş/diger): Eğer varsa tanı almış aile bireyleri.
Akciğer Fonksiyon Testleri
•	FEV1 Değeri (FEV1): Zorlu ekspiratuvar volüm (bir saniyede çıkartılan hava miktarı).
•	FEV1 Yüzdesi (FEV1_yuzde): FEV1 değerinin normal popülasyona göre yüzdesi.
•	PEF Değeri (PEF): Pik ekspiratuvar akış hızı.
•	PEF Yüzdesi (PEF_yuzde): PEF değerinin normal popülasyona göre yüzdesi.
•	FEV1/FVC Oranı (FEV1_FVC_sDeğeri): FEV1'in FVC'ye (zorlu vital kapasite) oranı.

	
2.	VERİ ÖN İŞLEME 
2.1 EKSİK GÖZLEM
 ![image](https://github.com/user-attachments/assets/10d4ef08-8854-4d65-afca-a8126d60bf5b)

Sigara Kullanımı ile İlgili Değişkenler:
	sigara_bırakan_ne_kadar_gun_icmis, sigara_bırakan_gunde_kac_adet_icmis ve ne_zaman_birakmis_gun sütunlarında %55,62 oranında eksiklik mevcut.
	sigara_devam_eden_gunde_kac_adet_iciyor sütununda eksiklik oranı %78,31 ile en yüksek değere sahiptir. 
Solunum Fonksiyon Testleri:
	FEV1 ve PEF değişkenlerinde %32,93 oranında veri kaybı bulunmakta. 
Tanı Süresi Değişkenleri:
	tani_suresi_yil ve tani_suresi_ay değişkenlerinde eksik değer oranı %1.20 gibi düşük bir düzeydedir.
Ailede Benzer Hastalık Varlığı:
	varsa_kimde_anne, varsa_kimde_baba, varsa_kimde_kardes ve varsa_kimde_diger değişkenlerinde eksiklik oranları %0.20 ve %0,60 değerlere sahiptir.

2.3 EKSİK GÖZLEM DOLDURMA
Sigara Kullanımı ile İlgili Değişkenler:
Sigara bırakan ne kadar gün içmiş, sigara bırakan günde kaç adet içmiş ve ne zaman bırakmış gün bu sütunlardaki eksik değerler, halen sigara içen veya hiç sigara kullanmamış bireyler için doğal olarak oluşmuştur. 
	Bu durumda eksik değerler, 0 değeri ile doldurulmuştur.
sigara_devam_eden_gunde_kac_adet_iciyor bu sütundaki eksik değerler, sigarayı bırakmış ya da hiç sigara kullanmamış bireyler için gözlemlenmiştir.
	Benzer mantıkla bu boş değerler, 0 değeri ile doldurulmuştur.
Tanı Süresi Değişkenleri:
En çok tekrar eden değerler:
•	tanı_suresi_yil: 1.00 (67 kez tekrar)
•	tanı_suresi_ay: 0.00 (456 kez tekrar)
Gözlemler sonucunda, tanı_suresi_yil 1 olan bireylerde tanı_suresi_ay değerinin 0 olduğu belirlenmiştir.
	Bu tutarlılığa dayanarak eksik değerler en sık tekrar eden değerlerle doldurulmuştur.

Ailede Benzer Hastalık Varlığı:
varsa_kimde_anne, varsa_kimde_baba, varsa_kimde_kardes ve varsa_kimde_diger sütunlarındaki eksik değerler, ailede_koah_veya_astim_tanili_hasta_var_mi sütununa göre tutarlı bir şekilde doldurulmuştur.
	Eğer ailede KOAH veya astım tanılı hasta varsa ancak sadece belirli bir birey (örneğin anne) belirtilmişse, diğer sütunlardaki boş değerler 0 ile doldurulmuştur.
	Eksik verilerin sadece gerçekçi durumları yansıtacak şekilde doldurulmasını sağlamıştır.


2.4 MAKİNE ÖĞRENMELERİ İLE FEV1 DEĞERLERİ TAHMİN MODELLEMESİ

2.4.1 En iyi performansı sağlayan model seçimi
Eksik değerleri doldurmak için MARS (Multivariate Adaptive Regression Splines) ve Random Forest modelleri kullanılmış ve 10 katlı çapraz doğrulama ile performansları karşılaştırılmıştır. En iyi performansı sağlayan model seçilerek, eksik FEV1 değerleri tahmin edilmiştir.
 ![image](https://github.com/user-attachments/assets/b0368415-700e-4f4b-8a66-14b7fa572056)

Çapraz doğrulama sonuçlarına göre MARS modeli, Random Forest modeline kıyasla daha iyi performans göstermiştir. MARS modeli R² = 0.9714 maksimum değeriyle doğruluk açısından öne çıkarken, Random Forest modeli R² = 0.9460 ile ikinci sırada yer almıştır. Ayrıca MARS modeli, daha düşük hata oranlarına sahip olup MAE = 0.1224 ve RMSE = 0.1639 değerleriyle tahminlerde daha başarılı sonuçlar elde edilmiştir. Random Forest modeli ise göreceli olarak daha yüksek hatalar üretmiştir. Bu nedenle eksik FEV1 değerlerini doldurmak için MARS modeli en iyi seçim olarak belirlenmiştir.
2.4.2 MARS modeli için hiperparametre ayarlama
MARS modeli için hiperparametre ayarlaması degree (1-3) ve nprune (2'den 100'e, 5'er artışlarla) değerleri üzerinden gerçekleştirilerek en uygun parametreler belirlenmiştir.

2.4.3 MARS modelinin Eğitim ve test seti performans değerlendirmesi
 ![image](https://github.com/user-attachments/assets/d361b143-6fb6-4047-a4ad-dd8a3856d8c2)

MARS modeli hem eğitim hem de test setlerinde oldukça başarılı sonuçlar elde etmiştir. Eğitim setinde R² = 0.927, test setinde ise R² = 0.979 değerleri elde edilmiştir. Düşük RMSE ve MAE değerleri ise tahmin hatalarının oldukça düşüktür. 
Model, test setinde eğitim setine kıyasla daha iyi performans göstermiştir. Bu durum eğitim sürecinde veriyi aşırı öğrenmeden (overfitting) modelin genelleme yeteneğinin güçlü olduğunu gösterir.


2.5 MAKİNE ÖĞRENMELERİ İLE PEF DEĞERLERİ TAHMİN MODELLEMESİ
2.5.1 En iyi performansı sağlayan model seçimi
 ![image](https://github.com/user-attachments/assets/b0cf3750-25cf-496e-8618-60c50a22e88a)

MARS modeli, ortalama olarak Random Forest'tan biraz daha yüksek R² skoru bulunmakta. Random Forest'ın maksimum R²'si daha yüksek olsa da istikrarsız gözükmekte.

2.5.2 MARS modeli için hiperparametre ayarlama
MARS modeli için hiperparametre ayarlaması FEV1 değeri için aynı degree (1-3) aralığı seçilmiştir fakat daha uzun sürmesine rağmen daha iyi sonuçlar elde edebilmek için nprune (2'den 1000'e, 5'er artışlarla) değerleri üzerinden gerçekleştirilmiştir.


2.5.3 MARS modelinin Eğitim ve test seti performans değerlendirmesi
 ![image](https://github.com/user-attachments/assets/ea1c9139-749b-4e83-8d83-01edf57aff63)

R² değeri %90,9 olup, modelin test verilerindeki değişkenliğin büyük bir kısmını açıkladığını gösteriyor. 
Eğitim R²: 0.9942 → Test R²: 0.9087. Bu fark, modelin eğitim verisine genelleme yaparken zorlandığını göstermektedir.

2.6 Aykırı Gözlem Analizi
 ![image](https://github.com/user-attachments/assets/7e55cef6-ce0f-4294-b19f-40fec95a0075)

Veri setindeki aykırı değerlerin etkisini azaltmak ve model performansını iyileştirmek amacıyla logaritmik dönüşüm uygulanmıştır. Bu işlem, yalnızca sayısal sütunlara yapılmış olup, encoding yapılmış kategorik sütunlar dönüşümden hariç tutulmuştur.

2.7 STANDARTLAŞTIRMA
standartlaştırma işlemi yapılarak veriler ortalama = 0 ve standart sapma = 1 olacak şekilde ölçeklendirilmiştir. Log dönüşüm, sadece dağılımın normalize edilmesine yardımcı olurken, standartlaştırma tüm değişkenleri aynı ölçekte değerlendirir.




2.8 ÇOKLU DOĞRUSAL BAĞINTI İÇİN VIF ANALİZİ
 ![image](https://github.com/user-attachments/assets/bbed1a83-56e2-470c-9310-8ffd1e396be0)

 sigara_kullanimi (46.10), ailede_koah_veya_astim_tanili_hasta_var_mi (40.24), cinsiyet (22.89), sigara_birakan_ne_kadar_gun_icmis (23.91), servis_yatis_sayisi (20.53), egitim_duzeyi (18.80) ve sigara_birakan_gunde_kac_adet_icmis (15.99) değişkenlerinde güçlü doğrusal bağıntı vardır. Bu değişkenlerin diğer bağımsız değişkenlerle yüksek bir korelasyona sahiptir.

2.9 ANA BİLEŞEN ANALİZİ (PCA)
Değişkenler arasında gözlemlenen yüksek doğrusal bağıntıyı azaltmak ve veri boyutunu optimize etmek amacıyla Ana Bileşen Analizi (PCA) uygulanmıştır. PCA ile, değişkenlerin toplam varyansının büyük bir kısmını açıklayan iki yeni bileşen oluşturulmuştur. Bu işlem sonucunda, modelde kullanılan verilerin doğrusal bağıntı etkileri azaltılmış.

3.	VERİ ANALİZİ VE GÖRSELLEŞTİRME

 ![image](https://github.com/user-attachments/assets/2c4ce8a6-80c0-44bf-993c-77cd338d4d0f)
     
   ![image](https://github.com/user-attachments/assets/d91b1a72-ba27-4fd2-a822-1d9d907b547f)


   ![image](https://github.com/user-attachments/assets/4b6d798b-3560-4a03-8c15-d42d2423305c)


   ![image](https://github.com/user-attachments/assets/f7e5da46-2046-4594-914d-8d8aaef8898a)

![image](https://github.com/user-attachments/assets/fd7a08d3-8478-4efd-8a2a-22472aeb6401)

 ![image](https://github.com/user-attachments/assets/7dc8a950-74d0-4dbd-a504-6bda0c3e6213)



   ![image](https://github.com/user-attachments/assets/14ac9b86-06f1-4bcb-bcca-42493d7651fb)

 ![image](https://github.com/user-attachments/assets/0aa0d4b5-3ebd-45a9-8eee-da05c54b3d25)

 
 
 ![image](https://github.com/user-attachments/assets/5b915480-ccd5-4394-bf79-44c8d12a86ff)

 ![image](https://github.com/user-attachments/assets/9df4f11d-0896-4517-897d-f5c9dca8cf4c)

  ![image](https://github.com/user-attachments/assets/7cdefedb-8a44-4f74-8c1d-99940d50414b)

  ![image](https://github.com/user-attachments/assets/2c5b0924-f99c-4209-8b57-35cae60f65ca)

 ![image](https://github.com/user-attachments/assets/49ff8aed-2880-402a-a14d-560fb26d7ad0)

  ![image](https://github.com/user-attachments/assets/304115ee-af16-4e1e-945a-2c29c04d4862)

![image](https://github.com/user-attachments/assets/29ac95a8-b47e-48f6-8edf-8722a174e432)

   ![image](https://github.com/user-attachments/assets/87c88d4b-fc10-430c-9b36-968580aa2304)

 ![image](https://github.com/user-attachments/assets/c628639e-6f71-4935-954c-ae556dfecf38)

 ![image](https://github.com/user-attachments/assets/90911189-11a9-4747-90b2-c910e7e84b48)

 
![image](https://github.com/user-attachments/assets/aca99f92-9f67-40c9-9b7b-db4c1279b5f7)

   
  
4.	MODEL SEÇİMİ VE TEST HATASI

4.1 DEĞİŞKEN SEÇİMİ VE YENİ DEĞİŞKENLER
4.1.1 Özellik Mühendisliği ile Türetilen Değişkenler
BMI (Vücut Kitle İndeksi):
	Kişinin vücut ağırlığının boy oranına göre hesaplanarak obezite veya kilo durumlarını değerlendiren bir değişken oluşturulmuştur.
TANI Süresi Toplamı (Ay Olarak):
	Hastalığın toplam süresini yıl ve ay bilgilerini birleştirerek tek bir değişkende ifade etmek için türetilmiştir.
Sigara Paket Yılı:
	Sigara tüketiminin süresi ve yoğunluğuna dayalı bir gösterge oluşturulmuştur.
Kan Basıncı Skoru:
	Kan basıncı değerlerini sistolik ve diyastolik değerlere göre normalize ederek genel bir gösterge oluşturulmuştur.
Nabız Durumu (Kategori):
	Nabız değerleri düşük, normal ve yüksek olarak sınıflandırılmıştır.
Solunum Verimliliği:
	Solunum sayısı ve PEF yüzdesine dayalı bir gösterge oluşturulmuştur.
FEV1/FVC Durumu (Kategori):
	Solunum fonksiyonlarının değerlendirilmesi için düşük, riskli ve normal kategorilere ayrılmıştır.
Tansiyon Durumu (Kategori):
	Sistolik kan basıncı değerleri düşük, normal, yüksek ve hipertansiyon olarak sınıflandırılmıştır.
Ailede Hasta Sayısı:
	Aile bireylerinde hastalık geçmişine dayalı toplam hasta sayısını ifade eden bir değişken oluşturulmuştur.
Toplam Yatış Süresi:
	Hastanede yatış sürelerini farklı alanlardan (acil servis, yoğun bakım, servis) birleştirerek hesaplanan toplam süre.
Risk Skoru:
	TANI süresi, yatış süresi ve sigara paket yılına dayalı bir risk değerlendirme değişkeni.
Yaş Grupları (Kategori):
	Yaş değerleri genç, orta yaş ve yaşlı olarak kategorilere ayrılmıştır.
PEF Kategorileri:
	PEF değerleri çok düşük, düşük, orta ve yüksek olarak sınıflandırılmıştır.
PEF Normalizasyonu (FEV1 ile):
	PEF değerlerinin FEV1 yüzdesine oranı hesaplanarak bir değişken oluşturulmuştur.
PEF ve Yaş Oranı:
	PEF değerinin yaşa oranlanmasıyla solunum performansını gösteren bir değişken oluşturulmuştur.
PEF Z-Skoru:
	PEF değerlerinin z-skoru ile normalize edilmesi sağlanmıştır.
PEF ve Tansiyon Skoru:
	PEF değerinin sistolik kan basıncına oranı hesaplanmıştır.

4.1.2 Model için önemli değişkenler
1. Principal_Component_1 (Fisher Skoru: 312.11)
2. FEV1/FVC_Değeri (Fisher Skoru: 232.35)
3.Cinsiyet (Fisher Skoru: 188.65)
4. Yaş (Fisher Skoru: 170.43)
5. Sigara Bırakan Gün Sayısı (142.26)
6. Günde Kaç Adet İçmiş (162.08
7. FEV1 (135.90) 
8. FEV1_yüzde (116.61)
9. PEF (96.72)
10. PEF_yüzde (49.44)
11. Servis yatış sayısı (46.70)
12. servis toplam yatış süresi (44.28)
13. Kan Basıncı (Sistolik: 22.15)
14. Kan Basıncı (Diyastolik: 8.90)
15. Sigara devam eden günde kaç adet içiyor (6.94) 

Fisher skoru yüksek olan ilk 15-20 değişkeni seçmek, modelin performansını artırabilir.

4.2 MODELLEME
Kullanılan Modeller ve Parametreleri:
1.	Lojistik Regresyon (Logistic Regression):
•	Hiperparametreler:
	C: Ceza parametresi (0.01, 0.1, 1, 10).

2.	Karar Ağacı (Decision Tree):
•	Hiperparametreler:
	max_depth: Maksimum derinlik (3, 5, 10).
	min_samples_split: Dallanma için gereken minimum örnek sayısı (2, 5, 10).

3.	Rastgele Orman (Random Forest):
•	Hiperparametreler:
	n_estimators: Ağaç sayısı (50, 100, 200).
	max_depth: Maksimum derinlik (None, 10, 20).

4.	Gradyan Artırma (Gradient Boosting):
•	Hiperparametreler:
	learning_rate: Öğrenme hızı (0.01, 0.1, 0.2).
	n_estimators: Ağaç sayısı (50, 100, 200).

5.	XGBoost (Extreme Gradient Boosting):
•	Hiperparametreler:
	learning_rate: Öğrenme hızı (0.01, 0.1, 0.2).
	n_estimators: Ağaç sayısı (50, 100, 200).



6.	Destek Vektör Makineleri (SVM - Support Vector Machines):
•	Hiperparametreler:
	C: Ceza parametresi (0.1, 1, 10).
	kernel: Çekirdek tipi (linear, rbf).

7.	K-En Yakın Komşu (KNN - K-Nearest Neighbors):
•	Hiperparametreler:
	n_neighbors: Komşu sayısı (3, 5, 10).

Her model için hiperparametre optimizasyonu GridSearchCV yöntemiyle gerçekleştirilmiştir. Bu süreçte, her parametre kombinasyonu 10 katlı çapraz doğrulama ile değerlendirilmiştir. Bu sayede, modellerin en uygun hiperparametreleri seçilmiştir.

4.3 MODELİN PERFORMANSININ DEĞERLENDİRİLMESİ
4.3.1 Çalışmada kullanılan performans metrikleri:
Accuracy (Doğruluk):
•	Modelin doğru tahmin ettiği örneklerin toplam örnek sayısına oranıdır.
•	Yorum: Sınıflandırma modellerinde temel bir performans göstergesi olarak kullanılır.
RMSE (Kök Ortalama Kare Hatası):
•	Gerçek ve tahmin edilen değerler arasındaki hata büyüklüğünü ölçer.
•	Yorum: Daha düşük RMSE, modelin tahmin doğruluğunun yüksek olduğunu gösterir.
RRMSE:
•	RMSE değerinin gerçek değerlerin ortalamasına oranıdır.
•	Yorum: Modelin hata oranını nispi olarak değerlendirmek için kullanılır.
SDR (Standard Deviation Ratio):
•	Tahmin edilen değerlerin standart sapmasının, gerçek değerlerin standart sapmasına oranıdır.
•	Yorum: Modelin değişkenlik düzeyini analiz eder.
CV (Varyasyon Katsayısı):
•	Gerçek değerlerin standart sapmasının ortalamaya oranıdır.
•	Yorum: Değişkenlik düzeyini yüzdelik olarak değerlendirir.
MAPE (Ortalama Mutlak Yüzde Hatası):
•	Gerçek ve tahmin edilen değerler arasındaki ortalama yüzde hatasını ölçer.
•	Yorum: Tahmin hatasının yüzdelik olarak ne kadar büyük olduğunu gösterir.
MAD (Ortalama Mutlak Sapma):
•	Gerçek ve tahmin edilen değerler arasındaki mutlak hatanın ortalamasıdır.
•	Yorum: Model hatasının ortalama büyüklüğünü ölçer.
R² (R-Kare):
•	Modelin açıklayabildiği varyansın toplam varyansa oranıdır.
•	Yorum: Modelin ne kadar iyi bir uyum sağladığını ölçer. Yüksek R², modelin daha iyi performans gösterdiğini belirtir.
Adjusted R² (Düzeltilmiş R-Kare):
•	R² değerinin değişken sayısı göz önünde bulundurularak düzeltilmiş halidir.
•	Yorum: Özellikle çoklu değişken içeren modellerde, aşırı uyumun (overfitting) etkisini azaltmak için kullanılır.
AIC (Akaike Bilgi Kriteri):
•	Modelin uyumu ile karmaşıklığı arasındaki dengeyi ölçer. Daha düşük AIC değeri, daha iyi bir model seçimini işaret eder.
•	Yorum: Model performansını ve parsimoni (sadelik) düzeyini değerlendirir.
CAIC:
•	AIC’nin düzeltilmiş bir versiyonudur ve daha büyük veri setleri için uygundur.
•	Yorum: Model seçiminde cezalandırma faktörünü artırarak karmaşık modelleri sınırlamayı amaçlar.
ROC-AUC (Alan Altındaki Eğri):
•	Modelin pozitif ve negatif sınıfları doğru bir şekilde ayırma yeteneğini ölçer.
•	Yorum: ROC-AUC değeri 1’e ne kadar yakınsa, modelin sınıflandırma performansı o kadar iyidir.






Genel Sonuç ve Model değerlendirmesi:
 ![image](https://github.com/user-attachments/assets/3534d76a-1e57-456f-adf2-ea27c129e6a9)

XGBoost:
	Test setinde yüksek %92 doğruluk ve 0.95 ROC-AUC değerlerini sağlamıştır.

Lojistik Regresyon:
	Basit ve hızlı bir model olmasına rağmen test setinde %92 doğruluk ve 0.96 ROC-AUC değerleri ile en yüksek değere sahiptir.

Gradient Boosting:
	Test setinde %89 doğruluk ve tutarlı bir ROC-AUC değeri (0.96) değerlerini sağlamıştır.

SVM:
	Test setinde %88 doğruluk ve tutarlı bir ROC-AUC değeri (0.95) ile sağlam bir performans göstermiştir.

Random Forest:
	Test setinde %89 doğruluk ve 0.96’ya kadar ROC-AUC değeri sağlamış olsa da Lojistik Regresyon ve XGBoost’tan hafif geride kalmıştır.

Elde edilen metriklerin sonuçlarına göre çok iyi sonuçlar elde edilmemiştir. Modellerin PEF değerleri tahmini, veri standartlaştırma işlemi ve Ana Bileşen Analizi (PCA) yapılmadan tekrar değerlendirilmesi yapılarak model performanslarını yeniden gözden geçirilecektir.



5.	HAM VERİ ÜZERİNDE MAKİNE ÖĞRENİMİ MODELLERİNİN DEĞERLENDİRİLMESİ
PEF değerleri tahmini, standartlaştırma ve PCA gibi ön işleme adımları olmadan modellerin performansında ne tür değişimler meydana geldiğine bakılacaktır. 
 ![image](https://github.com/user-attachments/assets/00711bee-e176-4dcc-b1a0-dbb5768e481c)

Genel Sonuç ve Model değerlendirmesi:
XGBoost modeli hem eğitim hem de test setinde en yüksek performansı göstermiştir.
•	Test Seti:
	Accuracy: 0.9701
	RMSE: 0.1727
	R²: 0.8792
	ROC-AUC: 0.9991
	Eğitim setinde ise %100 doğruluk (Accuracy: 1) elde edilmiş. Bu durum overfitting şüphesi doğurabilir.
	Model eğitimde %100 doğruluk gösteriyor ancak test sonuçları hâlâ çok iyi. Bu durum hafif bir overfitting ihtimali barındırsa da model performansı test setinde tatmin edici.
Random Forest modeli XGBoost'a yakın performans gösteriyor. Ancak test setindeki sonuçları biraz daha düşük.
•	Test Seti 
	Accuracy: 0.9403 
	RMSE: 0.2443
	R²: 0.7958 (XGBoost'a göre daha düşük)
	ROC-AUC değeri: 0.9961

Accuracy (Doğruluk):XGBoost, Random Forest ve Logistic Regression öne çıkmaktadır.
RMSE: Hata oranı açısından XGBoost en düşük değere sahiptir (0.1727).
R² ve Adjusted R²: XGBoost modeli en yüksek değerleri sunuyor (0.8792 test setinde).
ROC-AUC: Tüm modellerde yüksek, ancak XGBoost ve Random Forest burada da liderdir.
