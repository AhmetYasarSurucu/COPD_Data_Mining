import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, accuracy_score, r2_score
import xgboost as xgb

# Veri setini yükleme
file_path = "final_dataset4_asx.xlsx"  # Dosya yolunu güncelleyin
data = pd.read_excel(file_path)

# Özellik mühendisliği (Feature Engineering)

# 1. BMI (Vücut Kitle İndeksi)
data['BMI_x'] = data['vucut_agirligi'] / ((data['boy'] / 100) ** 2)

# 2. TANI süresi toplamını ay olarak birleştirme
data['tani_suresi_toplam_ay_x'] = (data['tani_suresi_yil'] * 12) + data['tani_suresi_ay']

# 3. Sigara ile ilgili türev değişkenler (paket yılı)
data['paket_yili_x'] = (data['sigara_birakan_ne_kadar_gun_icmis'] / 365.25) * (
    data['sigara_birakan_gunde_kac_adet_icmis'] / 20)

# 4. Kan Basıncı Skoru
data['kan_basinci_skoru_x'] = (data['kan_basinci_sistolik'] + (2 * data['kan_basinci_diastolik'])) / 3

# 5. Nabız Durumu
data['nabiz_kategori_x'] = pd.cut(
    data['nabiz'],
    bins=[0, 60, 100, float('inf')],
    labels=['Düşük', 'Normal', 'Yüksek']
)

# 6. Solunum Verimliliği
data['solunum_verimliligi_x'] = data['PEF_yuzde'] / data['solunum_sayisi']

# 7. FEV1/FVC Durumu
data['FEV1_FVC_kategori_x'] = pd.cut(
    data['FEV1_FVC_Değeri'],
    bins=[0, 0.7, 0.8, float('inf')],
    labels=['Düşük', 'Riskli', 'Normal']
)

# 8. Tansiyon Durumu
data['tansiyon_durumu_x'] = pd.cut(
    data['kan_basinci_sistolik'],
    bins=[0, 90, 120, 140, float('inf')],
    labels=['Düşük', 'Normal', 'Yüksek', 'Hipertansiyon']
)

# 9. Ailede hasta sayısı
data['aile_hasta_toplam_x'] = (data['varsa_kimde_anne'] +
                               data['varsa_kimde_baba'] +
                               data['varsa_kimde_kardes'] +
                               data['varsa_kimde_diger'])

# 10. Toplam yatış süresi
data['toplam_yatis_suresi_x'] = (data['acil_servis_toplam_yatis_suresi_gun'] +
                                 data['yogun_bakim_toplam_yatis_suresi_gun'] +
                                 data['servis_toplam_yatis_suresi_gu'])

# 11. Risk Skoru
data['risk_skoru_x'] = (
    data['tani_suresi_toplam_ay_x'] * 0.5 +
    data['toplam_yatis_suresi_x'] * 1.0 +
    data['paket_yili_x'] * 0.8
)

# 12. Yaş Grupları
data['yas_grubu_x'] = pd.cut(data['yas'], bins=[0, 30, 50, float('inf')], labels=['Genç', 'Orta Yaş', 'Yaşlı'])

# 13. PEF Kategorileri
data['PEF_kategori_x'] = pd.cut(
    data['PEF'],
    bins=[0, 250, 400, 600, float('inf')],
    labels=['Çok Düşük', 'Düşük', 'Orta', 'Yüksek']
)

# 14. PEF Normalizasyonu (FEV1 ile)
data['PEF_normalize_FEV1_x'] = data['PEF'] / data['FEV1_yuzde']



# 16. PEF ve Yaş İlişkisi
data['PEF_yas_orani_x'] = data['PEF'] / data['yas']

# 17. PEF Z-Skoru
pef_mean = data['PEF'].mean()
pef_std = data['PEF'].std()
data['PEF_zskoru_x'] = (data['PEF'] - pef_mean) / pef_std

# 18. PEF ve Tansiyon İlişkisi
data['PEF_tansiyon_skoru_x'] = data['PEF'] / data['kan_basinci_sistolik']

# Orijinal sütunları kaldırma
data = data.drop(columns=['tani_suresi_yil', 'tani_suresi_ay'])

# Kategorik değişkenlerin listesi
categorical_columns = [
    'nabiz_kategori_x', 'FEV1_FVC_kategori_x', 'yas_grubu_x', 'tansiyon_durumu_x', 'PEF_kategori_x'
]

# Eksik sütunları kontrol etme
missing_columns = [col for col in categorical_columns if col not in data.columns]
if missing_columns:
    raise ValueError(f"Kategorik sütunlar eksik: {missing_columns}")

# Kategorik sütunları one-hot encode yapma
encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded_categorical = pd.DataFrame(encoder.fit_transform(data[categorical_columns]),
                                    columns=encoder.get_feature_names_out(categorical_columns))

# Orijinal veri çerçevesinden kategorik sütunları çıkar ve encoded sütunları ekle
data = data.drop(columns=categorical_columns)
data = pd.concat([data, encoded_categorical], axis=1)

# Hedef ve bağımsız değişkenlerin seçimi
X = data.drop(columns=["hasta_no", "tani", "PEF"])
y = data["tani"]

# Datetime türündeki sütunları kontrol etme ve çıkarma
datetime_columns = X.select_dtypes(include=["datetime64"]).columns.tolist()
if datetime_columns:
    print(f"Çıkarılan datetime sütunları: {datetime_columns}")
    X = X.drop(columns=datetime_columns)

# Eksik değer kontrolü ve temizleme
if X.isnull().sum().sum() > 0:
    X = X.dropna()
    y = y[X.index]

# Hedef sınıfları sıfır tabanlı yapma
y = y - y.min()

# Eğitim ve test setine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Özellik ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# XGBoost modeli
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")

# RandomizedSearchCV ile hiperparametre optimizasyonu
random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions={
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.2],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "gamma": [0, 0.1, 0.2]
    },
    n_iter=10,
    scoring="accuracy",
    cv=5,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

# Model eğitimi
random_search.fit(X_train_scaled, y_train)

# En iyi hiperparametreleri yazdırma
print("En İyi Parametreler:", random_search.best_params_)

# Tahminler ve performans
best_model = random_search.best_estimator_
y_train_pred = best_model.predict(X_train_scaled)
y_test_pred = best_model.predict(X_test_scaled)

print("\nEğitim Performansı:")
print(classification_report(y_train, y_train_pred))
print("\nTest Performansı:")
print(classification_report(y_test, y_test_pred))

# R² Skorları
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)
print(f"R² Skoru (Eğitim): {r2_train:.4f}")
print(f"R² Skoru (Test): {r2_test:.4f}")

# Doğruluk değerleri
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Doğruluk (Eğitim): {train_accuracy:.4f}")
print(f"Doğruluk (Test): {test_accuracy:.4f}")
