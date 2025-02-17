import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
import xgboost as xgb

# Özellik mühendisliği fonksiyonu
def feature_engineering(data):
    # 1. BMI (Vücut Kitle İndeksi)
    data['BMI_x'] = pd.to_numeric(data['vucut_agirligi'], errors='coerce') / (
        (pd.to_numeric(data['boy'], errors='coerce') / 100) ** 2)

    # 2. TANI süresi toplamını ay olarak birleştirme (Sütunlar mevcutsa)
    if 'tani_suresi_yil' in data.columns and 'tani_suresi_ay' in data.columns:
        data['tani_suresi_toplam_ay_x'] = (pd.to_numeric(data['tani_suresi_yil'], errors='coerce').fillna(0) * 12) + pd.to_numeric(data['tani_suresi_ay'], errors='coerce').fillna(0)
    else:
        data['tani_suresi_toplam_ay_x'] = 0

    # 3. Sigara ile ilgili türev değişkenler (paket yılı)
    data['paket_yili_x'] = (pd.to_numeric(data['sigara_birakan_ne_kadar_gun_icmis'], errors='coerce').fillna(0) / 365.25) * (
        pd.to_numeric(data['sigara_birakan_gunde_kac_adet_icmis'], errors='coerce').fillna(0) / 20)

    # 4. Kan Basıncı Skoru
    data['kan_basinci_skoru_x'] = (pd.to_numeric(data['kan_basinci_sistolik'], errors='coerce') +
                                   (2 * pd.to_numeric(data['kan_basinci_diastolik'], errors='coerce'))) / 3

    # 5. Nabız Durumu
    data['nabiz_kategori_x'] = pd.cut(
        pd.to_numeric(data['nabiz'], errors='coerce'),
        bins=[0, 60, 100, float('inf')],
        labels=['Düşük', 'Normal', 'Yüksek']
    )

    # 6. Solunum Verimliliği
    data['solunum_verimliligi_x'] = pd.to_numeric(data['PEF_yuzde'], errors='coerce') / pd.to_numeric(
        data['solunum_sayisi'], errors='coerce')

    # 7. FEV1/FVC Durumu
    data['FEV1_FVC_kategori_x'] = pd.cut(
        pd.to_numeric(data['FEV1_FVC_Değeri'], errors='coerce'),
        bins=[0, 0.7, 0.8, float('inf')],
        labels=['Düşük', 'Riskli', 'Normal']
    )

    # 8. Tansiyon Durumu
    data['tansiyon_durumu_x'] = pd.cut(
        pd.to_numeric(data['kan_basinci_sistolik'], errors='coerce'),
        bins=[0, 90, 120, 140, float('inf')],
        labels=['Düşük', 'Normal', 'Yüksek', 'Hipertansiyon']
    )

    # 9. Ailede hasta sayısı
    data['aile_hasta_toplam_x'] = (
            pd.to_numeric(data['varsa_kimde_anne'], errors='coerce').fillna(0) +
            pd.to_numeric(data['varsa_kimde_baba'], errors='coerce').fillna(0) +
            pd.to_numeric(data['varsa_kimde_kardes'], errors='coerce').fillna(0) +
            pd.to_numeric(data['varsa_kimde_diger'], errors='coerce').fillna(0)
    )

    # 10. Toplam yatış süresi
    data['toplam_yatis_suresi_x'] = (
            pd.to_numeric(data['acil_servis_toplam_yatis_suresi_gun'], errors='coerce').fillna(0) +
            pd.to_numeric(data['yogun_bakim_toplam_yatis_suresi_gun'], errors='coerce').fillna(0) +
            pd.to_numeric(data['servis_toplam_yatis_suresi_gu'], errors='coerce').fillna(0)
    )

    # 11. Risk Skoru
    data['risk_skoru_x'] = (
            data['tani_suresi_toplam_ay_x'] * 0.5 +
            data['toplam_yatis_suresi_x'] * 1.0 +
            data['paket_yili_x'] * 0.8
    )

    # 12. Yaş Grupları
    data['yas_grubu_x'] = pd.cut(pd.to_numeric(data['yas'], errors='coerce'), bins=[0, 30, 50, float('inf')],
                                 labels=['Genç', 'Orta Yaş', 'Yaşlı'])

    # 13. PEF Kategorileri
    data['PEF_kategori_x'] = pd.cut(
        pd.to_numeric(data['PEF'], errors='coerce'),
        bins=[0, 250, 400, 600, float('inf')],
        labels=['Çok Düşük', 'Düşük', 'Orta', 'Yüksek']
    )

    # 14. PEF Normalizasyonu (FEV1 ile)
    data['PEF_normalize_FEV1_x'] = pd.to_numeric(data['PEF'], errors='coerce') / pd.to_numeric(data['FEV1_yuzde'], errors='coerce')

    # 16. PEF ve Yaş İlişkisi
    data['PEF_yas_orani_x'] = pd.to_numeric(data['PEF'], errors='coerce') / pd.to_numeric(data['yas'], errors='coerce')

    # 17. PEF Z-Skoru
    pef_mean = pd.to_numeric(data['PEF'], errors='coerce').mean()
    pef_std = pd.to_numeric(data['PEF'], errors='coerce').std()
    data['PEF_zskoru_x'] = (pd.to_numeric(data['PEF'], errors='coerce') - pef_mean) / pef_std

    # 18. PEF ve Tansiyon İlişkisi
    data['PEF_tansiyon_skoru_x'] = pd.to_numeric(data['PEF'], errors='coerce') / pd.to_numeric(
        data['kan_basinci_sistolik'], errors='coerce')

    return data

# Veri setlerini yükleme
final_data = pd.read_excel("final_dataset4_asx.xlsx")
ist405_data = pd.read_excel("ist405_ogrenci_test(3).xlsx")

# Özellik mühendisliğini uygulama
final_data = feature_engineering(final_data)
ist405_data = feature_engineering(ist405_data)

# Eğitim veri setinde eksik değerlerin doldurulması
categorical_columns = ['nabiz_kategori_x', 'FEV1_FVC_kategori_x', 'yas_grubu_x', 'tansiyon_durumu_x', 'PEF_kategori_x']
numeric_columns = final_data.select_dtypes(include=["number"]).columns.tolist()
categorical_columns_final = [col for col in categorical_columns if col in final_data.columns]

numeric_imputer = SimpleImputer(strategy='mean')
categorical_imputer = SimpleImputer(strategy='most_frequent')

numeric_data = pd.DataFrame(numeric_imputer.fit_transform(final_data[numeric_columns]), columns=numeric_columns)
categorical_data = pd.DataFrame(categorical_imputer.fit_transform(final_data[categorical_columns_final]),
                                columns=categorical_columns_final)

# Veriyi birleştirme
final_data_imputed = pd.concat([numeric_data, categorical_data], axis=1)

# One-hot encoding işlemi
encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
encoded_categorical = pd.DataFrame(
    encoder.fit_transform(final_data_imputed[categorical_columns_final]),
    columns=encoder.get_feature_names_out(categorical_columns_final)
)

final_data_encoded = pd.concat([final_data_imputed.drop(columns=categorical_columns_final), encoded_categorical],
                               axis=1)

# Eğitim ve test veri setini ayırma
y_final = final_data["tani"] - final_data["tani"].min()
X_final = final_data_encoded.drop(columns=["hasta_no", "PEF"], errors='ignore')
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.2, random_state=42, stratify=y_final)

# Ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# XGBoost modeli ve hiperparametre optimizasyonu
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.2],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "gamma": [0, 0.1, 0.2]
}

random_search = RandomizedSearchCV(estimator=xgb_model, param_distributions=param_grid, n_iter=10, scoring="accuracy",
                                   cv=5, verbose=2, random_state=42, n_jobs=-1)
random_search.fit(X_train_scaled, y_train)

# En iyi model ile tahmin
best_model = random_search.best_estimator_
y_train_pred = best_model.predict(X_train_scaled)
y_test_pred = best_model.predict(X_test_scaled)

print("\nEğitim Seti Performansı:")
print(classification_report(y_train, y_train_pred))
print("\nTest Seti Performansı:")
print(classification_report(y_test, y_test_pred))

# IST405 veri setinde temizleme ve eksik değer doldurma
categorical_columns_ist405 = [col for col in categorical_columns if col in ist405_data.columns]

# "na" gibi geçersiz verileri NaN olarak işaretle
ist405_data.replace("na", pd.NA, inplace=True)

# Sayısal sütunları tespit et
numeric_columns_ist405 = [col for col in ist405_data.columns if ist405_data[col].dtype in ["float64", "int64"]]

# Sayısal sütun isimlerini eşleştir
numeric_columns_ist405 = [col for col in numeric_columns if col in ist405_data.columns]

# Eksik değerleri doldurma
ist405_data[numeric_columns_ist405] = pd.DataFrame(
    numeric_imputer.transform(ist405_data[numeric_columns_ist405]),
    columns=numeric_columns_ist405
)

# Kategorik sütunları doldurma
categorical_columns_ist405 = [col for col in categorical_columns if col in ist405_data.columns]
ist405_data[categorical_columns_ist405] = pd.DataFrame(
    categorical_imputer.transform(ist405_data[categorical_columns_ist405]),
    columns=categorical_columns_ist405
)

# One-hot encoding işlemi
encoded_ist405 = pd.DataFrame(
    encoder.transform(ist405_data[categorical_columns_ist405]),
    columns=encoder.get_feature_names_out(categorical_columns_ist405)
)

# Veri setini birleştir
ist405_encoded = pd.concat(
    [ist405_data.drop(columns=categorical_columns_ist405), encoded_ist405],
    axis=1
)

# Modelin beklediği sütunlara göre hizalama
ist405_encoded = ist405_encoded.reindex(columns=X_final.columns, fill_value=0)

# Ölçeklendirme ve tahmin
X_ist405_scaled = scaler.transform(ist405_encoded)
ist405_predictions = best_model.predict(X_ist405_scaled)

# Tahmin sonuçlarını ekleme
ist405_data['tani_predicted'] = ist405_predictions + final_data["tani"].min()

# Sonuçları kaydetme
ist405_data.to_excel("ist405_with_predictions.xlsx", index=False)
print("Tahminler kaydedildi.")
