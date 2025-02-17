import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    mean_squared_error, explained_variance_score, mean_absolute_error,
    roc_auc_score, r2_score, accuracy_score
)
from math import log

# Veri Yükleme
DATA_PATH = "pca_reduced_features_with_original.xlsx"  # Excel dosyasının yolu
SHEET_NAME = "Sheet1"  # Sayfa adı
data = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME)

# Türetilmiş Değişkenler
# 1. BMI (Vücut Kitle İndeksi)
data['BMI_x'] = data['vucut_agirligi'] / ((data['boy'] / 100) ** 2)

# 2. TANI süresi toplamı (ay olarak)
data['tani_suresi_toplam_ay_x'] = (data['tani_suresi_yil'] * 12) + data['tani_suresi_ay']

# 3. Sigara Paket Yılı
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

# 9. Ailede Hasta Sayısı
data['aile_hasta_toplam_x'] = (
    data['varsa_kimde_anne'] +
    data['varsa_kimde_baba'] +
    data['varsa_kimde_kardes'] +
    data['varsa_kimde_diger']
)

# 10. Toplam Yatış Süreleri
data['toplam_yatis_suresi_x'] = (
    data['acil_servis_toplam_yatis_suresi_gun'] +
    data['yogun_bakim_toplam_yatis_suresi_gun'] +
    data['servis_toplam_yatis_suresi_gu']
)

# 11. Risk Skoru
data['risk_skoru_x'] = (
    data['tani_suresi_toplam_ay_x'] * 0.5 +
    data['toplam_yatis_suresi_x'] * 1.0 +
    data['paket_yili_x'] * 0.8
)

# 12. Yaş Grupları
data['yas_grubu_x'] = pd.cut(
    data['yas'], bins=[0, 30, 50, float('inf')], labels=['Genç', 'Orta Yaş', 'Yaşlı']
)

# 13. PEF Kategorileri
data['PEF_kategori_x'] = pd.cut(
    data['PEF'],
    bins=[0, 250, 400, 600, float('inf')],
    labels=['Çok Düşük', 'Düşük', 'Orta', 'Yüksek']
)

# 14. PEF Normalizasyonu (FEV1 ile)
data['PEF_normalize_FEV1_x'] = data['PEF'] / data['FEV1_yuzde']

# 15. PEF ve Yaş Oranı
data['PEF_yas_orani_x'] = data['PEF'] / data['yas']

# 16. PEF Z-Skoru
pef_mean = data['PEF'].mean()
pef_std = data['PEF'].std()
data['PEF_zskoru_x'] = (data['PEF'] - pef_mean) / pef_std

# 17. PEF ve Tansiyon Skoru
data['PEF_tansiyon_skoru_x'] = data['PEF'] / data['kan_basinci_sistolik']

# Orijinal Sütunları Kaldırma
data.drop(columns=['tani_suresi_yil', 'tani_suresi_ay'], inplace=True)

# Kategorik Sütunların Listesi
categorical_columns = [
    'nabiz_kategori_x', 'FEV1_FVC_kategori_x', 'yas_grubu_x', 'tansiyon_durumu_x', 'PEF_kategori_x'
]

# Eksik Kategorik Sütunları Kontrol Etme
missing_columns = [col for col in categorical_columns if col not in data.columns]
if missing_columns:
    raise ValueError(f"Kategorik sütunlar eksik: {missing_columns}")

# Kategorik Sütunları One-Hot Encode
encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded_categorical = pd.DataFrame(
    encoder.fit_transform(data[categorical_columns]),
    columns=encoder.get_feature_names_out(categorical_columns)
)

# Veri Setini Güncelleme
data.drop(columns=categorical_columns, inplace=True)
data = pd.concat([data, encoded_categorical], axis=1)

# Veri Hazırlık
X = data.drop(columns=["tani", "hasta_no", "basvuru_tarihi", "yogun_bakim_toplam_yatis_suresi_saat", "servis_toplam_yatis_suresi_saat"])
y = data["tani"]

# Bağımlı Değişkeni Etiketleme
le = LabelEncoder()
y = le.fit_transform(y)

# Veri Setini Bölme
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Verileri Ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Performans Metrikleri Hesaplama Fonksiyonu
def calculate_metrics(y_true, y_pred, y_pred_proba=None, model_name="Model", n_features=None):
    metrics = {}
    n = len(y_true)
    k = n_features if n_features else len(y_pred)

    # Accuracy
    metrics["Accuracy"] = accuracy_score(y_true, y_pred)

    # RMSE
    metrics["RMSE"] = np.sqrt(mean_squared_error(y_true, y_pred))

    # RRMSE
    metrics["RRMSE"] = metrics["RMSE"] / np.mean(y_true) if np.mean(y_true) != 0 else np.nan

    # SDR
    metrics["SDR"] = np.std(y_pred) / np.std(y_true) if np.std(y_true) != 0 else np.nan

    # CV
    metrics["CV"] = np.std(y_true) / np.mean(y_true) if np.mean(y_true) != 0 else np.nan

    # MAPE
    non_zero_indices = y_true != 0
    if np.any(non_zero_indices):
        metrics["MAPE"] = np.mean(np.abs((y_true[non_zero_indices] - y_pred[non_zero_indices]) / y_true[non_zero_indices])) * 100
    else:
        metrics["MAPE"] = np.nan

    # MAD
    metrics["MAD"] = mean_absolute_error(y_true, y_pred)

    # R-squared
    metrics["R²"] = r2_score(y_true, y_pred)

    # Adjusted R-squared
    metrics["Adjusted R²"] = 1 - (1 - metrics["R²"]) * ((n - 1) / (n - k - 1))

    # AIC
    rss = sum((y_true - y_pred) ** 2)
    if rss > 0:
        metrics["AIC"] = n * log(rss / n) + 2 * k
    else:
        metrics["AIC"] = np.nan

    # CAIC
    if "AIC" in metrics and not np.isnan(metrics["AIC"]):
        metrics["CAIC"] = metrics["AIC"] + (log(n) - 2) * k
    else:
        metrics["CAIC"] = np.nan

    # ROC-AUC
    if y_pred_proba is not None:
        metrics["ROC-AUC"] = roc_auc_score(y_true, y_pred_proba)

    return pd.DataFrame(metrics, index=[model_name])

# Model Tanımlamaları
models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "XGBoost": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
    "SVM": SVC(probability=True, random_state=42),
    "KNN": KNeighborsClassifier(),
}

# Hiperparametre Arama
param_grids = {
    "Logistic Regression": {"C": [0.01, 0.1, 1, 10]},
    "Decision Tree": {"max_depth": [3, 5, 10], "min_samples_split": [2, 5, 10]},
    "Random Forest": {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20]},
    "Gradient Boosting": {"learning_rate": [0.01, 0.1, 0.2], "n_estimators": [50, 100, 200]},
    "XGBoost": {"learning_rate": [0.01, 0.1, 0.2], "n_estimators": [50, 100, 200]},
    "SVM": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]},
    "KNN": {"n_neighbors": [3, 5, 10]}
}

best_models = {}

for model_name, model in models.items():
    print(f"Tuning hyperparameters for {model_name}...")
    grid_search = GridSearchCV(model, param_grids[model_name], cv=10, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)
    best_models[model_name] = grid_search.best_estimator_
    print(f"Best parameters for {model_name}: {grid_search.best_params_}")

# Model Eğitim ve Değerlendirme
results_train = []
results_test = []

for model_name, model in best_models.items():
    # Modeli eğit
    model.fit(X_train_scaled, y_train)

    # Eğitim seti tahminleri
    y_train_pred = model.predict(X_train_scaled)
    y_train_proba = model.predict_proba(X_train_scaled)[:, 1] if hasattr(model, "predict_proba") else None

    # Test seti tahminleri
    y_test_pred = model.predict(X_test_scaled)
    y_test_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else None

    # Eğitim Metrikleri
    train_metrics = calculate_metrics(
        y_true=y_train,
        y_pred=y_train_pred,
        y_pred_proba=y_train_proba,
        model_name=f"{model_name} (Train)",
        n_features=X_train.shape[1]
    )
    results_train.append(train_metrics)

    # Test Metrikleri
    test_metrics = calculate_metrics(
        y_true=y_test,
        y_pred=y_test_pred,
        y_pred_proba=y_test_proba,
        model_name=f"{model_name} (Test)",
        n_features=X_train.shape[1]
    )
    results_test.append(test_metrics)

# Sonuçları Birleştirme
final_results = pd.concat(results_train + results_test)

# Sonuçları Kaydetme
final_results.to_excel("model_performance_results99.xlsx", index=True)
print("Sonuçlar 'model_performance_results.xlsx' dosyasına kaydedildi.")
