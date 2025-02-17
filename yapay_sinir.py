import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, r2_score

# 1. Veri Yükleme
df = pd.read_excel("final_dataset4.xlsx")

# 2. Tarih Kolonunu Kaldırma (Modeli bozuyor)
df = df.select_dtypes(exclude=['datetime64'])

# 3. Eksik Değerleri Kontrol Etme ve Doldurma
print("Eksik değerler:")
print(df.isnull().sum())

df.fillna(df.median(numeric_only=True), inplace=True)

# 4. Özellik Mühendisliği
df['tani_suresi_x'] = df['tani_suresi_yil'] * 12 + df['tani_suresi_ay']

df['hastane_servis_yatis_x'] = (
    df['hastaneye_yatti_mi'] +
    df['acil_servis_yatis_sayisi'] +
    df['acil_servis_toplam_yatis_suresi_saat'] / 24 +
    df['acil_servis_toplam_yatis_suresi_gun'] +
    df['servis_yatis_sayisi'] +
    df['servis_toplam_yatis_suresi_saat'] / 24 +
    df['servis_toplam_yatis_suresi_gu']
)

df['yogun_bakim_yatis_x'] = (
    df['yogun_bakim_yatis_sayisi'] +
    df['yogun_bakim_toplam_yatis_suresi_saat'] / 24 +
    df['yogun_bakim_toplam_yatis_suresi_gun']
)

df['BMI_x'] = df['vucut_agirligi'] / ((df['boy'] / 100) ** 2)

df['paket_yili_x'] = (df['sigara_birakan_ne_kadar_gun_icmis'] / 365.25) * (
    df['sigara_birakan_gunde_kac_adet_icmis'] / 20
)

df['nabiz_kategori_x'] = pd.cut(
    df['nabiz'],
    bins=[0, 60, 100, float('inf')],
    labels=['Düşük', 'Normal', 'Yüksek']
)

df['aile_hasta_toplam_x'] = (
    df['varsa_kimde_anne'] +
    df['varsa_kimde_baba'] +
    df['varsa_kimde_kardes'] +
    df['varsa_kimde_diger']
)

df['yas_grubu_x'] = pd.cut(df['yas'], bins=[0, 30, 50, float('inf')], labels=['Genç', 'Orta Yaş', 'Yaşlı'])

df['tansiyon_durumu_x'] = pd.cut(
    df['kan_basinci_sistolik'],
    bins=[0, 90, 120, 140, float('inf')],
    labels=['Düşük', 'Normal', 'Yüksek', 'Hipertansiyon']
)

df['kan_basinci_skoru_x'] = (
    df['kan_basinci_sistolik'] + (2 * df['kan_basinci_diastolik'])) / 3

df['solunum_verimliligi_x'] = df['PEF_yuzde'] / df['solunum_sayisi']

df['FEV1_FVC_kategori_x'] = pd.cut(
    df['FEV1_FVC_Değeri'],
    bins=[0, 0.7, 0.8, float('inf')],
    labels=['Düşük', 'Riskli', 'Normal']
)

df['toplam_yatis_suresi_x'] = (
    df['acil_servis_toplam_yatis_suresi_gun'] +
    df['yogun_bakim_toplam_yatis_suresi_gun'] +
    df['servis_toplam_yatis_suresi_gu']
)

df['risk_skoru_x'] = (
    df['tani_suresi_x'] * 0.5 +
    df['toplam_yatis_suresi_x'] * 1.0 +
    df['paket_yili_x'] * 0.8
)

df['PEF_kategori_x'] = pd.cut(
    df['PEF'],
    bins=[0, 250, 400, 600, float('inf')],
    labels=['Çok Düşük', 'Düşük', 'Orta', 'Yüksek']
)

df['PEF_normalize_FEV1_x'] = df['PEF'] / df['FEV1_yuzde']
df['PEF_yas_orani_x'] = df['PEF'] / df['yas']
pef_mean = df['PEF'].mean()
pef_std = df['PEF'].std()
df['PEF_zskoru_x'] = (df['PEF'] - pef_mean) / pef_std
df['PEF_tansiyon_skoru_x'] = df['PEF'] / df['kan_basinci_sistolik']

# Kullanılmayan kolonları kaldırma
drop_columns = [
    'tani_suresi_yil', 'tani_suresi_ay', 'hastaneye_yatti_mi', 'acil_servis_yatis_sayisi',
    'acil_servis_toplam_yatis_suresi_saat', 'acil_servis_toplam_yatis_suresi_gun',
    'servis_yatis_sayisi', 'servis_toplam_yatis_suresi_saat', 'servis_toplam_yatis_suresi_gu',
    'yogun_bakim_yatis_sayisi', 'yogun_bakim_toplam_yatis_suresi_saat',
    'yogun_bakim_toplam_yatis_suresi_gun', 'vucut_agirligi', 'boy'
]
df.drop(columns=drop_columns, inplace=True)

# Kategorik verilerin sayısallaştırılması
categorical_columns = ['nabiz_kategori_x', 'yas_grubu_x', 'tansiyon_durumu_x', 'FEV1_FVC_kategori_x', 'PEF_kategori_x']
for col in categorical_columns:
    df[col] = df[col].cat.codes  # Pandas kategorik kodlama (Düşük: 0, Normal: 1, ...)

# 5. Kategorik Verileri Sayısallaştırma
label_encoders = {}
for col in df.select_dtypes(include=["object"]):
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# 6. Bağımsız ve Bağımlı Değişkenleri Ayırma
X = df.drop(columns=['tani'])
y = df['tani']

# 7. Veriyi Eğitim ve Test Setine Ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# 8. Veri Standardizasyonu
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 9. Modeller ve Parametreler
define_models = {
    "SVM": SVC(),
    "k-NN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(),
    "MLP": MLPClassifier(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "Ensemble Tree": VotingClassifier(
        estimators=[
            ('rf', RandomForestClassifier(random_state=42)),
            ('gb', GradientBoostingClassifier(random_state=42)),
            ('dt', DecisionTreeClassifier(random_state=42))
        ],
        voting='soft'
    )
}

param_grids = {
    "SVM": {
        "C": [0.1, 1, 10],
        "gamma": [0.001, 0.01, 0.1],
        "kernel": ["linear", "rbf"]
    },
    "k-NN": {
        "n_neighbors": range(1, 21),
        "weights": ["uniform", "distance"]
    },
    "Random Forest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2]
    },
    "MLP": {
        "hidden_layer_sizes": [(50,), (100,)],
        "activation": ["relu", "tanh"],
        "solver": ["adam", "sgd"]
    },
    "Decision Tree": {
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2]
    },
    "Gradient Boosting": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1],
        "max_depth": [3, 5, 10]
    }
}

# 10. Model Eğitim ve Optimizasyon
best_params = {}
for model_name, model in define_models.items():
    if model_name in param_grids:
        print(f"\nOptimizing {model_name}...")
        grid = GridSearchCV(model, param_grids[model_name], scoring='accuracy', cv=3, verbose=1, n_jobs=-1)
        grid.fit(X_train, y_train)
        best_params[model_name] = grid.best_params_
        print(f"Best parameters for {model_name}: {grid.best_params_}")
        print(f"Best score: {grid.best_score_:.4f}")
    else:
        model.fit(X_train, y_train)

# 11. Performans Değerlendirme
for model_name, model in define_models.items():
    if model_name in best_params:
        model.set_params(**best_params[model_name])
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    accuracy = accuracy_score(y_test, y_test_pred)
    print(f"\n{model_name} Model")
    print(f"Eğitim R2: {train_r2:.4f}")
    print(f"Test R2: {test_r2:.4f}")
    print(f"Test Doğruluk: {accuracy:.4f}")
    print("Classification Report:\n", classification_report(y_test, y_test_pred))
