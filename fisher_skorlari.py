import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

# 1. Veri Yükleme
df = pd.read_excel("pca_reduced_features_with_original.xlsx")

# 2. Tarih Kolonunu Kaldırma (Modeli bozuyor)
df = df.select_dtypes(exclude=['datetime64'])

# 3. Eksik Değerleri Kontrol Etme
print("Eksik değerler:")
print(df.isnull().sum())

# Eksik verileri doldurmak (median ile sayısal değişkenler için)
df.fillna(df.median(numeric_only=True), inplace=True)

# 4. Kategorik Verileri Sayısallaştırma
label_encoders = {}
for col in df.select_dtypes(include=["object"]):
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# 5. Bağımsız ve Bağımlı Değişkenleri Ayırma
X = df.drop(columns=['tani'])  # Bağımsız değişkenler
y = df['tani']                # Bağımlı değişken

# 6. Veriyi Eğitim ve Test Setine Ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Fisher Skoruna Göre En İyi Özellik Seçimi
selector = SelectKBest(score_func=f_classif, k='all')  # Tüm özelliklerin skorunu hesapla
selector.fit(X_train, y_train)

# Fisher skorlarını ve sıralamaları alalım
scores = selector.scores_
feature_scores = pd.DataFrame({'Feature': X.columns, 'Fisher Score': scores})
feature_scores = feature_scores.sort_values(by='Fisher Score', ascending=False)

print("Fisher Skorlarına Göre En İyi Özellikler:")
print(feature_scores)

# En İyi K Özelliği Seçmek İçin (Opsiyonel):
# k = 10 gibi bir değer belirlerseniz, örneğin ilk 10 özelliği seçebilirsiniz.
# X_train_selected = selector.transform(X_train)
# X_test_selected = selector.transform(X_test)
