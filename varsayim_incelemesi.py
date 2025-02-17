import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, kstest, probplot
from statsmodels.stats.diagnostic import het_breuschpagan
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from statsmodels.api import OLS, add_constant
from sklearn.preprocessing import StandardScaler
import openpyxl

# Dosyayı okuma
df = pd.read_excel('final_dataset_pef2.xlsx')

# Verinin ilk 5 satırını kontrol etme
print("DataFrame'in İlk 5 Satırı:")
print(df.head())

# Eksik veri kontrolü
print("\nEksik Gözlemler:")
missing_values = df.isnull().sum()
print(missing_values)

# Sadece belirtilen sayısal değişkenleri seçme (kontrol edilen sütunlar)
selected_columns = [
    'yas', 'sigara_birakan_ne_kadar_gun_icmis', 'sigara_birakan_gunde_kac_adet_icmis',
    'ne_zaman_birakmis_gun', 'sigara_devam_eden_gunde_kac_adet_iciyor', 'tani_suresi_yil',
    'tani_suresi_ay', 'acil_servis_yatis_sayisi', 'acil_servis_toplam_yatis_suresi_saat',
    'acil_servis_toplam_yatis_suresi_gun', 'yogun_bakim_yatis_sayisi',
    'yogun_bakim_toplam_yatis_suresi_gun',
    'servis_yatis_sayisi', 'servis_toplam_yatis_suresi_gu',
    'boy', 'vucut_agirligi', 'kan_basinci_sistolik', 'kan_basinci_diastolik', 'nabiz',
    'solunum_sayisi', 'FEV1', 'FEV1_yuzde', 'PEF', 'PEF_yuzde', 'FEV1_FVC_Değeri'
]

# Mevcut sütunları kontrol etme
selected_columns = [col for col in selected_columns if col in df.columns]

# Aykırı gözlem analizi
print("\n\nAykırı Gözlemler (Seçili Değişkenler)")
results = []
fig, axes = plt.subplots((len(selected_columns) + 3) // 4, 4, figsize=(20, 8 * ((len(selected_columns) + 3) // 4)))
axes = axes.flatten()

for idx, col in enumerate(selected_columns):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    results.append({"Değişken": col, "Aykırı Sayısı": len(outliers)})
    print(f"{col} değişkenindeki aykırı gözlem sayısı: {len(outliers)}")

    sns.boxplot(x=df[col], ax=axes[idx])
    axes[idx].set_title(f'Boxplot: {col}')

# Boş eksenleri gizle
for i in range(idx + 1, len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.show()

# Sonuçları Excel'e kaydetme
results_df = pd.DataFrame(results)
results_df.to_excel("secilmis_aykiri_degerler.xlsx", index=False)
print("Sonuçlar 'secilmis_aykiri_degerler.xlsx' dosyasına kaydedildi.")
