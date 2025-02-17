import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Veriyi yükleme
file_path = 'esofman_pca.xlsx'  # Dosyanın adı ve yolu
data = pd.read_excel(file_path)

# 1. VIF Hesaplama
def calculate_vif(df):
    """
    Veri setindeki her bağımsız değişken için VIF hesaplar.
    """
    # Sadece sayısal sütunları seçelim (encoding sütunları hariç tutulabilir)
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    X = df[numeric_columns].dropna()  # Eksik değerler varsa kaldırılır
    vif_data = pd.DataFrame()
    vif_data['Feature'] = X.columns
    vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data

# VIF hesaplama
vif_results = calculate_vif(data)
print("VIF Analizi Sonuçları:")
print(vif_results)
