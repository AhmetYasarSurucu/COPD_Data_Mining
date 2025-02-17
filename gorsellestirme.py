import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Veriyi yükleme
file_path = 'esofman_pef_bitti.xlsx'  # Dosya yolu
data = pd.read_excel(file_path)

# Değişken tanımlamaları
variable_descriptions = {
    'cinsiyet': 'Cinsiyet',
    'sigara_kullanimi': 'Sigara Kullanımı',
    'tani': 'Tanı (ASTIM/KOAH)',
}

# Kategorik değişkenler için countplot (boşluksuz sütunlar)
categorical_columns = ['cinsiyet', 'sigara_kullanimi', 'tani']

for column in categorical_columns:
    plt.figure(figsize=(8, 6))
    sns.countplot(x=data[column], palette='viridis', order=data[column].value_counts().index)
    plt.title(f"{variable_descriptions.get(column, column)} Dağılımı")
    plt.xlabel(variable_descriptions.get(column, column))
    plt.ylabel("Frekans")
    plt.show()


# 2. Değişken tanımlamalarını oluşturma
variable_descriptions = {
    'hasta_no': 'Hasta No',
    'yas': 'Yaş',
    'cinsiyet': 'Cinsiyet',
    'egitim_duzeyi': 'Eğitim Düzeyi',
    'meslek': 'Meslek',
    'sigara_kullanimi': 'Sigara Kullanımı',
    'sigara_birakan_ne_kadar_gun_icmis': 'Sigara İçme Süresi (Gün)',
    'sigara_birakan_gunde_kac_adet_icmis': 'Günde İçilen Sigara Adedi',
    'ne_zaman_birakmis_gun': 'Sigara Bırakma Süresi (Gün)',
    'tani': 'Tanı (ASTIM/KOAH)',
    'tani_suresi_yil': 'Tanı Süresi (Yıl)',
    'FEV1': 'FEV1 (lt)',
    'FEV1_yuzde': 'FEV1 (%)',
    'PEF': 'PEF (lt)',
    'PEF_yuzde': 'PEF (%)',
    'boy': 'Boy (cm)',
    'vucut_agirligi': 'Vücut Ağırlığı (kg)',
    'kan_basinci_sistolik': 'Sistolik Kan Basıncı (mmHg)',
    'kan_basinci_diastolik': 'Diastolik Kan Basıncı (mmHg)',
    'nabiz': 'Nabız (Vuru/Dk)',
    'solunum_sayisi': 'Solunum Sayısı',
    'ailede_koah_veya_astim_tanili_hasta_var_mi': 'Ailede KOAH/Astım Tanısı Var mı?'
}


# Örnek Görselleştirme 1: Yaş ve PEF Arasındaki İlişki
if 'yas' in data.columns and 'PEF' in data.columns:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='yas', y='PEF', hue='tani', data=data, palette='coolwarm')
    plt.title("Yaş ile PEF Arasındaki İlişki")
    plt.xlabel(variable_descriptions['yas'])
    plt.ylabel(variable_descriptions['PEF'])
    plt.legend(title=variable_descriptions['tani'])
    plt.show()

# Örnek Görselleştirme 2: Sigara İçme Süresi ve FEV1 (%) Arasındaki İlişki
if 'sigara_birakan_ne_kadar_gun_icmis' in data.columns and 'FEV1_yuzde' in data.columns:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='sigara_birakan_ne_kadar_gun_icmis', y='FEV1_yuzde', hue='tani', data=data, palette='viridis')
    plt.title("Sigara İçme Süresi (Gün) ile FEV1 (%) Arasındaki İlişki")
    plt.xlabel(variable_descriptions['sigara_birakan_ne_kadar_gun_icmis'])
    plt.ylabel(variable_descriptions['FEV1_yuzde'])
    plt.legend(title=variable_descriptions['tani'])
    plt.show()

# Örnek Görselleştirme 3: Vücut Ağırlığı ve Kan Basıncı Arasındaki İlişki
if 'vucut_agirligi' in data.columns and 'kan_basinci_sistolik' in data.columns:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='vucut_agirligi', y='kan_basinci_sistolik', hue='tani', data=data, palette='magma')
    plt.title("Vücut Ağırlığı ile Sistolik Kan Basıncı Arasındaki İlişki")
    plt.xlabel(variable_descriptions['vucut_agirligi'])
    plt.ylabel(variable_descriptions['kan_basinci_sistolik'])
    plt.legend(title=variable_descriptions['tani'])
    plt.show()

# Örnek Görselleştirme 4: Nabız ve Solunum Sayısı Arasındaki İlişki
if 'nabiz' in data.columns and 'solunum_sayisi' in data.columns:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='nabiz', y='solunum_sayisi', hue='tani', data=data, palette='cubehelix')
    plt.title("Nabız ile Solunum Sayısı Arasındaki İlişki")
    plt.xlabel(variable_descriptions['nabiz'])
    plt.ylabel(variable_descriptions['solunum_sayisi'])
    plt.legend(title=variable_descriptions['tani'])
    plt.show()

# Örnek 1: Yaş ve FEV1 Arasındaki İlişki (Ailede hasta varlığına göre)
if 'yas' in data.columns and 'FEV1' in data.columns and 'ailede_koah_veya_astim_tanili_hasta_var_mi' in data.columns:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='yas', y='FEV1', hue='ailede_koah_veya_astim_tanili_hasta_var_mi', data=data, palette='coolwarm')
    plt.title("Yaş ile FEV1 Arasındaki İlişki (Ailede KOAH/Astım Hasta Durumuna Göre)")
    plt.xlabel(variable_descriptions['yas'])
    plt.ylabel(variable_descriptions['FEV1'])
    plt.legend(title=variable_descriptions['ailede_koah_veya_astim_tanili_hasta_var_mi'])
    plt.show()

# Örnek 2: Sigara İçme Süresi ve FEV1 Yüzdesi Arasındaki İlişki
if 'sigara_birakan_ne_kadar_gun_icmis' in data.columns and 'FEV1_yuzde' in data.columns and 'ailede_koah_veya_astim_tanili_hasta_var_mi' in data.columns:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='sigara_birakan_ne_kadar_gun_icmis', y='FEV1_yuzde', hue='ailede_koah_veya_astim_tanili_hasta_var_mi', data=data, palette='viridis')
    plt.title("Sigara İçme Süresi (Gün) ile FEV1 (%) Arasındaki İlişki (Ailede KOAH/Astım Hasta Durumuna Göre)")
    plt.xlabel(variable_descriptions['sigara_birakan_ne_kadar_gun_icmis'])
    plt.ylabel(variable_descriptions['FEV1_yuzde'])
    plt.legend(title=variable_descriptions['ailede_koah_veya_astim_tanili_hasta_var_mi'])
    plt.show()

# Örnek 3: Vücut Ağırlığı ve Sistolik Kan Basıncı Arasındaki İlişki
if 'vucut_agirligi' in data.columns and 'kan_basinci_sistolik' in data.columns and 'ailede_koah_veya_astim_tanili_hasta_var_mi' in data.columns:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='vucut_agirligi', y='kan_basinci_sistolik', hue='ailede_koah_veya_astim_tanili_hasta_var_mi', data=data, palette='magma')
    plt.title("Vücut Ağırlığı ile Sistolik Kan Basıncı Arasındaki İlişki (Ailede KOAH/Astım Hasta Durumuna Göre)")
    plt.xlabel(variable_descriptions['vucut_agirligi'])
    plt.ylabel(variable_descriptions['kan_basinci_sistolik'])
    plt.legend(title=variable_descriptions['ailede_koah_veya_astim_tanili_hasta_var_mi'])
    plt.show()

# Örnek 4: Nabız ve Solunum Sayısı Arasındaki İlişki
if 'nabiz' in data.columns and 'solunum_sayisi' in data.columns and 'ailede_koah_veya_astim_tanili_hasta_var_mi' in data.columns:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='nabiz', y='solunum_sayisi', hue='ailede_koah_veya_astim_tanili_hasta_var_mi', data=data, palette='cubehelix')
    plt.title("Nabız ile Solunum Sayısı Arasındaki İlişki (Ailede KOAH/Astım Hasta Durumuna Göre)")
    plt.xlabel(variable_descriptions['nabiz'])
    plt.ylabel(variable_descriptions['solunum_sayisi'])
    plt.legend(title=variable_descriptions['ailede_koah_veya_astim_tanili_hasta_var_mi'])
    plt.show()

# 3. Tek Değişkenli Analiz
# Sürekli değişkenler için histogram ve kutu grafikleri
numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns

for column in numeric_columns:
    plt.figure(figsize=(12, 6))

    # Histogram
    plt.subplot(1, 2, 1)
    sns.histplot(data[column].dropna(), kde=True, color='skyblue')
    plt.title(f"{variable_descriptions.get(column, column)} Dağılımı")
    plt.xlabel(variable_descriptions.get(column, column))
    plt.ylabel("Frekans")

    # Kutu grafiği
    plt.subplot(1, 2, 2)
    sns.boxplot(x=data[column], color='lightgreen')
    plt.title(f"{variable_descriptions.get(column, column)} Kutu Grafiği")
    plt.xlabel(variable_descriptions.get(column, column))

    plt.tight_layout()
    plt.show()

# Kategorik değişkenler için barplot
categorical_columns = ['cinsiyet', 'sigara_kullanimi', 'tani', 'ailede_koah_veya_astim_tanili_hasta_var_mi']

for column in categorical_columns:
    plt.figure(figsize=(8, 6))
    value_counts = data[column].value_counts().sort_index()
    sns.barplot(x=value_counts.index, y=value_counts.values, palette='viridis')
    plt.title(f"{variable_descriptions.get(column, column)} Dağılımı")
    plt.xlabel(variable_descriptions.get(column, column))
    plt.ylabel("Frekans")
    plt.show()

# 4. Çok Değişkenli Analiz
# Korelasyon matrisi
plt.figure(figsize=(10, 8))
correlation_matrix = data[numeric_columns].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Numerik Değişkenler Arası Korelasyon Matrisi")
plt.show()



# 5. Örnek Görselleştirme: Yaş ve FEV1 Scatterplot
if 'yas' in data.columns and 'FEV1' in data.columns:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='yas', y='FEV1', hue='tani', data=data, palette='coolwarm')
    plt.title("Yaş ile FEV1 Arasındaki İlişki")
    plt.xlabel(variable_descriptions['yas'])
    plt.ylabel(variable_descriptions['FEV1'])
    plt.legend(title=variable_descriptions['tani'])
    plt.show()
