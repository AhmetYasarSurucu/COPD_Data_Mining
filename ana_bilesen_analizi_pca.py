
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Veriyi yükleme
file_path = 'asd1.xlsx'  # Dosyanın adı ve yolu
data = pd.read_excel(file_path)

# 1. PCA uygulanacak değişkenleri seçme
selected_columns = [
    'sigara_kullanimi',
    'ailede_koah_veya_astim_tanili_hasta_var_mi',
    'cinsiyet',
    'sigara_birakan_ne_kadar_gun_icmis',
    'servis_yatis_sayisi',
    'egitim_duzeyi',
    'sigara_birakan_gunde_kac_adet_icmis'
]

# Seçilen değişkenleri filtrele
data_pca = data[selected_columns].dropna()

# 2. Veriyi standartlaştırma
scaler = StandardScaler()
data_pca_scaled = scaler.fit_transform(data_pca)

# 3. PCA modelini oluşturma ve uygulama
pca = PCA(n_components=2)  # 2 ana bileşen oluştur
pca_components = pca.fit_transform(data_pca_scaled)

# 4. Ana bileşenleri DataFrame'e dönüştürme
pca_df = pd.DataFrame(data=pca_components, columns=['Principal_Component_1', 'Principal_Component_2'])

# 5. PCA bileşenlerini orijinal veri setine ekleme
data_with_pca = pd.concat([data.reset_index(drop=True), pca_df], axis=1)

# 6. PCA açıklama oranlarını yazdırma
explained_variance = pca.explained_variance_ratio_
print(f"PCA Açıklama Oranları: {explained_variance}")

# 7. Sonuçları kaydetme
output_file_path = 'asd2.xlsx'
data_with_pca.to_excel(output_file_path, index=False)
print(f"PCA sonuçları başarıyla kaydedildi: {output_file_path}")
