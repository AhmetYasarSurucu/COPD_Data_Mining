import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Veriyi yükleme
file_path = 'esofman_pef_bitti.xlsx'
data = pd.read_excel(file_path)

# 2. Sayısal değişkenlerin seçilmesi
numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns
data_numeric = data[numeric_columns]

# 3. Korelasyon matrisinin hesaplanması
correlation_matrix = data_numeric.corr()

# 4. Korelasyon grafiği (ısı haritası) - Düzenlenmiş
plt.figure(figsize=(16, 16))  # Daha geniş boyut
sns.set(font_scale=0.5)  # Font boyutunu küçültme

# Isı haritası oluşturma
sns.heatmap(correlation_matrix,
            annot=True,        # Hücre içi değerleri göster
            fmt=".2f",         # Ondalık formatı
            cmap="coolwarm",   # Renk teması
            linewidths=0.2,    # Hücre kenarlıkları
            cbar_kws={'shrink': 0.5},  # Renk çubuğu küçültme
            square=True)       # Hücreleri kare yapma

# Başlık ve ayarlar
plt.title("Okunabilir Detaylı Korelasyon Grafiği", fontsize=10)
plt.xticks(rotation=45, ha="right")  # X ekseni isimlerini döndürme
plt.yticks(rotation=0)  # Y ekseni isimlerini düz bırakma
plt.tight_layout()
plt.show()
