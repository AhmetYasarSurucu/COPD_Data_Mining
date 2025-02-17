import pandas as pd
from sklearn.preprocessing import StandardScaler

# 1. Veriyi yükleme
file_path = 'final_dataset4_asx.xlsx'  # Dosyanın adı ve yolu
data = pd.read_excel(file_path)


# 2. Standartlaştırma fonksiyonu
def standardize_numeric_columns(df):
    """
    Sayısal sütunlara standartlaştırma uygular, encoding yapılmış sütunları hariç tutar.
    """
    # Sadece sayısal sütunları seçelim
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns

    # Encoding yapılmış sütunları dışarıda bırakmak için belirlenen sütunlar
    excluded_columns = [
        'cinsiyet', 'egitim_duzeyi', 'meslek', 'sigara_kullanimi',
        'hastaneye_yatti_mi', 'ailede_koah_veya_astim_tanili_hasta_var_mi',
        'varsa_kimde_anne', 'varsa_kimde_baba', 'varsa_kimde_kardes', 'varsa_kimde_diger'
    ]

    # Standartlaştırma yapılacak sütunları seç
    standardize_columns = [col for col in numeric_columns if col not in excluded_columns]

    # Sadece standartlaştırılacak sütunlar için scaler oluştur
    scaler = StandardScaler()
    df[standardize_columns] = scaler.fit_transform(df[standardize_columns])

    return df


# 3. Standartlaştırmayı uygula
standardized_data = standardize_numeric_columns(data)

# 4. Dönüştürülmüş veriyi kontrol etme
print("Standartlaştırma uygulandı! İlk 5 satır:")
print(standardized_data.head())

# 5. Yeni veriyi kaydetme
output_file_path = 'asd1.xlsx'  # Yeni dosya adı
standardized_data.to_excel(output_file_path, index=False)
print(f"Standartlaştırılmış veri başarıyla kaydedildi: {output_file_path}")
