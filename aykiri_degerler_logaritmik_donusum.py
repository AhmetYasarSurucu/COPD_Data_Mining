import pandas as pd
import numpy as np

# 1. Veriyi yükleme
file_path = 'esofman_pef_bitti.xlsx'  # Dosyanın adı ve yolu
data = pd.read_excel(file_path)


# 2. Log dönüşümünü uygulama
def log_transform_numeric_columns(df):
    """
    Sayısal sütunlara log dönüşümü uygular, encoding sütunlarını hariç tutar.
    """
    # Sadece sayısal sütunları seçelim
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns

    # Encoding yapılmış sütunları dışarıda bırakmak için belirlenen sütunlar
    excluded_columns = [
        'cinsiyet', 'egitim_duzeyi', 'meslek', 'sigara_kullanimi',
        'hastaneye_yatti_mi', 'ailede_koah_veya_astim_tanili_hasta_var_mi',
        'varsa_kimde_anne', 'varsa_kimde_baba', 'varsa_kimde_kardes', 'varsa_kimde_diger'
    ]

    # Log dönüşüm yapılacak sütunları seç
    log_columns = [col for col in numeric_columns if col not in excluded_columns]

    # Log dönüşümünü uygula (pozitif değerlere)
    for col in log_columns:
        df[col] = df[col].apply(lambda x: np.log1p(x) if x > 0 else 0)

    return df


# 3. Log dönüşümünü uygulama
transformed_data = log_transform_numeric_columns(data)

# 4. Dönüştürülmüş veriyi kontrol etme
print("Log dönüşümü uygulandı! İlk 5 satır:")
print(transformed_data.head())

# 5. Yeni veriyi kaydetme
output_file_path = 'esofman_pef_bitti_transformed.xlsx'  # Yeni dosya adı
transformed_data.to_excel(output_file_path, index=False)
print(f"Log dönüşümü uygulanmış veri başarıyla kaydedildi: {output_file_path}")
