import pandas as pd

# Excel dosyasını yükle
file_path = 'final_dataset4.xlsx'  # Dosya adını uygun şekilde değiştirin
data = pd.ExcelFile(file_path)

# İlk sayfayı yükle
df = data.parse('Sayfa1')

# Tarih sütunlarını hariç tutmak için tarih formatında olmayan sütunları filtrele
non_date_columns = df.select_dtypes(exclude=['datetime64[ns]', 'datetime64', 'timedelta64']).columns

# En çok tekrar eden değerleri ve sayısını hesapla
most_frequent_df = pd.DataFrame({
    'Column': non_date_columns,
    'Most Frequent Value': [df[col].mode().iloc[0] if not df[col].mode().empty else None for col in non_date_columns],
    'Frequency': [df[col].value_counts().iloc[0] if not df[col].value_counts().empty else 0 for col in non_date_columns]
})

# En çok tekrar eden değerler tablosunu göster
print(most_frequent_df)

# CSV olarak kaydetmek isterseniz:
# most_frequent_df.to_csv('most_frequent_values.csv', index=False)
