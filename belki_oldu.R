# Gerekli kütüphaneler
library(readxl)     # Excel dosyasını okumak için
library(earth)      # MARS modeli
library(caret)      # Çapraz doğrulama ve model değerlendirme
library(dplyr)      # Veri manipülasyonu

# Veri Yükleme
file_path <- "final_dataset4_a.xlsx"
df <- read_excel(file_path)

# Özellik Mühendisliği
# Fisher skorlarına göre önemli değişkenlere dayalı yeni özellikler ve temizleme
df <- df %>%
  mutate(
    # BMI hesaplama
    BMI = vucut_agirligi / (boy / 100)^2,
    
    # Sigara skoru (gün * adet)
    sigara_skoru = sigara_birakan_ne_kadar_gun_icmis * sigara_birakan_gunde_kac_adet_icmis,
    
    # TANI toplam süresi (ay olarak)
    tani_toplam_ay = tani_suresi_yil * 12 + tani_suresi_ay,
    
    # Acil servis kullanımı
    acil_servis_kullanimi = ifelse(
      acil_servis_yatis_sayisi > 0,
      acil_servis_toplam_yatis_suresi_saat / acil_servis_yatis_sayisi,
      0
    ),
    
    # PCA bileşen etkileşimi
    pca_interaction = Principal_Component_1 * Principal_Component_2,
    
    # Logaritmik dönüşüm
    log_ne_zaman_birakmis_gun = log1p(ne_zaman_birakmis_gun),
    
    # Solunum oranı (FEV1/PEF)
    solunum_orani = FEV1 / PEF,
    
    # Yaş grupları
    yas_grubu = case_when(
      yas < 30 ~ "Genç",
      yas >= 30 & yas < 50 ~ "Orta_Yaş",
      TRUE ~ "Yaşlı"
    ),
    
    # Tansiyon Skoru
    tansiyon_skoru = kan_basinci_sistolik / kan_basinci_diastolik,
    
    # Solunum Z-Skoru
    FEV1_zskoru = (FEV1 - mean(FEV1, na.rm = TRUE)) / sd(FEV1, na.rm = TRUE),
    PEF_zskoru = (PEF - mean(PEF, na.rm = TRUE)) / sd(PEF, na.rm = TRUE),
    
    # Solunum normalizasyonu (PEF/FEV1)
    solunum_normalize = PEF / FEV1,
    
    # Sigara ve yaş grubuna göre risk skoru
    sigara_yas_risk = case_when(
      yas_grubu == "Genç" ~ sigara_skoru * 0.5,
      yas_grubu == "Orta_Yaş" ~ sigara_skoru * 1.0,
      yas_grubu == "Yaşlı" ~ sigara_skoru * 1.5
    ),
    
    # Logaritmik PCA etkileşimi
    log_pca_interaction = log1p(abs(pca_interaction)),
    
    # FEV1 ve yaş etkisi
    fev1_yas_orani = FEV1 / yas
  )

# Seçilen önemli değişkenler ve türetilmiş özellikler
df_selected <- df %>%
  select(
    Principal_Component_1, FEV1_FVC_Değeri, cinsiyet, yas, 
    sigara_birakan_gunde_kac_adet_icmis, sigara_birakan_ne_kadar_gun_icmis,
    FEV1, FEV1_yuzde, PEF_yuzde, ne_zaman_birakmis_gun,
    BMI, sigara_skoru, tani_toplam_ay, acil_servis_kullanimi, 
    pca_interaction, log_ne_zaman_birakmis_gun, solunum_orani, 
    tansiyon_skoru, FEV1_zskoru, PEF_zskoru, solunum_normalize, 
    sigara_yas_risk, log_pca_interaction, fev1_yas_orani
  )

# Eğitim Verisi Hazırlama
X <- df_selected %>% select(-FEV1_zskoru)  # Hedef değişken hariç diğer özellikler
y <- df_selected$FEV1_zskoru              # Hedef değişken (örnek olarak FEV1_zskoru seçildi)
mars_full_data <- cbind(X, FEV1_zskoru = y)

# Hiperparametre Grid: Derece ve düğüm sayısı (nprune) optimizasyonu
mars_grid <- expand.grid(
  degree = c(1, 2, 3),         # MARS modelinin polinom dereceleri
  nprune = seq(2, 100, by = 2)  # Maksimum düğüm sayısı
)

# Çapraz Doğrulama Ayarları
control <- trainControl(
  method = "cv",             # Çapraz doğrulama
  number = 5,                # 5 katmanlı CV
  verboseIter = TRUE         # Model eğitimi sırasında ilerlemeyi göster
)

# MARS Model Eğitimi
mars_model <- train(
  FEV1_zskoru ~ .,            # Hedef değişken ve açıklayıcı değişkenler
  data = mars_full_data,      # Eğitim verisi
  method = "earth",           # MARS modeli
  tuneGrid = mars_grid,       # Hiperparametre grid
  trControl = control,        # Çapraz doğrulama ayarları
  metric = "Rsquared"         # Performans metriği
)

# En iyi model ve hiperparametreler
best_model <- mars_model$finalModel
best_params <- mars_model$bestTune
cat("En iyi model parametreleri:\n")
print(best_params)

# Çapraz Doğrulama Sonuçları
cv_results <- mars_model$results
cat("\nÇapraz doğrulama sonuçları:\n")
print(cv_results)

# Tahminler ve Performans Değerlendirme
predictions <- predict(best_model, newdata = X)
r2 <- cor(predictions, y)^2
cat("\nR^2 Skoru:", r2, "\n")

# Modeli Kaydetme (opsiyonel)
saveRDS(best_model, "mars_model.rds")
cat("\nModel 'mars_model.rds' olarak kaydedildi.\n")
