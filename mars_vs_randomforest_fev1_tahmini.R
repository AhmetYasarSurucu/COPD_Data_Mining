# Gerekli Kütüphanelerin Listesi
packages <- c("caret", "earth", "dplyr", "readxl", "writexl", "ggplot2", "randomForest", "gbm")

# Eksik Kütüphaneleri Yükleme
for (pkg in packages) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg, dependencies = TRUE)
    library(pkg, character.only = TRUE)
  }
}

# Veriyi Yükle
file_path <- "esofman.xlsx"
data <- read_excel(file_path)

# Eksik Değer Içermeyen Satırları Eğitim Verisi Olarak Kullan
train_data <- data %>% filter(!is.na(FEV1))

# Bağımsız ve Bağımlı Değişkenleri Ayır (PEF Dahil Edilmez)
X_train <- subset(train_data, select = -c(FEV1, PEF))
y_train <- train_data$FEV1

# Çapraz Doğrulama Kontrol Nesnesi
set.seed(42)
control <- trainControl(method = "cv", number = 10)  # 10 katlı çapraz doğrulama

# MARS Modeli Için Hiperparametre İzgara
mars_grid <- expand.grid(
  degree = 1:3,
  nprune = seq(2, 100, by = 5)
)

# Model Karşılaştırması
models <- list()

# 1. MARS Modeli
mars_model <- train(
  x = X_train, y = y_train,
  method = "earth",
  tuneGrid = mars_grid,
  trControl = control
)
models$MARS <- mars_model

# 2. Random Forest Modeli
rf_model <- train(
  x = X_train, y = y_train,
  method = "rf",
  trControl = control
)
models$RandomForest <- rf_model

# 3. Gradient Boosting Modeli
gbm_model <- train(
  x = X_train, y = y_train,
  method = "gbm",
  verbose = FALSE,
  trControl = control
)
models$GBM <- gbm_model

# Modelleri Karşılaştırma
results <- resamples(models)
summary(results)

# En İyi Modeli Belirleme
best_model <- models[[which.max(sapply(models, function(m) max(m$results$Rsquared)))]]
cat("En iyi model: ", names(models)[which.max(sapply(models, function(m) max(m$results$Rsquared)))], "\n")

# Test Verisi (Eksik Değerler)
test_data <- data %>% filter(is.na(FEV1))
X_test <- subset(test_data, select = -c(FEV1, PEF))  # PEF dahil edilmez

# Eksik Değerleri Tahmin Et
predicted_values <- predict(best_model, newdata = X_test)
data$FEV1[is.na(data$FEV1)] <- predicted_values

# Doldurulmuş Veri Setini Excel'e Yaz
output_file <- "esofman_fev1.xlsx"
write_xlsx(data, output_file)

# Model Performans Metrikleri
train_predictions <- predict(best_model, newdata = X_train)
r2_train <- caret::R2(train_predictions, y_train)
rmse_train <- caret::RMSE(train_predictions, y_train)
mae_train <- caret::MAE(train_predictions, y_train)

# Performans Sonuçları
cat("Eğitim Seti Performans\n")
cat("R^2:", r2_train, "\n")
cat("RMSE:", rmse_train, "\n")
cat("MAE:", mae_train, "\n")

# Test Performansı İçin Veri Hazırlama
test_indices <- createDataPartition(data$FEV1[!is.na(data$FEV1)], p = 0.2, list = FALSE)
test_actual <- data$FEV1[test_indices]
X_test_perf <- data[test_indices, ] %>% select(-c(FEV1, PEF))

# Test Setinde Tahmin Yapma
test_predictions <- predict(best_model, newdata = X_test_perf)

# Test Performans Metrikleri
r2_test <- caret::R2(test_predictions, test_actual)
rmse_test <- caret::RMSE(test_predictions, test_actual)
mae_test <- caret::MAE(test_predictions, test_actual)

# Test Sonuçları
cat("Test Seti Performans\n")
cat("R^2:", r2_test, "\n")
cat("RMSE:", rmse_test, "\n")
cat("MAE:", mae_test, "\n")

