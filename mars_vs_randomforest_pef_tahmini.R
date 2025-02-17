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
file_path <- "esofman_fev1.xlsx"
data <- read_excel(file_path)

# Eksik Değer İçermeyen Satırları Eğitim Verisi Olarak Kullan
train_data <- data %>% filter(!is.na(PEF))

# Bağımsız ve Bağımlı Değişkenleri Ayır
X_train <- subset(train_data, select = -c(PEF))
y_train <- train_data$PEF

# Çapraz Doğrulama Kontrol Nesnesi
set.seed(42)
control <- trainControl(method = "cv", number = 10)  # 10 katlı çapraz doğrulama

# MARS Modeli İçin Hiperparametre İzgara
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
test_data <- data %>% filter(is.na(PEF))
X_test <- subset(test_data, select = -c(PEF))

# Eksik Değerleri Tahmin Et
predicted_values <- predict(best_model, newdata = X_test)
data$PEF[is.na(data$PEF)] <- predicted_values

# Doldurulmuş Veri Setini Excel'e Yaz
output_file <- "esofman_pef_bitti.xlsx"
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
test_indices <- createDataPartition(data$PEF[!is.na(data$PEF)], p = 0.2, list = FALSE)
test_actual <- data$PEF[test_indices]
X_test_perf <- data[test_indices, ] %>% select(-c(PEF))

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

