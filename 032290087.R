if (!require(titanic)) install.packages("titanic", dependencies=TRUE)
if (!require(dplyr)) install.packages("dplyr")
if (!require(ggplot2)) install.packages("ggplot2")
if (!require(naniar)) install.packages("naniar")

library(titanic)
library(dplyr)
library(ggplot2)
library(naniar)

data("titanic_train")
df <- titanic_train

str(df)
summary(df)
colSums(is.na(df))
vis_miss(df)

# kategorik değişkenlerin Survived ile ilişkisi
ggplot(df, aes(x = Sex, fill = factor(Survived))) +
  geom_bar(position = "fill") +
  labs(title = "Cinsiyete Göre Hayatta Kalma Oranları", y = "Oran", fill = "Hayatta Kaldı")

ggplot(df, aes(x = factor(Pclass), fill = factor(Survived))) +
  geom_bar(position = "fill") +
  labs(title = "Pclass'a Göre Hayatta Kalma Oranları", y = "Oran", x = "Pclass", fill = "Hayatta Kaldı")

ggplot(df, aes(x = Embarked, fill = factor(Survived))) +
  geom_bar(position = "fill") +
  labs(title = "Embarked'e Göre Hayatta Kalma", y = "Oran", fill = "Hayatta Kaldı")

# sayısal değişkenler ile Survived ilişkisi
ggplot(df, aes(x = factor(Survived), y = Age)) +
  geom_boxplot() +
  labs(title = "Yaşa Göre Hayatta Kalma", x = "Hayatta Kaldı", y = "Yaş")

ggplot(df, aes(x = factor(Survived), y = Fare)) +
  geom_boxplot() +
  labs(title = "Bilet Ücreti (Fare) ve Hayatta Kalma", x = "Hayatta Kaldı", y = "Fare")

##############################################################################################

# verinin kopyasını alıp çalışalım
data_clean <- df

# Age -> medyanla doldur
data_clean$Age[is.na(data_clean$Age)] <- median(data_clean$Age, na.rm = TRUE)

# Embarked -> mod ile doldur
mode_embarked <- names(sort(table(data_clean$Embarked), decreasing = TRUE))[1]
data_clean$Embarked[is.na(data_clean$Embarked)] <- mode_embarked

# Cabin değişkenini kaldır
data_clean <- data_clean %>% select(-Cabin)

# gereksiz sütunları çıkar
data_clean <- data_clean %>% select(-PassengerId, -Name, -Ticket)

# Survived'ı faktör yap
data_clean$Survived <- as.factor(data_clean$Survived)

# 4. One-hot encoding
if (!require(fastDummies)) install.packages("fastDummies")
library(fastDummies)

data_clean <- dummy_cols(data_clean, select_columns = c("Sex", "Embarked", "Pclass"), 
                         remove_selected_columns = TRUE, remove_first_dummy = TRUE)

##################################################################################################

if (!require(caret)) install.packages("caret")
library(caret)

set.seed(123)

train_index <- createDataPartition(data_clean$Survived, p = 0.7, list = FALSE)

train_data <- data_clean[train_index, ]
test_data  <- data_clean[-train_index, ]

cat("Eğitim seti boyutu:", nrow(train_data), "\n")
cat("Test seti boyutu:", nrow(test_data), "\n")

prop.table(table(train_data$Survived))
prop.table(table(test_data$Survived))

################################################################################################

if (!require(ranger)) install.packages("ranger")
library(ranger)

rf_model <- ranger(
  formula = Survived ~ .,
  data = train_data,
  probability = TRUE,
  importance = "impurity",
  num.trees = 500
)
# OOB tahmin doğruluğu
cat("OOB Error Rate: ", rf_model$prediction.error, "\n")

#######################################################################################

if (!require(caret)) install.packages("caret")
if (!require(ranger)) install.packages("ranger")
library(caret)
library(ranger)

set.seed(123)

# 10 katlı çapraz doğrulama
ctrl <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary)

# 0,1 yerine Yes,No
train_data$Survived <- factor(ifelse(train_data$Survived == 1, "Yes", "No"))

rf_grid <- expand.grid(
  mtry = c(2, 3, 4, 5),
  splitrule = c("gini"),
  min.node.size = c(1, 5, 10)
)

rf_tuned <- train(
  Survived ~ .,
  data = train_data,
  method = "ranger",
  tuneGrid = rf_grid,
  trControl = ctrl,
  metric = "ROC",
  importance = "impurity"
)

# En iyi parametreler
print(rf_tuned$bestTune)

print(rf_tuned)

varImpPlot <- varImp(rf_tuned)
plot(varImpPlot, top = 10)

#######################################################################################

if (!require(pROC)) install.packages("pROC")
library(pROC)

test_data$Survived <- factor(ifelse(test_data$Survived == 1, "Yes", "No"))

rf_probs <- predict(rf_tuned, newdata = test_data, type = "prob")

rf_preds <- predict(rf_tuned, newdata = test_data)

# Confusion Matrix
conf_mat <- confusionMatrix(rf_preds, test_data$Survived, positive = "Yes")
print(conf_mat)

# ROC eğrisi ve AUC
roc_obj <- roc(response = test_data$Survived, predictor = rf_probs$Yes, levels = c("No", "Yes"))
plot(roc_obj, main = "Random Forest ROC Curve", col = "blue")
cat("AUC: ", auc(roc_obj), "\n")


#########################################################################################

install.packages("xgboost")
library(xgboost)

# hedefi tekrar 0/1 yap (numeric)
train_data_xgb <- train_data
test_data_xgb <- test_data

train_data_xgb$Survived <- ifelse(train_data_xgb$Survived == "Yes", 1, 0)
test_data_xgb$Survived <- ifelse(test_data_xgb$Survived == "Yes", 1, 0)

# Ayrı hedef vektörü
train_label <- train_data_xgb$Survived
test_label <- test_data_xgb$Survived

# Girdi verileri (Survived hariç tüm sütunlar)
train_matrix <- as.matrix(train_data_xgb %>% select(-Survived))
test_matrix  <- as.matrix(test_data_xgb %>% select(-Survived))

# XGBoost DMatrix objesi
dtrain <- xgb.DMatrix(data = train_matrix, label = train_label)
dtest  <- xgb.DMatrix(data = test_matrix, label = test_label)


###################################################################################

if (!require(xgboost)) install.packages("xgboost")
library(xgboost)

# Parametreler
params <- list(
  objective = "binary:logistic",
  eval_metric = "auc",
  max_depth = 3,
  eta = 0.1,
  subsample = 0.8,
  colsample_bytree = 0.8
)

# Çapraz doğrulama ile en iyi iterasyonu bul
set.seed(123)
cv_model <- xgb.cv(
  params = params,
  data = dtrain,
  nrounds = 100,
  nfold = 5,
  early_stopping_rounds = 10,
  print_every_n = 10,
  maximize = TRUE
)

# En iyi iterasyon sayısı
best_iter <- cv_model$best_iteration
cat("En iyi iterasyon sayısı:", best_iter, "\n")

###################################################################################

param_grid <- expand.grid(
  eta = c(0.01, 0.1),
  max_depth = c(3, 5),
  min_child_weight = c(1, 5),
  subsample = c(0.8, 1),
  colsample_bytree = c(0.8, 1),
  gamma = c(0, 1)
)

# Sonuçları tutmak için boş liste
results <- list()

set.seed(123)

# Grid üzerinde döngü ile en iyi kombinasyonu bul
for (i in 1:nrow(param_grid)) {
  params <- list(
    objective = "binary:logistic",
    eval_metric = "auc",
    eta = param_grid$eta[i],
    max_depth = param_grid$max_depth[i],
    min_child_weight = param_grid$min_child_weight[i],
    subsample = param_grid$subsample[i],
    colsample_bytree = param_grid$colsample_bytree[i],
    gamma = param_grid$gamma[i]
  )
  
  cv <- xgb.cv(
    params = params,
    data = dtrain,
    nrounds = 100,
    nfold = 5,
    early_stopping_rounds = 10,
    verbose = 0
  )
  
  results[[i]] <- list(
    auc = max(cv$evaluation_log$test_auc_mean),
    iter = cv$best_iteration,
    params = params
  )
  
  cat("Grid", i, "AUC:", results[[i]]$auc, "\n")
}

# En iyi sonucu seç
best_index <- which.max(sapply(results, function(x) x$auc))
best_params <- results[[best_index]]$params
best_nrounds <- results[[best_index]]$iter

cat("\nEn iyi parametreler:\n")
print(best_params)
cat("En iyi iterasyon:", best_nrounds, "\n")

#########################################################################################

# tüm eğitim verisi ile modeli eğitmr
final_model <- xgboost(
  params = best_params,
  data = dtrain,
  nrounds = best_nrounds,
  verbose = 0
)

# Test verisi üzerindeki tahminler
xgb_probs <- predict(final_model, dtest)

# Olasılıklardan sınıf tahmini üret (0.5 eşiği)
xgb_preds <- ifelse(xgb_probs > 0.5, 1, 0)

# Confusion Matrix
conf_matrix <- table(Predicted = xgb_preds, Actual = test_label)
print(conf_matrix)

library(caret)
library(pROC)

xgb_preds_factor <- factor(xgb_preds, levels = c(0, 1))
test_label_factor <- factor(test_label, levels = c(0, 1))

conf <- confusionMatrix(xgb_preds_factor, test_label_factor, positive = "1")
print(conf)

roc_obj <- roc(test_label, xgb_probs)
plot(roc_obj, main = "XGBoost ROC Curve", col = "red")
cat("AUC:", auc(roc_obj), "\n")




