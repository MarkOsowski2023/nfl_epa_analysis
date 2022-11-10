library(tidyverse)
library(caret)
library(rminer)
# dataset will be the game data 

data_set <- game_data %>%
  select(-team, -game_id)

# check the dimensions of the data
str(data_set)
class(data_set$result)

# convert result column from character to factor
data_set$result <- as.factor(data_set$result)
class(data_set$result)
str(data_set)

# scale
dataset <- data_set %>%
  mutate(across(where(is.numeric), ~ scale(.x)))

data.pre <- preProcess(data_set, method = c("center", "scale"))
data_scaled <- predict(data.pre, data_set)

# validation set and training set

validation_index <- createDataPartition(data_scaled$result, p=0.80, list = FALSE)

validation_set <- data_scaled[-validation_index,]

train_test_set <- data_scaled[validation_index,]

str(train_test_set)

# summarise the test and train data

summary(train_test_set)


# visualize

# split
x <- train_test_set[,2:13]
y <- train_test_set[,1]

# boxplots for each attribute
par(mfrow=c(3,4))
for (i in 1:12) {
  boxplot(x[,i], main=names(x)[i])
}

# bar plot for factor breakdown
plot(y)

# mutlivariant plots
# scatterplot matrix

trellis.par.set(theme = col.whitebg(), warn = FALSE)
featurePlot(x=train_test_set[, 2:5], y=train_test_set$result, plot="box")
featurePlot(x=train_test_set[, 6:9], y=train_test_set$result, plot = "box")
featurePlot(x=train_test_set[, 10:13], y=train_test_set$result, plot = "box")


# evaluate algorithms
control <- trainControl(method = "cv", number = 10)
metric <- "Accuracy"


# linear algo
set.seed(7)
fit.lda <- train(result~., data = train_test_set, method = "lda", metric=metric, trControl=control)

# nonlinear cart
set.seed(7)
fit.cart <- train(result~., data = train_test_set, method = "rpart", metric=metric, trControl=control)

# nonlinear knn
set.seed(7)
fit.knn <- train(result~., data = train_test_set, method = "knn", metric=metric, trControl=control)

# svm
set.seed(7)
fit.svm <- train(result~., data = train_test_set, method = "svmRadial", metric=metric, trControl=control)

# random forest
set.seed(7)
fit.rf <- train(result~., data = train_test_set, method = "rf", metric=metric, trControl=control)


# summary of algo accuracy
model_results <- resamples(list(lda=fit.lda, cart=fit.cart, knn=fit.knn, svm=fit.svm, rf=fit.rf))
summary(model_results)


# dotplot
dotplot(model_results)

# summarize best model
print(fit.svm)

# make predictions on validation set

predictions_svm <- predict(fit.svm, validation_set)

# confusion matrix
confusionMatrix(predictions_svm, validation_set$result)

# rank features by importance
importancesvm <- varImp(fit.svm, data = train_test_set)
print(importance_svm)
