#### This code uses svm with gaussian kernal, gradient boosted tree and
#### elastic net respectively on the original and the up-sampled data to 
#### create a total of 6 models. The 6 models are then blended together using
#### using a simple glm to create the final model
#### 5-fold cv is used to determine hyperparameters of the models and to 
#### evaluate the accuracy and the weighted penalty of the model according
#### to the given penalty matrix 
#### The whole process is run 10 times with different randomly chosen seeds.
#### In most of the time, the final model obtained by blending the 6 models
#### achieves higher accuracy and lower penalty than any of the 6 models.

library(caret)
data(GermanCredit)
# check if missing data
any(sapply(GermanCredit,function(x)length(which(is.na(x)==T)))!=0)
X <- GermanCredit[,-10]
Y <- GermanCredit[,10]

#### Delete linearly dependent ####
X_ldinfo <- findLinearCombos(X)
X <- X[,-X_ldinfo$remove]

#### Preprocess control ####
preproc <- c("nzv","zv","scale","center")

#### Custom metrics (Accuracy+WeightedPenalty ####
# Positive being "Bad"
false_neg_cost = 5
false_pos_cost = 1

customM <- function(data, lev = NULL, model = NULL) {
  penalty<- mean(ifelse(data[, "obs"]=="Bad",
                        false_neg_cost*(data[, "obs"]!=data[, "pred"]),
                        false_pos_cost*(data[, "obs"]!=data[, "pred"])))
  accu <- mean(data[, "obs"]==data[, "pred"])
  c(Accuracy= accu,WeightedPenalty=penalty)
}


#### Training control ####
fitControl<- trainControl(
  method = "cv",
  number=5,
  summaryFunction=customM)

fitControlUp <- trainControl(
  method = "cv",
  number = 5,
  sampling="up",
  summaryFunction=customM)

#### Seed ####
seedlist <- floor(runif(10)*1000)
for (i in 1:length(seedlist)) {
#seed <- 70
seed <- seedlist[i]

#### Split training testing ####
set.seed(seed)
Train_indx <- createDataPartition(GermanCredit$Class,p=.75,list=F)
Data_train <- cbind(X[Train_indx,],Class=Y[Train_indx])
Data_test <- cbind(X[-Train_indx,],Class=Y[-Train_indx])

#### SVM radial  ####
# original data
set.seed(seed)
svmgrid <- expand.grid(C=seq(0.1,3,0.1),
                       sigma=seq(0.01,0.1,0.01))
svmradial<-  train(Class ~ ., data = Data_train, 
                    method = "svmRadial",
                    trControl=fitControl,
                    preProcess= preproc,
                    tuneGrid=svmgrid)
#svmradial
#plot(svmradial)
#svmimportance <- varImp(svmradial)
#plot(svmimportance,top="20")
#confusionMatrix(data = predict(svmradial,Data_test), reference = Data_test$Class)

# up sampled train data
set.seed(seed)
svmradial2<-  train(Class ~ ., data = Data_train, 
                   method = "svmRadial",
                   trControl=fitControlUp,
                   preProcess= preproc,
                   tuneGrid=svmgrid)
#svmradial2
#svmradial2$bestTune
#plot(svmradial2)
#confusionMatrix(data = predict(svmradial2,Data_test), reference = Data_test$Class)

#### Gradient boosting ####
# original train data
set.seed(seed)
xgb<-  train(Class ~ ., data = Data_train,
             method = "xgbTree",
             preProcess=preproc,
             trControl=fitControl)
#xgb
#plot(xgb)
#xgbimportance <- varImp(xgb)
#plot(xgbimportance,top="20")
#confusionMatrix(data = predict(xgb,Data_test), reference = Data_test$Class)

# up sampled train data
set.seed(seed)
xgb2<-  train(Class ~ ., data = Data_train, 
             method = "xgbTree",
             preProcess=preproc,
             trControl=fitControlUp)
#xgb2
#plot(xgb2)
#confusionMatrix(data = predict(xgb2,Data_test), reference = Data_test$Class)

#### Elastic net ####
# original data
set.seed(seed)
enetgrid <-  expand.grid(alpha= seq(0.1,1,0.05),
                        lambda=seq(0.01,0.1,0.01))

enet<-  train(Class ~ ., data = Data_train, 
             method = "glmnet",
             preProcess=preproc,
             trControl=fitControl,
             tuneGrid=enetgrid)
#enet
#plot(enet)
#confusionMatrix(data = predict(enet,Data_test), reference = Data_test$Class)
#coef(enet$finalModel, enet$bestTune$lambda)
# up sampled train data
set.seed(seed)
enet2<-  train(Class ~ ., data = Data_train, 
              method = "glmnet",
              preProcess=preproc,
              trControl=fitControlUp,
              tuneGrid= enetgrid)
#plot(enet2)
#enet2$bestTune
#confusionMatrix(data = predict(enet2,Data_train), reference = Data_train$Class)
#confusionMatrix(data = predict(enet2,Data_test), reference = Data_test$Class)
#coef(enet$finalModel, enet$bestTune$lambda)


#### Compare models ####
# Performance
#resamps <- resamples(list(Enet=enet,
#                          Enet2=enet2,
#                          RadialSVM=svmradial,
#                          RadialSVM2=svmradial2,
#                          Xgb1 =xgb,
#                          Xgb2=xgb2))
                          
#bwplot(resamps)
#dotplot(resamps)
#summary(resamps)
#summary(diff(resamps))

# Correlation
#library(corrplot)
#modelCor(resamps)
#corrplot(modelCor(resamps))
#splom(resamps)
#findCorrelation(modelCor(resamps))


#### Ensemble models ###
Data_ensem <- data.frame(svm_1=predict(svmradial,Data_test),
                     svm_2= predict(svmradial2,Data_test),
                     enet_1=predict(enet,Data_test),
                     enet_2=predict(enet2,Data_test),
                     xgb_1=predict(xgb,Data_test),
                     xgb_2=predict(xgb2,Data_test))
Data_ensem$Class <- Data_test$Class
# blending using elastic net (too unstable)
#enet_blend<-  train(Class ~ ., data = Data_ensem, 
#                     method = "glmnet",
#                     trControl=fitControl,
#                    tuneGrid=enetgrid)
#enet_blend
#plot(enet_blend)
#enet_blend$bestTune

# blending using glm
glm_blend<-  train(Class ~ ., data = Data_ensem, 
                    method = "glm",
                    trControl=fitControl)
glm_blend
#### Compare2 ####
resamps2 <- resamples(list(Enet=enet,
                          Enet2=enet2,
                          RadialSVM=svmradial,
                          RadialSVM2=svmradial2,
                          Xgb1 =xgb,
                          Xgb2=xgb2,
                          Blended=glm_blend))
#bwplot(resamps2)
#dotplot(resamps2)
#summary(resamps2)
#class(resamps2)
out <- capture.output(summary(resamps2))
cat(paste("Round",i), out, file="C:/Users/Client/Desktop/result.txt", sep="n", fill=TRUE,append=TRUE)
print(paste("Round",i,"finished!"))
}