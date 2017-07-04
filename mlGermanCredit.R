#### This code uses svm with gaussian kernal, gradient boosted tree and
#### elastic net respectively on an up-sampled data to create 3 base models. 
#### The 3 base models are then blended together using a simple glm to 
#### create the final model.
wd = "C:/Users/Client/Desktop/"
setwd(wd)

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

#### Custom metrics  ####
# Positive being "Bad"
weightedAccu <- function(obs,pred,positive,false_pos_cost,false_neg_cost) {
return(1-(sum(ifelse(obs==positive,
                     false_neg_cost*(obs!=pred),
                     false_pos_cost*(obs!=pred)))/
            (sum(obs==positive)*false_neg_cost+sum(obs!=positive))))  
}

customM <- function(data, lev = NULL, model = NULL) {
  accu_w <- weightedAccu(obs=data[,"obs"],pred=data[,"pred"],
                         positive="Bad",
                         false_pos_cost=1,
                         false_neg_cost=5)
  accu <- mean(data[, "obs"]==data[, "pred"])
  precis <- (sum(data[, "pred"]=="Bad"&data[, "obs"]=="Bad")/sum(data[, "pred"]=="Bad"))
  recall <- (sum(data[, "pred"]=="Bad"&data[, "obs"]=="Bad")/sum(data[, "obs"]=="Bad"))
  specific <- (sum(data[, "pred"]=="Good"&data[, "obs"]=="Good")/sum(data[, "obs"]=="Good"))
  c(WeightedAccuracy=accu_w,Accuracy= accu,Precision= precis,Recall= recall,Specificity = specific)
}


#### Training control ####
fitControl<- trainControl(
  method = "repeatedcv",
  number=10,
  repeats=5,
  summaryFunction=customM)

fitControlUp <- trainControl(
  method = "cv",
  number = 10,
  repeats=5,
  sampling="up",
  summaryFunction=customM)

#### Train ####
# n rounds, without seed
#seedlist <- floor(runif(10)*1000)
n = 5
for (i in 1:n) {
seed <- seedlist[i]
#### Split training testing ####
set.seed(seed)
Train_indx <- createDataPartition(GermanCredit$Class,p=.8,list=F)
Data_train <- cbind(X[Train_indx,],Class=Y[Train_indx])
Data_blend <- cbind(X[-Train_indx,],Class=Y[-Train_indx])

#### SVM radial  ####
set.seed(seed)
svmradial<-  train(Class ~ ., data = Data_train, 
                   method = "svmRadial",
                   trControl=fitControlUp,
                   preProcess= preproc,
                   metric="WeightedAccuracy",
                   class.weights=c("Bad"=5/6,"Good"=1/6))

#svmradial
#svmradial$bestTune
#plot(svmradial)
#confusionMatrix(data = predict(svmradial,Data_blend), reference = Data_blend$Class)

#### Gradient boosting ####
set.seed(seed)
xgb<-  train(Class ~ ., data = Data_train, 
               method = "xgbTree",
               preProcess=preproc,
               trControl=fitControlUp,
               metric="WeightedAccuracy")
              
#xgb
#plot(xgb)
#confusionMatrix(data = predict(xgb,Data_blend), reference = Data_blend$Class)

#### Elastic net ####
set.seed(seed)
enetgrid <-  expand.grid(alpha= seq(0.1,1,0.05),
                        lambda=seq(0.01,0.1,0.01))

#weights <- ifelse(Data_train$Class=="Bad",5,1)
enet<-  train(Class ~ ., data = Data_train, 
              method = "glmnet",
              preProcess=preproc,
              trControl=fitControlUp,
              tuneGrid= enetgrid,
              metric="WeightedAccuracy")
              #weights=weights) #problem : len(weights)!= upsampled x


#plot(enet)
#enet$results
#enet$bestTune
#confusionMatrix(data = predict(enet,Data_blend), reference = Data_blend$Class)
#coef(enet$finalModel, enet$bestTune$lambda)

#### Compare models ####
# Performance
modelList <- list(RadialSVM=svmradial,
                 Xgb1 =xgb,
                 RF=rf)
resamps<- resamples(modelList)

#bwplot(resamps)
#summary(resamps)
#summary(diff(resamps))

# Correlation
#library(corrplot)
#modelCor(resamps)
#corrplot(modelCor(resamps),method="number")
#splom(resamps)

#### Ensemble models ###
basemodel <- lapply(modelList,predict,newdata=Data_blend)
Data_ensem<- do.call(cbind.data.frame, basemodel)
Data_ensem$Class <- Data_blend$Class

# blending using glm
set.seed(seed)
blend<-  train(Class ~ ., data = Data_ensem, 
                    method = "glm",
                    trControl=fitControl)

out <- capture.output(blend$results)
cat(paste("Round",i), out, file="result.txt", sep="n", fill=TRUE,append=TRUE)
print(paste("Round",i,"finished!"))
}