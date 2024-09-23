#install.packages('randomForest')
#install.packages('gbm')
#install.packages('rpart')
library(gbm)
library(randomForest)
library(rpart)
library(MASS)
library(class)

#load data
data<-read.csv('Auto.csv')
head(data)
dim(data)
data

pie(table(data$origin),labels=c("American","European","Japanese"))
table(data$origin)
#EDA
cor_vals<-c(cor(data$mpg,data$origin),
            cor(data$cylinders,data$origin),
            cor(data$displacement,data$origin),
            cor(data$horsepower,data$origin),
            cor(data$weight,data$origin),
            cor(data$acceleration,data$origin),
            cor(data$year,data$origin)
)

colnames<-(colnames(data))
covariance_response<-cbind(colnames[1:7],cor_vals)
covariance_response


corr<-cor(data[,-8])
corr
auto_vec <- as.vector(corr)
auto_vec[!lower.tri(corr)] <- NA
indices <- order(abs(auto_vec), decreasing=TRUE,na.last = NA)
sorted_cor <- auto_vec[indices]
row_indices <- row(corr)[indices]
col_indices <- col(corr)[indices]
cor_pairs <- data.frame(
  Var1 = rownames(corr)[row_indices[0:20]],
  Var2 = colnames(corr)[col_indices[0:20]],
  Correlation = sorted_cor[0:20]
)
cor_pairs

displacement+acceleration+year
#set test data set to 30% due to small size of data set
set.seed(7406)
flag <- sort(sample(392,118, replace = FALSE))
autotrain <- data[-flag,]
autotest <- data[flag,]

y1<-autotrain$origin
y2<-autotest$origin

#random forest
rforestout<-NULL
for (i in 101:550){
rfmodel<-randomForest(as.factor(origin) ~displacement+acceleration+year, data=autotrain, type='class',importance=TRUE,ntree=i)
testpred<-predict(rfmodel, autotest, type='class')
rf_class_error<-mean(testpred!=y2)
rforestout<-c(rforestout,rf_class_error)
}
which.min(rforestout)
plot(rforestout,ylab="Categorization Error",xlab="Number Trees")
#70
rfmodel2<-randomForest(as.factor(origin) ~., data=autotrain, type='class',importance=TRUE,ntree=178)
importance(rfmodel2)
varImpPlot(rfmodel2)

#randomForest(as.factor(origin)~displacement+weight+mpg, data=autotrain, type='class',importance=TRUE,ntree=178)

#boost model
gbmmodel <- gbm(as.factor(origin) ~displacement+acceleration+year,data=autotrain, distribution = 'multinomial',cv=5)
gbmperf<-gbm.perf(gbmmodel, method="cv") 
summary(gbmperf)

gbmpredict<-predict(gbmmodel,newdata = autotest, n.trees=gbmperf, type="response",cv=5)
gbmpredict
out<-NULL
for (i in 1:nrow(gbmpredict)){
  row<-which.max(gbmpredict[i,,])
  out<-c(out,row)
}
out

b_class_error<-mean(out!=y2)
b_class_error

#interaction depth (2)
interaction_depth_test<-NULL
for (i in 1:10){
gbmmodel2<-gbm(as.factor(origin) ~displacement+acceleration+year,data=autotrain, distribution = 'multinomial',cv=5,interaction.depth=i)
gbmpredicttemp<-predict(gbmmodel2,newdata = autotest, n.trees=gbmperf, type="response",cv=5)
out<-NULL
for (i in 1:nrow(gbmpredicttemp)){
  row<-which.max(gbmpredicttemp[i,,])
  out<-c(out,row)
}
b_class_error_temp<-mean(out!=y2)
interaction_depth_test<-c(interaction_depth_test,b_class_error_temp)
}
interaction_depth_test

#shrinkage
shrinkage_test<-NULL
for (i in seq(.01,.1,.01)){
gbmmodel2<-gbm(as.factor(origin) ~displacement+acceleration+year,data=autotrain, distribution = 'multinomial',cv=5,interaction.depth=2,shrinkage=i)
gbmpredicttemp<-predict(gbmmodel2,newdata = autotest, n.trees=gbmperf, type="response",cv=5)
out<-NULL
for (i in 1:nrow(gbmpredicttemp)){
  row<-which.max(gbmpredicttemp[i,,])
  out<-c(out,row)
}
b_class_error_temp<-mean(out!=y2)
shrinkage_test<-c(shrinkage_test,b_class_error_temp)
}
shrinkage_test

#gbm(as.factor(origin) ~displacement+acceleration+year,data=autotrain, distribution = 'multinomial',cv=5,interaction.depth=2,shrinkage=.04)

#tree
rpartmodel <- rpart(as.factor(origin) ~displacement+acceleration+year,data=autotrain, method="class", parms=list(split="information"))
plot(rpartmodel,compress=TRUE)
text(rpartmodel)

rpartpredict<-predict(rpartmodel, autotest,type="class")
tree_error<-mean(rpartpredict!=y2)
tree_error

plotcp(rpartmodel)
print(rpartmodel$cptable)

optcp <- which.min(rpartmodel$cptable[, "xerror"]);
cp1 <- rpartmodel$cptable[optcp, "CP"]
cp1
rpartprune <- prune(rpartmodel,cp=cp1)

rpartpredict2<-predict(rpartprune, autotest,type="class")
tree_error2<-mean(rpartpredict2!=y2)
tree_error2


#LDA
ldamodel <- lda(as.factor(origin)~displacement+acceleration+year,data=autotrain)
ldapredict <- predict(ldamodel, autotest[,1:7])$class
lda_class_error<-mean(ldapredict!=y2)
lda_class_error

#knn
knnout<-NULL
for (i in 1:99){
knnpred<-knn(autotrain[,c('displacement','acceleration','year')],autotest[,c('displacement','acceleration','year')],autotrain[,8],k=i)
knn_class_error<-mean(knnpred!=y2)
knnout<-c(knnout,knn_class_error)
}
which.min(knnout)
plot(knnout)

#knn(autotrain[,c('displacement','acceleration','year')],autotest[,c('displacement','acceleration','year')],autotrain[,8],k=38)

#CV model test
finalout<-NULL
for (i in 1:500){
  flag2 <- sort(sample(392,118, replace = FALSE))
  autotrain2 <- data[-flag2,]
  autotest2 <- data[flag2,]
  y12<-autotrain2$origin
  y22<-autotest2$origin  
  
  #random forrest
  rffinal<-randomForest(as.factor(origin)~displacement+weight+mpg, data=autotrain2, type='class',importance=TRUE,ntree=178)
  testpred2<-predict(rffinal, autotest2, type='class')
  te0<-mean(testpred2!=y22)
  
  #gbm
  gbmfinal<-gbm(as.factor(origin) ~displacement+acceleration+year,data=autotrain2, distribution = 'multinomial',cv=5,interaction.depth=2,shrinkage=.04)
  gbmpredict2<-predict(gbmfinal,newdata = autotest2, n.trees=90, type="response",cv=5)
  gbmout<-NULL
  for (j in 1:nrow(gbmpredict2)){
    row<-which.max(gbmpredict2[j,,])
    gbmout<-c(gbmout,row)
  }
  te1<-mean(gbmout!=y22)
  
  #tree
  rpartmodel2 <- rpart(as.factor(origin) ~displacement+acceleration+year,data=autotrain2, method="class", parms=list(split="information"))
  optcp2 <- which.min(rpartmodel2$cptable[, "xerror"]);
  cp2 <- rpartmodel2$cptable[optcp2, "CP"]
  rpartprune2 <- prune(rpartmodel2,cp=cp2)
  rpartpredict_final<-predict(rpartprune2, autotest2,type="class")
  te2<-mean(rpartpredict_final!=y22)
  
  #Lda
  ldamodel2 <- lda(as.factor(origin)~displacement+acceleration+year,data=autotrain2)
  ldapredict2 <- predict(ldamodel2, autotest2[,c("displacement", "acceleration", "year")])$class
  te3<-mean(ldapredict2!=y22)
  
  #knn
  knnpred2<-knn(autotrain2[,c('displacement','acceleration','year')],autotest2[,c('displacement','acceleration','year')],autotrain2[,8],k=38)
  te4<-mean(knnpred2!=y22)
  
  finalout<-rbind(finalout,c(te0,te1,te2,te3,te4))
}
finalout

apply(finalout,2, mean)
apply(finalout,2, var)

