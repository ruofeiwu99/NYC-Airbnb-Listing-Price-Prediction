### load R packages
library(janitor)
library(readxl)
library(dplyr)
library(tidyr)
library(tidyverse)
library(yaImpute)
library(mgcv)
library(class)
library(gbm)
library(randomForest)
library(splitTools)
library(xgboost)
library(caret)
library(ALEPlot)
library(boot)
library(caret)
library(rpart)
library(nnet)

### Functions
CVInd <- function(n,K) {  #n is sample size; K is number of parts; returns K-length list of indices for each part
  m<-floor(n/K)  #approximate size of each part
  r<-n-m*K  
  I<-sample(n,n)  #random reordering of the indices
  Ind<-list()  #will be list of indices for all K parts
  length(Ind)<-K
  for (k in 1:K) {
    if (k <= r) kpart <- ((m+1)*(k-1)+1):((m+1)*k)  
    else kpart<-((m+1)*r+m*(k-r-1)+1):((m+1)*r+m*(k-r))
    Ind[[k]] <- I[kpart]  #indices for kth part of data
  }
  Ind
}

### Read In Data
abb = read.csv("airbnb_nyc_clean.csv") %>% clean_names()
abb = abb %>% select(c(6:13, 15:20))

abb.std = abb %>% 
  mutate(price=log(price+1), 
         availability_365=log(availability_365+1), 
         days_since_last_review=log(days_since_last_review+1), 
         reviews_per_month=log(reviews_per_month+1), 
         neighbourhood_group=as.factor(neighbourhood_group), 
         neighbourhood=as.factor(neighbourhood), 
         room_type=as.factor(room_type),
         last_review_year=replace_na(last_review_year, 0),
         last_review_month=replace_na(last_review_month, 0)) %>% 
  mutate_at(c(3,4, 6:11, 14), ~(scale(.) %>% as.vector))

map.room_type = c("Private room"=1, "Entire home/apt"=2, "Shared room"=0)

### GAM
### Get VIFs
library(car)
vif(lm(price~., data=abb.std %>% select(-neighbourhood)))

### Cross-validation
Nrep<-5 #number of replicates of CV
K<-10 #K-fold CV on each replicate
n.models = 3 #number of different models to fit
n=nrow(abb.std)
y<-abb.std$price
yhat=matrix(0,n,n.models) 
MSE<-matrix(0,Nrep,n.models)

for (j in 1:Nrep) {
  print(paste("------- Rep ", j, "-------:"))
  Ind<-CVInd(n,K)
  for (k in 1:K) {
    print(paste("Fold:", k))
    
    out = gam(price ~ neighbourhood_group 
              + last_review_year + last_review_month + room_type
              + s(latitude) + s(longitude) 
              + s(minimum_nights) + s(number_of_reviews)
              + s(reviews_per_month) + s(calculated_host_listings_count)
              + s(availability_365) + s(days_since_last_review),
              data=abb.std[-Ind[[k]],],
              family=gaussian(),
              sp=c(-1,-1,-1,
                   -1,-1,-1,
                   -1,-1,-1, 
                   -1,-1,-1))
    yhat[Ind[[k]],1]<-as.numeric(predict(out,abb.std[Ind[[k]],]))
    
    out = gam(price ~ neighbourhood_group
              + last_review_year + last_review_month + room_type
              + s(latitude) + s(longitude) 
              + s(minimum_nights) + s(number_of_reviews)
              + s(reviews_per_month) + s(calculated_host_listings_count)
              + s(availability_365),
              data=abb.std[-Ind[[k]],],
              family=gaussian(),
              sp=c(-1,-1,-1,
                   -1,-1,-1,
                   -1,-1,-1, 
                   -1,-1))
    yhat[Ind[[k]],2]<-as.numeric(predict(out,abb.std[Ind[[k]],]))
    
    
    out = gam(price ~ last_review_year + last_review_month 
              + room_type
              + s(latitude) + s(longitude) 
              + s(minimum_nights) + s(number_of_reviews)
              + s(reviews_per_month) + s(calculated_host_listings_count)
              + s(availability_365),
              data=abb.std[-Ind[[k]],],
              family=gaussian(),
              sp=c(-1,-1,-1,
                   -1,-1,-1,
                   -1,-1,-1, 
                   -1))
    yhat[Ind[[k]],3]<-as.numeric(predict(out,abb.std[Ind[[k]],]))
  } #end of k loop
  MSE[j,]=apply(yhat,2,function(x) sum((y-x)^2))/n
} #end of j loop
MSEAve<- apply(MSE,2,mean); cat("MSEAve:", MSEAve,"\n") #averaged mean square CV error
MSEsd <- apply(MSE,2,sd); cat("MSEsd:", MSEsd,"\n")   #SD of mean square CV error
r2<-1-MSEAve/var(y); cat("r2:", r2,"\n")  #CV r^2

### Best Model
out = gam(price ~ neighbourhood_group 
          + last_review_year + last_review_month 
          + room_type
          + s(latitude) + s(longitude) 
          + s(minimum_nights) + s(number_of_reviews)
          + s(reviews_per_month) + s(calculated_host_listings_count)
          + s(availability_365) + s(days_since_last_review),
          data=abb.std,
          family=gaussian(),
          sp=c(-1,-1,-1,
               -1,-1,-1,
               -1,-1,-1, 
               -1,-1,-1))
summary(out)
par(mfrow=c(2,4))
plot(out)
par(mfrow=c(1,1))

## PPR
### Cross-validation
Nrep<-5 #number of replicates of CV
K<-10  #K-fold CV on each replicate
n.models = 6 #number of different models to fit
abb.std.new = abb.std %>% 
  select(-neighbourhood)
n=nrow(abb.std.new)
y<-abb.std.new$price
yhat=matrix(0,n,n.models) 
MSE<-matrix(0,Nrep,n.models)
for (j in 1:Nrep) {
  print(paste("------- Rep ", j, "-------:"))
  Ind<-CVInd(n,K)
  for (k in 1:K) {
    
    cat("fold:", k, "\n")
    out<- ppr(price~., data=abb.std.new[-Ind[[k]],] , 
              nterms=5)  
    yhat[Ind[[k]],1] <-as.numeric(predict(out,abb.std.new[Ind[[k]],]))
    out<- ppr(price~., data=abb.std.new[-Ind[[k]],] , 
              nterms=7)  
    yhat[Ind[[k]],2] <-as.numeric(predict(out,abb.std.new[Ind[[k]],]))
    
    out<- ppr(price~., data=abb.std.new[-Ind[[k]],] , 
              nterms=10)  
    yhat[Ind[[k]],3] <-as.numeric(predict(out,abb.std.new[Ind[[k]],]))
    
    out<- ppr(price~., data=abb.std.new[-Ind[[k]],] , 
              nterms=11)  
    yhat[Ind[[k]],4] <-as.numeric(predict(out,abb.std.new[Ind[[k]],]))
    
    out<- ppr(price~., data=abb.std.new[-Ind[[k]],] , 
              nterms=12)  
    yhat[Ind[[k]],5] <-as.numeric(predict(out,abb.std.new[Ind[[k]],]))
    
    out<- ppr(price~., data=abb.std.new[-Ind[[k]],] , 
              nterms=13)  
    yhat[Ind[[k]],6] <-as.numeric(predict(out,abb.std.new[Ind[[k]],]))
    
  } #end of k loop
  MSE[j,]=apply(yhat,2,function(x) sum((y-x)^2))/n
} #end of j loop
MSEAve<- apply(MSE,2,mean); cat("MSEAve:", MSEAve,"\n") #averaged mean square CV error
MSEsd <- apply(MSE,2,sd); cat("MSEsd:", MSEsd,"\n")   #SD of mean square CV error
r2<-1-MSEAve/var(y); cat("r2:", r2,"\n")  #CV r^2 


Nrep<-1 #number of replicates of CV
K<-3  #K-fold CV on each replicate
n.models = 6 #number of different models to fit
abb.std.new = abb.std %>% 
  select(-neighbourhood)
n=nrow(abb.std.new)
y<-abb.std.new$price
yhat=matrix(0,n,n.models) 
MSE<-matrix(0,Nrep,n.models)
for (j in 1:Nrep) {
  print(paste("------- Rep ", j, "-------:"))
  Ind<-CVInd(n,K)
  for (k in 1:K) {
    
    cat("fold:", k, "\n")
    out<- ppr(price~., data=abb.std.new[-Ind[[k]],] , 
              nterms=8)  
    yhat[Ind[[k]],1] <-as.numeric(predict(out,abb.std.new[Ind[[k]],]))
    out<- ppr(price~., data=abb.std.new[-Ind[[k]],] , 
              nterms=9)  
    yhat[Ind[[k]],2] <-as.numeric(predict(out,abb.std.new[Ind[[k]],]))
    
    out<- ppr(price~., data=abb.std.new[-Ind[[k]],] , 
              nterms=10)  
    yhat[Ind[[k]],3] <-as.numeric(predict(out,abb.std.new[Ind[[k]],]))
    
    
  } #end of k loop
  MSE[j,]=apply(yhat,2,function(x) sum((y-x)^2))/n
} #end of j loop
MSEAve<- apply(MSE,2,mean); cat("MSEAve:", MSEAve,"\n") #averaged mean square CV error
MSEsd <- apply(MSE,2,sd); cat("MSEsd:", MSEsd,"\n")   #SD of mean square CV error
r2<-1-MSEAve/var(y); cat("r2:", r2,"\n")  #CV r^2 

### Random Forest
abb.std.new = abb.std %>% 
  select(-neighbourhood) %>% 
  mutate(last_review_year = as.numeric(last_review_year),
         last_review_month = as.numeric(last_review_month))
abb.std.new

rForest1 <- randomForest(price~., data=abb.std.new , 
                         mtry=3, 
                         ntree = 500, 
                         maxnodes = 4500, 
                         importance = TRUE,
                         do.trace=TRUE)
rForest1
plot(rForest1)
rForest2 <- randomForest(price~., data=abb.std.new , 
                         mtry=3, 
                         ntree = 200, 
                         maxnodes = 100,
                         importance = TRUE)
rForest2
rForest3 <- randomForest(price~., data=abb.std.new , 
                         mtry=3, 
                         ntree = 200, 
                         maxnodes = 100,
                         importance = TRUE)
rForest3
# Best 1 - 60
rForest4 <- randomForest(price~., data=abb.std.new , 
                         mtry=3, 
                         ntree = 200, 
                         nodesize = 100,
                         importance = TRUE)
rForest4
# Best 2 - 60.5
rForest5 <- randomForest(price~., data=abb.std.new , 
                         mtry=5, 
                         ntree = 200, 
                         nodesize = 100,
                         importance = TRUE)
rForest5
# Best 3 - 60.53
rForest6 <- randomForest(price~., data=abb.std.new , 
                         mtry=5, 
                         ntree = 250, 
                         nodesize = 100,
                         importance = TRUE)
rForest6
# Best 4 - 60.59
rForest7<- randomForest(price~., data=abb.std.new , 
                        mtry=8, 
                        ntree = 250, 
                        nodesize = 100,
                        importance = TRUE,
                        do.trace=TRUE)
rForest7
# Best 5 - 60.6
rForest8 <- randomForest(price~., data=abb.std.new , 
                         mtry=7, 
                         ntree = 250, 
                         nodesize = 100,
                         importance = TRUE)
rForest8
# Best 6 - 61.13
rForest9 <- randomForest(price~., data=abb.std.new , 
                         mtry=7, 
                         ntree = 250, 
                         nodesize = 50,
                         importance = TRUE)
rForest9
# Best 7 - 61.36
rForest10 <- randomForest(price~., data=abb.std.new , 
                          mtry=7, 
                          ntree = 250, 
                          nodesize = 25,
                          importance = TRUE)
rForest10
# Best 8 - 61.47
rForest11 <- randomForest(price~., data=abb.std.new , 
                          mtry=7, 
                          ntree = 500, 
                          nodesize = 25,
                          importance = TRUE,
                          do.trace=TRUE)
rForest11

plot(rForest11)

# Best 8 - 61.47
rForest11 <- randomForest(price~., data=abb.std.new , 
                          mtry=7, 
                          ntree = 20, 
                          nodesize = 25,
                          importance = TRUE,
                          do.trace=TRUE)
abb.std.new
partialPlot(rForest11, pred.data=abb.std.new, 
            x.var = latitude, 
            xlab = "latitude", main=NULL)
par(mfrow=c(2,4))
for (i in c(8,1,4,2,7,5,6,3)) {
  partialPlot(rForest11, pred.data=abb.std.new, 
              x.var = names(abb.std.new)[i], 
              xlab = names(abb.std.new)[i], main=NULL) #creates "partial dependence" plots 
}
par(mfrow=c(1,1))
c(rForest11$mse[rForest1$ntree], 
  sum((rForest11$predicted - abb.std.new$price)^2)/nrow(abb.std.new)) #both give the OOB MSE


### Boosted Tree
boosted_tree_param_tuning <- function(dt, n_trees, lambda, tree_depth) {
  set.seed(123)
  Nrep = 5
  r2 = rep(0, Nrep)
  y = dt$price
  n = nrow(dt)
  for (i in 1:Nrep) {
    gbm.fit <- gbm(price~., data = dt[,-1], var.monotone = rep(0,12), distribution = "gaussian", n.trees = n_trees, 
                   shrinkage = lambda, interaction.depth = tree_depth, bag.fraction = .5, train.fraction = 1, 
                   n.minobsinnode = 10, cv.folds = 10, keep.data = TRUE, verbose = FALSE)
    MSE = min(gbm.fit$cv.error)*n/length(y)
    r2[i] = 1 - MSE/var(y)
  }
  return(mean(r2))
}

hyper_grid <- expand.grid(
  shrinkage = c(.01, .05),
  interaction.depth = c(4, 5),
  n.trees = c(200, 500, 800), 
  #MSE = 0,
  CV_r2 = 0
)

n.models = nrow(hyper_grid)
for (i in 1:n.models) {
  hyper_grid$CV_r2[i] = boosted_tree_param_tuning(abb.std, hyper_grid$n.trees[i], hyper_grid$shrinkage[i], hyper_grid$interaction.depth[i])
}

tuning_df = hyper_grid %>% arrange(desc(CV_r2))
tuning_df


### xgboost
abb.std.oh = data.frame(predict(dummyVars(" ~ .", data = abb.std), 
                                newdata = abb.std))

data = xgb.DMatrix(data=as.matrix(abb.std.oh[,-c(9)]), label=abb.std.oh[,9])

# winner
xgb.4 = xgb.cv(data, 
               params=list(max.depth=6, eta=0.06), 
               nrounds=800, 
               nfold=10,
               verbose=F,
               early_stopping_rounds=15)
MSE[3] =  xgb.4$evaluation_log$test_rmse_mean[xgb.4$best_iteration]^2
1-mean(MSE)

xgb.5 = xgboost(data = data, 
                max.depth = 6,
                eta = 0.06,
                nrounds = 491,
                verbose=F)

importance_matrix = xgb.importance(colnames(data), model = xgb.5)
xgb.plot.importance(importance_matrix, rel_to_first = TRUE, xlab = "Relative importance")

# ALE plot
yhat = function(X.model, newdata) as.numeric(predict(X.model, xgb.DMatrix(as.matrix(newdata[,-c(9)]), label=as.matrix(newdata[,9]))))

par(mfrow=c(3,4))
for (j in c(c(6:8), c(10:17)))  {
  ALEPlot(abb.std.oh, xgb.5, pred.fun=yhat, J=j, K=50, NA.plot = TRUE)
}
par(mfrow=c(1,1))

a=ALEPlot(abb.std.oh, xgb.5, pred.fun=yhat, J=c(7,6), K=50, NA.plot = TRUE)

### K-NN
n = nrow(abb.std)
n.rep = 5
n.models = c(1,5,8,10,15,18,20,25,30)
n.folds = 10
y.hat = matrix(0, n, length(n.models))
MSE = matrix(0, n.rep, length(n.models))
for (i in 1:n.rep) {
  Ind = create_folds(1:n, n.folds)
  for (j in 1:n.folds) {
    X.train = as.matrix(abb.std[Ind[[j]],-c(1,5)])
    X.test = as.matrix(abb.std[-Ind[[j]],-c(1,5)])
    y.train = abb.std[Ind[[j]],5]
    
    for (k in 1:length(n.models)) {
      knn.k = ann(X.train, X.test, n.models[k], verbose=F)
      ind = as.matrix(knn.k$knnIndexDist[,1:n.models[k]])
      y.hat[-Ind[[j]],k] = apply(ind, 1, function(x) mean(as.matrix(y.train[x])))
    }
  }
  MSE[i,] = apply(y.hat, 2, function(x) sum((x-abb.std[5])^2)/n) 
}
MSEave = apply(MSE, 2, mean)
r2 = 1-MSEave/c(var(abb.std[5]))
r2

### loess
options(warn=-1)
n = nrow(abb.std)
n.rep = 3
n.folds = 10
n.models = 9
y.hat = matrix(0, n, n.models)
MSE = matrix(0, n.rep, n.models)
spans = c(0.2, 0.4, 0.6)
for (i in 1:n.rep) {
  Ind = create_folds(1:n, n.folds)
  for (j in 1:n.folds) {
    train = abb.std[Ind[[j]],c(2:5,8)]
    X.test = abb.std[-Ind[[j]],c(2,3,4,8)]
    
    for (k in 1:n.models) {
      kw.i = loess(price~., train, degree=(k-1)%/%3, span=spans[(k-1)%%3+1], 
                   control=loess.control(surface = "direct"))
      y.hat[-Ind[[j]],k] = predict(kw.i, X.test)
    }
  }
  MSE[i,] = apply(y.hat, 2, function(x) sum((abb.std$price - x)^2)/n)
}
MSEave = apply(MSE, 2, mean)
r2 = 1 - MSEave/var(abb.std$price)
r2
options(warn=0)


### processed data
abb = read.csv("./airbnb_nyc_clean.csv") %>% clean_names()
abb = abb %>% 
  select(c(6:9, 10:13, 15:20))
map.room_type = c("Private room"=1, "Entire home/apt"=2, "Shared room"=0)
abb.std = abb %>% 
  mutate(price=log(price+1), 
         availability_365=log(availability_365+1), 
         days_since_last_review=log(days_since_last_review+1), 
         reviews_per_month=log(reviews_per_month+1), 
         neighbourhood_group=as.factor(neighbourhood_group), 
         neighbourhood=as.factor(neighbourhood), 
         room_type=map.room_type[abb$room_type], 
         last_review_year=replace_na(last_review_year, 0),
         last_review_month=replace_na(last_review_month, 0)) %>% 
  mutate_at(c(3:11,14), ~(scale(.) %>% as.vector)) %>% 
  select(-c(2))
### Linear Reg
fit <- lm(price ~., data = abb.std)
summary(fit)
# VIF
library(car)
car::vif(fit)
# VarImp
Imp <- varImp(fit, scale = FALSE)
sort(Imp$Overall, decreasing = TRUE)

# Linear Reg CV
# 10-fold CV
##Now use multiple reps of CV to compare Neural Nets###
Nrep<-5 #number of replicates of CV
K<-10  #K-fold CV on each replicate
n.models = 1 #number of different models to fit
n=nrow(abb.std)
y<-abb.std$price
yhat=matrix(0,n,n.models)
MSE<-matrix(0,Nrep,n.models)
# fix lambda = .1 while adjusting M
for (j in 1:Nrep) {
  Ind<-CVInd(n,K)
  for (k in 1:K) {
    out<-fit <- lm(price ~., data = abb.std[-Ind[[k]],])
    yhat[Ind[[k]],1]<-as.numeric(predict(out,abb.std[Ind[[k]],]))
    
  } #end of k loop
  MSE[j,]=apply(yhat,2,function(x) sum((y-x)^2))/n
} #end of j loop
MSE
MSEAve<- apply(MSE,2,mean); MSEAve #averaged mean square CV error
MSEsd <- apply(MSE,2,sd); MSEsd   #SD of mean square CV error
r2<-1-MSEAve/var(y); r2  #CV r^2

### Neural Network
# model specifications
grid = list(size = c(25, 30, 35),
            decay = c(0.15, 0.2, 25)) %>%
  cross_df()
Nrep = 5 #number of replicates of CV
K = 10  #K-fold CV on each replicate
n.models = nrow(grid) #number of different models to fit
n = nrow(abb.std)
y = abb.std$price
yhat = matrix(0, n, n.models)
MSE<-matrix(0,Nrep,n.models)
for (j in 1:Nrep) {
  cat("\n************ replicate:", j, "**********\n")
  Ind<-CVInd(n,K)
  for (k in 1:K) {
    cat("Fold:", k, "\n")
    # fit all models
    for (m in 1:n.models) {
      cat("  > model", m, "\n")
      out<-nnet(price~.,abb.std[-Ind[[k]],], linout=T, skip=F, 
                size=grid[m,]$size, decay=grid[m,]$decay, 
                maxit=500, trace=F)
      yhat[Ind[[k]],m]<-as.numeric(predict(out,abb.std[Ind[[k]],]))
    }
  } #end of k loop
  MSE[j,]=apply(yhat,2,function(x) sum((y-x)^2))/n
} #end of j loop
MSEAve<- apply(MSE,2,mean); MSEAve #averaged mean square CV error
MSEsd <- apply(MSE,2,sd); MSEsd   #SD of mean square CV error
r2<-1-MSEAve/var(y); r2  #CV r^2
grid = grid %>%
  mutate(r2 = r2, MSEAve = MSEAve, MSEsd=MSEsd) %>%
  arrange(MSEAve)
grid
# best model
nn1<-nnet(price~.,abb.std, linout=T, skip=F, size=30, decay=0.2, maxit=1000, trace=F) 
##From CV, these are about the best tuning parameters
## Use ALEPlot package to create accumulated local effects (ALE) plots
library(ALEPlot)
yhat <- function(X.model, newdata) as.numeric(predict(X.model, newdata))
par(mfrow=c(2,5))
for (j in 1:10)  {ALEPlot(abb.std[,c(1:4, 6:11)], nn1, pred.fun=yhat, J=j, K=50, NA.plot = TRUE)
  rug(abb.std[,j]) }  ## This creates main effect ALE plots for all 8 predictors
par(mfrow=c(1,1))
par(mfrow=c(2,2))  ## This creates 2nd-order interaction ALE plots for x1, x2, x8
a=ALEPlot(abb.std[,c(1:4, 6:11)], nn1, pred.fun=yhat, J=c(9,10), K=50, NA.plot = TRUE)
par(mfrow=c(1,1))

