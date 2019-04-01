require(Matrix)
library(e1071)
library(SparseM)
library(caret)

### Calculate the distance from the hyperplane

hyper_distance <-function(X_d,model){
  W <- t(model$coefs) %*% model$SV
  b <- model$rho
  val <- apply(X_d,1,function(X) W%*%X+b)
  return(val)
}
# Change to the working directory where the data files are located.
# TODO: You should change the following ... to your working directory
setwd("C:/Users/gselva3/Documents/ass2_575/")

############################### Reading the artciles ##############################
dataFile <- file("articles.test", "r")
dataLines <- readLines(dataFile)
m <- length(dataLines)
close(dataFile)
dataTokens  <- strsplit(dataLines, "[: ]")

# Extract every first token from each line as a vector of numbers, which is the class label.
Y_test <-  sapply(dataTokens, function(example) {as.numeric(example[1])})

# Extract the rest of tokens from each line as a list of matrices (one matrix for each line)
# where each row consists of two columns: (feature number, its occurrences)
X_list <-  lapply(dataTokens, function(example) {n = length(example) - 1; matrix(as.numeric(example[2:(n+1)]), ncol=2, byrow=T)})

# Add one column that indicates the example number at the left
X_list <-  mapply(cbind, x=1:length(X_list), y=X_list)

# Merge a list of different examples vertcially into a matrix
X_test_data <-  do.call('rbind', X_list)
tmp2 <- as.data.frame(X_test_data)

# Get a sparse data matrix X (rows: training exmaples, columns: # of occurrences for each of features)
X_test <- sparseMatrix(x=X_test_data[,3], i=X_test_data[,1], j=X_test_data[,2])
rm(dataFile,dataLines,dataTokens,X_list)
################################### training the model ####################################
# Read all individual lines in a text file.
# m = the number of training examples
dataFile <- file("articles.train", "r")
dataLines <- readLines(dataFile)
m <- length(dataLines)
close(dataFile)

# Split every string element by tokenizing space and colon.
dataTokens  <- strsplit(dataLines, "[: ]")

# Extract every first token from each line as a vector of numbers, which is the class label.
Y_train <-  sapply(dataTokens, function(example) {as.numeric(example[1])})

# Extract the rest of tokens from each line as a list of matrices (one matrix for each line)
# where each row consists of two columns: (feature number, its occurrences)
X_list <-  lapply(dataTokens, function(example) {n = length(example) - 1; matrix(as.numeric(example[2:(n+1)]), ncol=2, byrow=T)})

# Add one column that indicates the example number at the left
X_list <-  mapply(cbind, x=1:length(X_list), y=X_list)

# Merge a list of different examples vertcially into a matrix
X_train_data <-  do.call('rbind', X_list)
tmp1 <- as.data.frame(X_train_data)
rest <- unique(tmp2$V2[-which(tmp2$V2 %in% tmp1$V2)])
row_num <- rep(4001,length(rest))
count <- rep(0,length(rest))
map <- cbind(row_num,rest,count)
X_train_data <- rbind(X_train_data,map)

Y_train <- c(Y_train,0)
# Get a sparse data matrix X (rows: training exmaples, columns: # of occurrences for each of features)
X_train <- sparseMatrix(x=X_train_data[,3], i=X_train_data[,1], j=X_train_data[,2])
######################################### SVM ##########################################
# 1 vs all models 

######### Class1 ############
Y_C1 <- ifelse(Y_train==1,1,-1)
Y_C1 <- factor(Y_C1, levels = c(1,-1))
model1 <- svm(X_train,as.factor(Y_C1),kernel="linear")

pred <- predict(model1,X_train,decision.values = TRUE)
confusionMatrix(data = pred,reference = as.factor(Y_C1),mode="prec_recall")

Y_C1 <- ifelse(Y_test==1,1,-1)
Y_C1 <- factor(Y_C1, levels = c(1,-1))
pred <- predict(model1,X_test,decision.values = TRUE)
confusionMatrix(data = pred,reference = as.factor(Y_C1),mode="prec_recall")

######## Class 2 #############
Y_C2 <- ifelse(Y_train==2,1,-1)
Y_C2 <- factor(Y_C2, levels = c(1,-1))
model2 <- svm(X_train,as.factor(Y_C2),kernel="linear")

pred <- predict(model2,X_train)
confusionMatrix(data = pred,reference = as.factor(Y_C2),mode="prec_recall")

Y_C2 <- ifelse(Y_test==2,1,-1)
Y_C2 <- factor(Y_C2, levels = c(1,-1))
pred <- predict(model2,X_test)
confusionMatrix(data = pred,reference = as.factor(Y_C2),mode="prec_recall")

######## Class 3 #############
Y_C3 <- ifelse(Y_train==3,1,-1)
Y_C3 <- factor(Y_C3, levels = c(1,-1))
model3 <- svm(X_train,as.factor(Y_C3),kernel="linear")

pred <- predict(model3,X_train)
confusionMatrix(data = pred,reference = as.factor(Y_C3),mode="prec_recall")

Y_C3 <- ifelse(Y_test==3,1,-1)
Y_C3 <- factor(Y_C3, levels = c(1,-1))
pred <- predict(model3,X_test)
confusionMatrix(data = pred,reference = as.factor(Y_C3),mode="prec_recall")


######## Class 4 #############
Y_C4 <- ifelse(Y_train==4,1,-1)
Y_C4 <- factor(Y_C4, levels = c(1,-1))
model4 <- svm(X_train,as.factor(Y_C4),kernel="linear")

pred <- predict(model4,X_train)
confusionMatrix(data = pred,reference = as.factor(Y_C4),mode="prec_recall")

Y_C4 <- ifelse(Y_test==4,1,-1)
Y_C4 <- factor(Y_C4, levels = c(1,-1))
pred <- predict(model4,X_test)
confusionMatrix(data = pred,reference = as.factor(Y_C4),mode="prec_recall")

########################## Predicting the test value by the formula ##########################
test_pred <- as.data.frame(hyper_distance(X_test,model1))
test_pred <- cbind(test_pred,hyper_distance(X_test,model2))
test_pred <- cbind(test_pred,hyper_distance(X_test,model3))
test_pred <- cbind(test_pred,hyper_distance(X_test,model4))

colnames(test_pred) <- c("1","2","3","4")
test_pred$Prediction <- colnames(test_pred)[apply(test_pred,1,which.max)]
print(paste("Accuracy=",mean(Y_test==test_pred$Prediction)*100))

######################## Soft margin classifier with linear kernel ###############################
C =c(0.125,0.25,0.5,1,2,4,8,16,32,64,128,256,512)


## 75% of the sample size
smp_size <- floor(0.75 * nrow(X_train))

## set the seed to make your partition reproducible
set.seed(123)
train_ind <- sample(seq_len(nrow(X_train)), size = smp_size)

X_train_train <- X_train[train_ind,]
X_train_Cv <- X_train[-train_ind,]

optimal_C <- 1
old_error <- 1/0
train_error <- c() 
validation_error <- c()
Y_train_train <- Y_train[train_ind]
Y_train_Cv <- Y_train[-train_ind]

for( i in 1:length(C)){
  Y_C1 <- ifelse(Y_train_train==1,1,-1)
  Y_C1 <- factor(Y_C1, levels = c(1,-1))
  model1 <- svm(X_train_train,as.factor(Y_C1),cost = C[i],kernel="linear")
  
  Y_C2 <- ifelse(Y_train_train==2,1,-1)
  Y_C2 <- factor(Y_C2, levels = c(1,-1))
  model2 <- svm(X_train_train ,as.factor(Y_C2),cost=C[i],kernel="linear")
  
  Y_C3 <- ifelse(Y_train_train==3,1,-1)
  Y_C3 <- factor(Y_C3, levels = c(1,-1))
  model3 <- svm(X_train_train,as.factor(Y_C3),cost=C[i],kernel="linear")
  
  Y_C4 <- ifelse(Y_train_train==4,1,-1)
  Y_C4 <- factor(Y_C4, levels = c(1,-1))
  model4 <- svm(X_train_train,as.factor(Y_C4),cost=C[i],kernel="linear")
  
  #train_error
  train_pred <- as.data.frame(hyper_distance(X_train_train,model1))
  train_pred <- cbind(train_pred,hyper_distance(X_train_train,model2))
  train_pred <- cbind(train_pred,hyper_distance(X_train_train,model3))
  train_pred <- cbind(train_pred,hyper_distance(X_train_train,model4))
  
  colnames(train_pred) <- c("1","2","3","4")
  train_pred$Prediction <- colnames(train_pred)[apply(train_pred,1,which.max)]
  train_error <- c(train_error,100-mean(Y_train_train==train_pred$Prediction)*100)
  
  #Cross Validation for model evaluvation
  CV_pred <- as.data.frame(hyper_distance(X_train_Cv,model1))
  CV_pred <- cbind(CV_pred,hyper_distance(X_train_Cv,model2))
  CV_pred <- cbind(CV_pred,hyper_distance(X_train_Cv,model3))
  CV_pred <- cbind(CV_pred,hyper_distance(X_train_Cv,model4))
  
  colnames(CV_pred) <- c("1","2","3","4")
  CV_pred$Prediction <- colnames(CV_pred)[apply(CV_pred,1,which.max)]
  validation_error <- c(validation_error,100-mean(Y_train_Cv==CV_pred$Prediction)*100)
  new_error <- 100-mean(Y_train_Cv==CV_pred$Prediction)*100
  if(new_error<old_error){
    print(C[i])
    optimal_C <- C[i]
    old_error <- new_error
  }
}


C_1 <- log(C,base=2)

plot(C_1,validation_error,type="o",xlab="Log(Cost)",col="blue",ylab="Error")
lines(C_1,train_error,type="o",col="red")
legend(7,81,legend=c("train","validation"),col=c("red","blue"),lty=1)

######### Soft Margin classifier with Cost obtained from previous step ######
######### Class1 ############
Y_C1 <- ifelse(Y_train==1,1,-1)
Y_C1 <- factor(Y_C1, levels = c(1,-1))
model1 <- svm(X_train,as.factor(Y_C1),cost=0.125,kernel="linear")

######## Class 2 #############
Y_C2 <- ifelse(Y_train==2,1,-1)
Y_C2 <- factor(Y_C2, levels = c(1,-1))
model2 <- svm(X_train,as.factor(Y_C2),cost=0.125,kernel="linear")

######## Class 3 #############
Y_C3 <- ifelse(Y_train==3,1,-1)
Y_C3 <- factor(Y_C3, levels = c(1,-1))
model3 <- svm(X_train,as.factor(Y_C3),cost=0.125,kernel="linear")

######## Class 4 #############
Y_C4 <- ifelse(Y_train==4,1,-1)
Y_C4 <- factor(Y_C4, levels = c(1,-1))
model4 <- svm(X_train,as.factor(Y_C4),cost=0.125,kernel="linear")

########################## Predicting the test value by the formula ##########################
test_pred <- as.data.frame(hyper_distance(X_test,model1))
test_pred <- cbind(test_pred,hyper_distance(X_test,model2))
test_pred <- cbind(test_pred,hyper_distance(X_test,model3))
test_pred <- cbind(test_pred,hyper_distance(X_test,model4))

colnames(test_pred) <- c("1","2","3","4")
test_pred$Prediction <- colnames(test_pred)[apply(test_pred,1,which.max)]
print(paste("Accuracy=",mean(Y_test==test_pred$Prediction)*100))

########## Normalzing the data for better performance ###########
norm_train <- X_train %*% Matrix::Diagonal(x = 1 / sqrt(Matrix::colSums(X_train^2)))
norm_test <- X_test %*% Matrix::Diagonal(x = 1 / sqrt(Matrix::colSums(X_test^2)))
norm_train[is.na(norm_train)] <- 0
norm_test[is.na(norm_test)] <- 0

Y_C1 <- ifelse(Y_train==1,1,-1)
Y_C1 <- factor(Y_C1, levels = c(1,-1))
model1 <- svm(norm_train,as.factor(Y_C1),cost=0.125,kernel="linear")

######## Class 2 #############
Y_C2 <- ifelse(Y_train==2,1,-1)
Y_C2 <- factor(Y_C2, levels = c(1,-1))
model2 <- svm(norm_train,as.factor(Y_C2),cost=0.125,kernel="linear")

######## Class 3 #############
Y_C3 <- ifelse(Y_train==3,1,-1)
Y_C3 <- factor(Y_C3, levels = c(1,-1))
model3 <- svm(norm_train,as.factor(Y_C3),cost=0.125,kernel="linear")

######## Class 4 #############
Y_C4 <- ifelse(Y_train==4,1,-1)
Y_C4 <- factor(Y_C4, levels = c(1,-1))
model4 <- svm(norm_train,as.factor(Y_C4),cost=0.125,kernel="linear")

########################## Predicting the test value by the formula ##########################
test_pred <- as.data.frame(hyper_distance(norm_test,model1))
test_pred <- cbind(test_pred,hyper_distance(norm_test,model2))
test_pred <- cbind(test_pred,hyper_distance(norm_test,model3))
test_pred <- cbind(test_pred,hyper_distance(norm_test,model4))

colnames(test_pred) <- c("1","2","3","4")
test_pred$Prediction <- colnames(test_pred)[apply(test_pred,1,which.max)]
print(paste("Accuracy=",mean(Y_test==test_pred$Prediction)*100))

############### 1 vs 1 softmargin classifier #############
Y_C1 <- Y_train[c(1:2000)]
Y_C1 <- ifelse(Y_train==1,1,-1)
Y_C1 <- factor(Y_C1, levels = c(1,-1))
model12 <- svm(norm_train,as.factor(Y_C1),cost=0.125,kernel="linear")

Y_C1 <- Y_train[c(1:1000,2001:3000)]
Y_C1 <- ifelse(Y_train==1,1,-1)
Y_C1 <- factor(Y_C1, levels = c(1,-1))
model13 <- svm(norm_train,as.factor(Y_C1),cost=0.125,kernel="linear")

Y_C1 <- Y_train[c(1:1000,3001:4000)]
Y_C1 <- ifelse(Y_train==1,1,-1)
Y_C1 <- factor(Y_C1, levels = c(1,-1))
model14 <- svm(norm_train,as.factor(Y_C1),cost=0.125,kernel="linear")

Y_C1 <- Y_train[c(1001:3000)]
Y_C1 <- ifelse(Y_train==2,1,-1)
Y_C1 <- factor(Y_C1, levels = c(1,-1))
model23 <- svm(norm_train,as.factor(Y_C1),cost=0.125,kernel="linear")

Y_C1 <- Y_train[c(1001:2000,3001:4000)]
Y_C1 <- ifelse(Y_train==2,1,-1)
Y_C1 <- factor(Y_C1, levels = c(1,-1))
model24 <- svm(norm_train,as.factor(Y_C1),cost=0.125,kernel="linear")

Y_C1 <- Y_train[c(2001:4000)]
Y_C1 <- ifelse(Y_train==3,1,-1)
Y_C1 <- factor(Y_C1, levels = c(1,-1))
model34 <- svm(norm_train,as.factor(Y_C1),cost=0.125,kernel="linear")

###################### Prediciton ########## 
test_pred <- as.data.frame(ifelse(predict(model12,norm_test)==1,1,2))
test_pred <- cbind(test_pred,(ifelse(predict(model13,norm_test)==1,1,3)))
test_pred <- cbind(test_pred,(ifelse(predict(model14,norm_test)==1,1,4)))
test_pred <- cbind(test_pred,(ifelse(predict(model23,norm_test)==1,2,3)))
test_pred <- cbind(test_pred,(ifelse(predict(model24,norm_test)==1,2,4)))
test_pred <- cbind(test_pred,(ifelse(predict(model34,norm_test)==1,3,4)))

colnames(test_pred) <- c("model12","model13","model14","model23","model24","model34")
test_pred$Prediction <- apply(test_pred,1,function (x) names(sort(summary(as.factor(x)),decreasing=TRUE))[1])
test_pred$Prediction <- as.numeric(test_pred$Prediction)
print(paste("Accuracy=",mean(Y_test==test_pred$Prediction)*100))
