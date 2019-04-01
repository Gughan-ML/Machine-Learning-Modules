require(Matrix)
library(e1071)
library(reshape2)
library(quanteda)
library("caret")
setwd("C:/Users/Gughan/Desktop/Advanced Statistics/")

###################### Reading the articles  ######################

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

##################################### train ########################################
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
# Get a sparse data matrix X (rows: training exmaples, columns: # of occurrences for each of features)
X_train <- sparseMatrix(x=X_train_data[,3], i=X_train_data[,1], j=X_train_data[,2])
rm(dataFile,dataLines,dataTokens,X_list)



#################### Naive Bayes ##################################
nb_train <- X_train_data
nb_train <- sparseMatrix(x=nb_train[,3], i=nb_train[,1], j=nb_train[,2]) #Creating SparseMatrix
nb_train<- as.matrix(nb_train)
words <- c(1:max(X_train_data[,2]))
colnames(nb_train) <- words
input_data<-as.dfm(nb_train) #document term matrix 


nb_test <- X_test_data
nb_test <- sparseMatrix(x=nb_test[,3], i=nb_test[,1], j=nb_test[,2])
nb_test<- as.matrix(nb_test)
words <- c(1:max(X_test_data[,2]))
colnames(nb_test) <- words
test_data<-as.dfm(nb_test)
test_data <- dfm_select(test_data,input_data)

################# Without Laplace Smoothing ############################
model<-textmodel_nb(input_data,Y_train,smooth = 0,distribution = c("Bernoulli"))
pred1 <- predict(model,input_data[1:100,])
pred2 <- predict(model,input_data[1001:2000,])
pred3 <- predict(model,input_data[2001:3000,])
pred4 <- predict(model,input_data[3001:4000,])
pred <- c(pred1,pred2,pred3,pred4)

confusionMatrix(as.factor(pred),as.factor(Y_train))

################## With Laplace Smoothing ###########################
model<-textmodel_nb(input_data,Y_train,smooth = 1,distribution = c("Bernoulli"))
pred1 <- predict(model,input_data[1:1000,])
pred2 <- predict(model,input_data[1001:2000,])
pred3 <- predict(model,input_data[2001:3000,])
pred4 <- predict(model,input_data[3001:4000,])
pred <- c(pred1,pred2,pred3,pred4)

confusionMatrix(as.factor(pred),as.factor(Y_train),mode="prec_recall")

pred5 <- predict(model,test_data[1:1000,])
pred6 <- predict(model,test_data[1001:2400,])
pred_test <- c(pred5,pred6)

confusionMatrix(as.factor(pred_test),as.factor(Y_test),mode = "prec_recall")

##################### Multinominal NB with laplace smoothing ############
multi_model<-textmodel_nb(input_data,Y_train,smooth = 1,distribution = c("multinomial"))
pred1 <- predict(multi_model,input_data[1:1000,])
pred2 <- predict(multi_model,input_data[1001:2000,])
pred3 <- predict(multi_model,input_data[2001:3000,])
pred4 <- predict(multi_model,input_data[3001:4000,])
pred <- c(pred1,pred2,pred3,pred4)

confusionMatrix(as.factor(pred),as.factor(Y_train),mode="prec_recall")

pred5 <- predict(multi_model,test_data[1:1000,])
pred6 <- predict(multi_model,test_data[1001:2400,])
pred_test <- c(pred5,pred6)
confusionMatrix(as.factor(pred_test),as.factor(Y_test))
