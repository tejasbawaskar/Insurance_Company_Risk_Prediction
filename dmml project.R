library(randomForest)
library(mlbench)
library(caret)
library(doParallel)
library(e1071)
library(caTools)
library(MASS)
library(pracma)
library(rpart)
library(rpart.plot)
library(glmnet)
library(ggplot2)
library(gridExtra)

# getwd()
# dataset <- read.csv("train_midterm_data.csv")
# test_dataset <- read.csv("test2.csv")
# ncol(test_dataset)
# nrow(dataset)
# nrow(test_dataset)
# nrow(test_train_data)
# #Introducing response column to test data

## file read
test <- read.csv("projecttest.csv", header = T)
dataset <- read.csv("projecttrain.csv", header = T)
test_dataset <- test
test_dataset[,"Response"] <- rep(NA,19765)

#Merging test and train data to impute missing values
test_train_data <- rbind(dataset,test_dataset)


#Imputing with mean and median
mean_impute <- function(dat) {
  w <- is.na(dat$Employment_Info_1)
  z <- dat$Employment_Info_1
  z[w] <- mean(dat$Employment_Info_1, na.rm = T)
  return(z)
}
test_train_data$Employment_Info_1 <- mean_impute(test_train_data)
summary(test_train_data$Employment_Info_1)
median_impute <- function(dat, column) {
  w <- is.na(dat[[column]])
  z <- dat[[column]]
  z[w] <- median(dat[[column]], na.rm = T)
  return(z)
}
for (column in c("Employment_Info_4","Employment_Info_6","Insurance_History_5",
                 "Family_Hist_2","Family_Hist_3","Family_Hist_4",
                 "Family_Hist_5","Medical_History_1","Medical_History_10",
                 "Medical_History_15","Medical_History_24",
                 "Medical_History_32")) {
  test_train_data[[column]] <- median_impute(test_train_data, column)
}

new_train <- mice(a,m=5,maxit=10,meth='pmm',seed=500) 
summary(new_train)
ncol(new_train$data)
test_train_data <- complete(new_train,1)
test_train_data
final_train_data <- test_train_data[1:59381,]
final_test_data <- test_train_data[59382:79146,]
str(final_train_data)

#to make the algorithms run via parallel processing
cl <- makeCluster(detectCores(), type='PSOCK')
registerDoParallel(cl)

str(test_train_data)
test_train_data <- na.omit(b)
nrow(test_train_data)
percentage_na(test_train_data,117)

#training with boruta
#       boruta.train <- Boruta(Response ~ ., data = a, doTrace=2)
#       print(boruta.train) 
#       final.boruta <- TentativeRoughFix(boruta.train)
#       print(final.boruta)
#       getSelectedAttributes(final.boruta, withTentative = F)
#       boruta.df <- attStats(final.boruta)
#       print(boruta.df[which(boruta.df$decision == "Rejected"),])


write.csv(test_train_data,"columnnumbers.csv")

#getting the important variables only
 new_test_train_data <- test_train_data[,-c(1,5,6,7,32,54,55,62,64,71,74,76,77,86,89,95,101,108,113)]
 nrow(new_test_train_data)
#splitting into train and test
     split <- sample.split(new_test_train_data$Response,SplitRatio = 0.75)
     new_test_train_data_train <- subset(new_test_train_data, split == TRUE)
     new_test_train_data_test <- subset(new_test_train_data, split == FALSE)
     str(new_test_train_data )
     ncol(new_test_train_data)

#applying linear regression

#predicting linear regression
    predictions <- predict(main_model, new_test_train_data_test[,1:98])
    summary(predictions)
l <- lm(Response ~ . , data = test_train_data)
summary(l)
#correlation matrix for the new data for further removing multicollinearity
   correlationMatrix <- cor(new_test_train_data[,-1])
   correlationMatrix
   highlycorrelated <- findCorrelation(correlationMatrix, cutoff = 0.5)
   highlycorrelated 

#stepwise regression
ridgec <- lm.ridge (Response ~ .,data = test_train_data, lambda = 0.05)
plot(ridgec)
select(ridgec)
names(ridgec)
str(ridgec$coef)
ridgec

#running random forest
new_test_train_data$Response <- as.factor(new_test_train_data$Response)
rf_output <- randomForest(as.factor(Response) ~ ., data = new_test_train_data_train, ntree = 1000 )
sort(rf_output$importance)
ncol(new_test_train_data_train)
varImpPlot(rf_output,n.var = min(40) )
importance(rf_output)
preditions_new <- predict(rf_output, new_test_train_data_test)

#accuracy
mean(preditions_new == new_test_train_data_test$Response)

#outcome
submit <- data.frame(PassengerId = dataset_new$Id, round(Survived = preditions_new))
submit

#Feature selection using Caret
control <- trainControl(method="repeatedcv", number=10, repeats=3)
model <- train(Response ~., data= test_train_data, method= "glmStepAIC", trControl=control)
varImp(model)
model
names(model)
str(model$decision.values)
?train

#Principal Component Analysis using caret
test_train_data <- test_train_data[,-1]
test_train_pca <- test_train_data[,c(4,8,9,10,11,12,15,17,29,34,35,36,37,47,52,69,127)]

ncol(test_train_pca)

pca1 <- preProcess(x = test_train_pca[-17], method = 'pca', pcaComp = 4)
test_train_pca <- predict(pca1, test_train_pca)
test_train_pca <- test_train_pca[c(seq(2,5,1),1)]
test_train_data$AgeWt <- test_train_data$Ins_Age*test_train_data$Wt

keywords_sum <- function(data) {
  sums <- as.numeric()
  cols <- grep("Medical_Keyword_", names(data))
  for (r in 1:nrow(data)) {
    s <- 0
    for (c in cols) {
      if (data[[c]][r] == 1) {
        s <- s + 1
      }
    }
    sums <- append(sums, s)
  }
  return (sums)
}
ababab <- keywords_sum(test_train_data)
test_train_data$No_Of_Keywords <- ababab

test_train_data <- test_train_data[,-c(4,8,9,10,11,12,15,17,29,34,35,36,37,47,52,69)]

test_train_data$PC1 <- test_train_pca$PC1
test_train_data$PC2 <- test_train_pca$PC2
test_train_data$PC3 <- test_train_pca$PC3
test_train_data$PC4 <- test_train_pca$PC4

ncol(test_train_data)
names(test_train_data)


main_test <- predict(pca1,new_test_train_data_test[-1])
main_test <- main_test[c(seq(2,67,1),1)]


#Random forest on Data after PCA
rf_output_new <- randomForest(as.factor(Response) ~ ., data = new_test_train_data_train_1, ntree = 1000 )
predictions_new_1 <- predict(rf_output_new,main_test)
mean(predictions_new_1 == main_test$Response)

#SVM on Data after PCA
model_svm <- svm(Response ~ ., kernel = "radial", cost = runif(10,min = 10, max = 50), 
                 gamma = runif(10,0,2), data = new_test_train_data_train_1, scale = F)


#Converting integers into factors
str(final_train_data[,71:127])
for(i in names(final_train_data[,-c(1,71:128)])) {
  if(class(final_train_data[,i]) == "integer")
  {
    final_train_data[,i] <- as.factor(final_train_data[,i])
  }
}

str(final_test_data[,71:127])
for(i in names(final_test_data[,-71:127])) {
  if(class(final_test_data[,i]) == "integer")
  {
    final_test_data[,i] <- as.factor(final_test_data[,i])
  }
}

#Making dummy variables
dummies <- dummyVars(Response ~ .  - Medical_History_2,final_train_data)
head(predict(dummies, newdata = final_train_data))
final_train_data <- (predict(dummies, newdata = final_train_data))
final_train_data <- as.data.frame(final_train_data)
head(final_train_data,5)
final_train_data$Response <- dataset$Response
final_train_data[,301:302]
ncol(final_train_data)

split <- sample.split(data_withdummy$Response,SplitRatio = 0.75)
data_withdummy_train <- subset(data_withdummy, split == TRUE)
data_withdummy_test <- subset(data_withdummy, split == FALSE)

ncol(data_withdummy_train)
ncol(data_withdummy_test)

#Running Random Forest on Data with 1-c
rf_output_withdummies <- randomForest(as.factor(Response) ~ ., data = data_withdummy_train, ntree = 2000 )
sort(rf_output$importance)
ncol(new_test_train_data_train)
varImpPlot(rf_output,n.var = min(40) )
importance(rf_output_withdummies)
preditions_new_binary <- predict(rf_output_withdummies, data_withdummy_test)
mean(preditions_new_binary == data_withdummy_test$Response)

#Running SVM on data with 1-c
model_svm <- svm(as.factor(Response) ~ ., kernel = "radial", cost = 1, 
                gamma = 1, data = data_withdummy_train, scale = F)
prediction_svm <- predict(model_svm,data_withdummy_test)
mean(prediction_svm == data_withdummy_test$Response)

k <- c(1,5,7,10,11,12,13,14,20,21,22,23,27,28,29,30,32,35,36,40,42,43,44,46,48)
kr <- mk+62
kr
est_train_data <- test_train_data[-mkr]

#Rearranging the response
r <- rep(NA,time=nrow(test_train_data[1:59381,]))
for (i in 1:nrow(test_train_data[1:59381,])) {
  if (test_train_data$Response[i] == 1) {
    r[i] <- 1
  } else if (test_train_data$Response[i] == 2) {
    r[i] <- 2
  } else if (test_train_data$Response[i] == 3) {
    r[i] <- 7
  } else if (test_train_data$Response[i] == 4) {
    r[i] <- 8
  } else if (test_train_data$Response[i] == 5) {
    r[i] <- 3
  } else if (test_train_data$Response[i] == 6) {
    r[i] <- 4
  } else if (test_train_data$Response[i] == 7) {
    r[i] <- 5
  } else if (test_train_data$Response[i] == 8) {
    r[i] <- 6
  }
}
reordered <- test_train_data
reordered$Response[1:59381] <- r

#Stepwise Regression
lmod <- lm(Response ~ ., data = test_train_data[1:59381,])
summary(lmod)
step <- stepAIC(lmod, direction="backward")
summary(step)

lmod_reordered <- lm(Response ~ ., data = reordered[1:59381,])
summary(lmod)
step_reordered <- stepAIC(lmod_reordered, direction="backward")
summary(step_reordered)# 

reordered_train <- reordered[1:59381,]
reordered_train$BMI <- dataset$BMI
reordered_train$Wt <- dataset$Wt
reordered_train$Ins_Age <- dataset$Ins_Age

### FUNCTION FOR TRANSFORMATIONS ###

trans <- function(dat,t1,t2) {
  for (c1 in (names(dat))[names(dat) != "Response"]) {
    bestc2forc1 <- as.character()
    bestc2forc1r2 <- 0
    for (c2 in (names(dat))[names(dat) != "Response"]) {
      form1 <- as.formula(paste0("Response~I(", c1,"*", c2,")"))
      mod1 <- lm(form1, data=reordered_train)
      rs1 <- summary(mod1)$r.squared
      form2 <- as.formula(paste0("Response~log((", c1,"*", c2,")+0.00000000001)"))
      mod2 <- lm(form2, data=reordered_train)
      rs2 <- summary(mod2)$r.squared
      form3 <- as.formula(paste0("Response~((", c1,"*", c2,"))^2"))
      mod3 <- lm(form3, data=reordered_train)
      rs3 <- summary(mod3)$r.squared  
      form4 <- as.formula(paste0("Response~(sigmoid(", c1,"*", c2,"))"))
      mod4 <- lm(form4, data=reordered_train)
      rs4 <- summary(mod4)$r.squared
      form5 <- as.formula(paste0("Response~(tanh(", c1,"*", c2,"))"))
      mod5 <- lm(form5, data=reordered_train)
      rs5 <- summary(mod5)$r.squared
      sorted <- sort(c(rs1,rs2,rs3,rs4,rs5), decreasing = T)
      if (sorted[1] >= t1) {
        if (sorted[1] == rs1) {
          print(paste("1:",c1,"&",c2,":","Use as it is, r2 =",rs1))
        } else if (sorted[1] == rs2) {
          print(paste("1:",c1,"&",c2,":","Use log, r2 =",rs2))
        } else if (sorted[1] == rs3) {
          print(paste("1:",c1,"&",c2,":","Use square, r2 =",rs3))
        } else if (sorted[1] == rs4) {
          print(paste("1:",c1,"&",c2,":","Use sigmoid, r2 =",rs4))
        } else if (sorted[1] == rs5) {
          print(paste("1:",c1,"&",c2,":","Use tanh, r2 =",rs5))
        }
      }
      if (sorted[2] >= t2) {
        if (sorted[2] == rs1) {
          print(paste("2:",c1,"&",c2,":","Use as it is, r2 =",rs1))
        } else if (sorted[2] == rs2) {
          print(paste("2:",c1,"&",c2,":","Use log, r2 =",rs2))
        } else if (sorted[2] == rs3) {
          print(paste("2:",c1,"&",c2,":","Use square, r2 =",rs3))
        } else if (sorted[2] == rs4) {
          print(paste("2:",c1,"&",c2,":","Use sigmoid, r2 =",rs4))
        } else if (sorted[2] == rs5) {
          print(paste("2:",c1,"&",c2,":","Use tanh, r2 =",rs5))
        }
        print ("")
      }
      if (sorted[1] > bestc2forc1r2) {
        bestc2forc1r2 <- sorted[1]
        bestc2forc1 <- c2
      }
    }
    cat ("\n")
    print (paste("The best c2 for",c1,"is",bestc2forc1,", r2=",bestc2forc1r2))
    print ("-------------------------------")
  }
}

trans(reordered_train[-c(2,45,63:110,112:115)],0.1,0.09)
#---------------------#

for (c1 in names(reordered_train)[-c(63:111,2,113:115,45)]) {
  bestc2forc1 <- as.character()
  bestc2forc1r2 <- 0
  for (c2 in names(reordered_train)[-c(63:111,2,113:115,45)]) {
    form1 <- as.formula(paste0("Response~I(", c1,"*", c2,")"))
    mod1 <- lm(form1, data=reordered_train)
    rs1 <- summary(mod1)$r.squared
    form2 <- as.formula(paste0("Response~log((", c1,"*", c2,")+0.00000000001)"))
    mod2 <- lm(form2, data=reordered_train)
    rs2 <- summary(mod2)$r.squared
    form3 <- as.formula(paste0("Response~((", c1,"*", c2,"))^2"))
    mod3 <- lm(form3, data=reordered_train)
    rs3 <- summary(mod3)$r.squared  
    form4 <- as.formula(paste0("Response~(sigmoid(", c1,"*", c2,"))"))
    mod4 <- lm(form4, data=reordered_train)
    rs4 <- summary(mod4)$r.squared
    form5 <- as.formula(paste0("Response~(tanh(", c1,"*", c2,"))"))
    mod5 <- lm(form5, data=reordered_train)
    rs5 <- summary(mod5)$r.squared
    sorted <- sort(c(rs1,rs2,rs3,rs4,rs5), decreasing = T)
    if (sorted[1] >= 0.17) {
      if (sorted[1] == rs1) {
        print(paste("1:",c1,"&",c2,":","Use as it is, r2 =",rs1))
      } else if (sorted[1] == rs2) {
        print(paste("1:",c1,"&",c2,":","Use log, r2 =",rs2))
      } else if (sorted[1] == rs3) {
        print(paste("1:",c1,"&",c2,":","Use square, r2 =",rs3))
      } else if (sorted[1] == rs4) {
        print(paste("1:",c1,"&",c2,":","Use sigmoid, r2 =",rs4))
      } else if (sorted[1] == rs5) {
        print(paste("1:",c1,"&",c2,":","Use tanh, r2 =",rs5))
      }
    }
    if (sorted[2] >= 0.14) {
      if (sorted[2] == rs1) {
        print(paste("2:",c1,"&",c2,":","Use as it is, r2 =",rs1))
      } else if (sorted[2] == rs2) {
        print(paste("2:",c1,"&",c2,":","Use log, r2 =",rs2))
      } else if (sorted[2] == rs3) {
        print(paste("2:",c1,"&",c2,":","Use square, r2 =",rs3))
      } else if (sorted[2] == rs4) {
        print(paste("2:",c1,"&",c2,":","Use sigmoid, r2 =",rs4))
      } else if (sorted[2] == rs5) {
        print(paste("2:",c1,"&",c2,":","Use tanh, r2 =",rs5))
      }
      print ("")
    }
    if (sorted[1] > bestc2forc1r2) {
      bestc2forc1r2 <- sorted[1]
      bestc2forc1 <- c2
    }
  }
  cat ("\n")
  print (paste("The best c2 for",c1,"is",bestc2forc1,", r2=",bestc2forc1r2))
  print ("-------------------------------")
}

#PI2,, BMI, Age, Wt, No_of_Keywords_Sq, MH 23, MH 39, MH 6, MH 16, MH 30, MH40

summary(lm(Response ~ Medical_History_1, data = reordered_train))

###

for (c1 in names(reordered_train)[c(45,60,30,38,52,61,112,113)]) {
  bestc2forc1 <- as.character()
  bestc2forc1r2 <- 0
  for (c2 in names(reordered_train)[-c(63:111,2,113:115,45)]) {
    form1 <- as.formula(paste0("Response~I(", c1,"*", c2,")"))
    mod1 <- lm(form1, data=reordered_train)
    rs1 <- summary(mod1)$r.squared
    form2 <- as.formula(paste0("Response~log((", c1,"*", c2,")+0.00000000001)"))
    mod2 <- lm(form2, data=reordered_train)
    rs2 <- summary(mod2)$r.squared
    form3 <- as.formula(paste0("Response~((", c1,"*", c2,"))^2"))
    mod3 <- lm(form3, data=reordered_train)
    rs3 <- summary(mod3)$r.squared  
    form4 <- as.formula(paste0("Response~(sigmoid(", c1,"*", c2,"))"))
    mod4 <- lm(form4, data=reordered_train)
    rs4 <- summary(mod4)$r.squared
    form5 <- as.formula(paste0("Response~(tanh(", c1,"*", c2,"))"))
    mod5 <- lm(form5, data=reordered_train)
    rs5 <- summary(mod5)$r.squared
    sorted <- sort(c(rs1,rs2,rs3,rs4,rs5), decreasing = T)
    if (sorted[1] >= 0.17) {
      if (sorted[1] == rs1) {
        print(paste("1:",c1,"&",c2,":","Use as it is, r2 =",rs1))
      } else if (sorted[1] == rs2) {
        print(paste("1:",c1,"&",c2,":","Use log, r2 =",rs2))
      } else if (sorted[1] == rs3) {
        print(paste("1:",c1,"&",c2,":","Use square, r2 =",rs3))
      } else if (sorted[1] == rs4) {
        print(paste("1:",c1,"&",c2,":","Use sigmoid, r2 =",rs4))
      } else if (sorted[1] == rs5) {
        print(paste("1:",c1,"&",c2,":","Use tanh, r2 =",rs5))
      }
    }
    if (sorted[2] >= 0.14) {
      if (sorted[2] == rs1) {
        print(paste("2:",c1,"&",c2,":","Use as it is, r2 =",rs1))
      } else if (sorted[2] == rs2) {
        print(paste("2:",c1,"&",c2,":","Use log, r2 =",rs2))
      } else if (sorted[2] == rs3) {
        print(paste("2:",c1,"&",c2,":","Use square, r2 =",rs3))
      } else if (sorted[2] == rs4) {
        print(paste("2:",c1,"&",c2,":","Use sigmoid, r2 =",rs4))
      } else if (sorted[2] == rs5) {
        print(paste("2:",c1,"&",c2,":","Use tanh, r2 =",rs5))
      }
      print ("")
    }
    if (sorted[1] > bestc2forc1r2) {
      bestc2forc1r2 <- sorted[1]
      bestc2forc1 <- c2
    }
  }
  cat ("\n")
  print (paste("The best c2 for",c1,"is",bestc2forc1,", r2=",bestc2forc1r2))
  print ("-------------------------------")
}

summary(lm(Response~Medical_History_23,data=reordered_train))
summary(lm(Response~Medical_History_39,data=reordered_train))

reordered_train$x23_39<-reordered_train$Medical_History_23*reordered_train$Medical_History_39

reordered_train$Medical_History_23 <- NULL
reordered_train$Medical_History_39 <- NULL
reordered_train$Wt <- NULL

#-------------------------------------------
#PI2,, BMI, Age, No_of_Keywords_Sq, MH 23, MH 39, MH 6, MH 16, MH 30, MH40

for (c1 in names(reordered_train)[c(111,110,112,113)]) {
  bestc2forc1 <- as.character()
  bestc2forc1r2 <- 0
  for (c2 in names(reordered_train)[-c(61:108,2,114,109,113)]) {
    form1 <- as.formula(paste0("Response~I(", c1,"*", c2,")"))
    mod1 <- lm(form1, data=reordered_train)
    rs1 <- summary(mod1)$r.squared
    form2 <- as.formula(paste0("Response~log((", c1,"*", c2,")+0.00000000001)"))
    mod2 <- lm(form2, data=reordered_train)
    rs2 <- summary(mod2)$r.squared
    form3 <- as.formula(paste0("Response~((", c1,"*", c2,"))^2"))
    mod3 <- lm(form3, data=reordered_train)
    rs3 <- summary(mod3)$r.squared  
    form4 <- as.formula(paste0("Response~(sigmoid(", c1,"*", c2,"))"))
    mod4 <- lm(form4, data=reordered_train)
    rs4 <- summary(mod4)$r.squared
    form5 <- as.formula(paste0("Response~(tanh(", c1,"*", c2,"))"))
    mod5 <- lm(form5, data=reordered_train)
    rs5 <- summary(mod5)$r.squared
    sorted <- sort(c(rs1,rs2,rs3,rs4,rs5), decreasing = T)
    if (sorted[1] >= 0.17) {
      if (sorted[1] == rs1) {
        print(paste("1:",c1,"&",c2,":","Use as it is, r2 =",rs1))
      } else if (sorted[1] == rs2) {
        print(paste("1:",c1,"&",c2,":","Use log, r2 =",rs2))
      } else if (sorted[1] == rs3) {
        print(paste("1:",c1,"&",c2,":","Use square, r2 =",rs3))
      } else if (sorted[1] == rs4) {
        print(paste("1:",c1,"&",c2,":","Use sigmoid, r2 =",rs4))
      } else if (sorted[1] == rs5) {
        print(paste("1:",c1,"&",c2,":","Use tanh, r2 =",rs5))
      }
    }
    if (sorted[2] >= 0.14) {
      if (sorted[2] == rs1) {
        print(paste("2:",c1,"&",c2,":","Use as it is, r2 =",rs1))
      } else if (sorted[2] == rs2) {
        print(paste("2:",c1,"&",c2,":","Use log, r2 =",rs2))
      } else if (sorted[2] == rs3) {
        print(paste("2:",c1,"&",c2,":","Use square, r2 =",rs3))
      } else if (sorted[2] == rs4) {
        print(paste("2:",c1,"&",c2,":","Use sigmoid, r2 =",rs4))
      } else if (sorted[2] == rs5) {
        print(paste("2:",c1,"&",c2,":","Use tanh, r2 =",rs5))
      }
      print ("")
    }
    if (sorted[1] > bestc2forc1r2) {
      bestc2forc1r2 <- sorted[1]
      bestc2forc1 <- c2
    }
  }
  cat ("\n")
  print (paste("The best c2 for",c1,"is",bestc2forc1,", r2=",bestc2forc1r2))
  print ("-------------------------------")
}

reordered_train$xBMI_Age<-reordered_train$BMI * reordered_train$Ins_Age

#Ins_Age ---- 1
for (p in seq(0.1,10,0.1)) {
  a <- (reordered_train$Ins_Age)^p
  moddddd <- lm(Response ~ a, data = reordered_train)
  if (summary(moddddd)$r.squared >= 0.0598) {
    print (paste(p,":",summary(moddddd)$r.squared))
  }
}
reordered_train$Ins_Age <- dataset$Ins_Age

#BMI ---- 0.9
for (p in seq(0.1,10,0.1)) {
  a <- (reordered_train$BMI)^p
  moddddd <- lm(Response ~ a, data = reordered_train)
  if (summary(moddddd)$r.squared >= 0.165) {
    print (paste(p,":",summary(moddddd)$r.squared))
  }
}
reordered_train$BMI <- dataset$BMI^0.9

#xBMI_Age ---- 1.2
for (p in seq(0.1,10,0.1)) {
  a <- (reordered_train$xBMI_Age)^p
  moddddd <- lm(Response ~ a, data = reordered_train)
  if (summary(moddddd)$r.squared >= 0.131) {
    print (paste(p,":",summary(moddddd)$r.squared))
  }
}

reordered_train$xBMI_Age <- reordered_train$xBMI_Age^1.2

#Product_Info_4 ---- <= 0.074

reordered_train$Product_Info_4 <- dataset$Product_Info_4

for (p in seq(0.1,10,0.1)) {
  a <- (reordered_train$Product_Info_4)^p
  moddddd <- lm(Response ~ a, data = reordered_train)
  if (summary(moddddd)$r.squared >= 0.04) {
    print (paste(p,":",summary(moddddd)$r.squared))
  }
}

part <- rpart(Response~Product_Info_4,data=reordered_train)
rpart.plot(part)

reordered_train$Product_Info_4 <- as.numeric(reordered_train$Product_Info_4 <= 0.074)

#PI2,, BMI, Age, No_of_Keywords_Sq, MH 23, MH 39, MH 6, MH 16, MH 30, MH40

# N Keywords ---- 2.1

summary(reordered_train$No_Of_Keywords)

for (p in seq(0.1,10,0.1)) {
  a <- (reordered_train$No_Of_Keywords)^p
  moddddd <- lm(Response ~ a, data = reordered_train)
  if (summary(moddddd)$r.squared >= 0.09) {
    print (paste(p,":",summary(moddddd)$r.squared))
  }
}

reordered_train$No_Of_Keywords <- reordered_train$No_Of_Keywords^0.7

#----------------

part <- rpart(Response~Medical_History_2,data=reordered_train)
rpart.plot(part)
part <- rpart(Response~Medical_History_1,data=reordered_train)
rpart.plot(part)
part <- rpart(Response~Employment_Info_2,data=reordered_train)
rpart.plot(part)

for (i in 1:ncol(reordered_train)) {
  print(paste(i,":",class(reordered_train[,i])))
}
names(reordered_train)
x.train <- as.matrix(reordered_train[,-c(2,109)])
y.train <- reordered_train$Response

cv.en <- cv.glmnet(x.train, y.train, family="gaussian", alpha=0.5)
best.lambda <- cv.en$lambda.min
en <- glmnet(x.train, y.train, family="gaussian", alpha=0.5, lambda=best.lambda)
summary(abs(as.numeric(coef(en))))
impvar <- x.train[,abs(as.numeric(coef(en)))[2:length(as.numeric(coef(en)))] > 0.15]
#----------------
j <- lm(Response~.,data=reordered_train)
summary(j)
jpred <- round(predict(j))
jpred[jpred < 1] <- 1
summary(jpred)
mean(jpred == reordered_train$Response)

#IMPORTANT VARS LM
s_reordered_train <- reordered_train[,-2]
summary(lm(Response~.,s_reordered_train)$coef)
imp_logi <- abs(lm(Response~.,s_reordered_train)$coef) >= 0.2563

reordered_train2 <- s_reordered_train[,-108]
imp_logi2 <- imp_logi[2:length(imp_logi)]
reordered_train2 <- s_reordered_train[,imp_logi2]
reordered_train <- cbind(reordered_train2,
                         Product_Info_2=dataset$Product_Info_2,
                         Response=r)
summary(lm(Response~.^2,s_reordered_train))
set.seed(666)
split <- sample.split(reordered_train, 0.6)
reordered_train_train <- reordered_train[split,]
reordered_train_test <- reordered_train[!split,]

Stepwise Regression

lmod2 <- lm(Response ~ ., data = reordered_train_train)

summary(lmod2)

step <- stepAIC(lmod2, direction="both")

summary(step)

RMSE <- function(actual, predict){
  sqrt(mean((actual - predict)^2))
}

## LINEAR REGRESSION ##
lmod2 <- lm(Response ~ ., data = reordered_train_train)
lmod2pred <- predict(lmod2,newdata = reordered_train_test)
lmod2pred[lmod2pred < 1] <- 1
summary(lmod2pred)

RMSE(reordered_train_test$Response,lmod2pred)

#-------------
for (aa in seq(0,1,0.1)) {
  resptr <- reordered_train_train$Response
  reordered_train_train$Response2<-2*((resptr-min(resptr))/(max(resptr)-min(resptr)))-1
  reordered_train_train$Response2<-sinh(2.2*reordered_train_train$Response2 - 0.5)
  reordered_train_train$Response2 <- reordered_train_train$Response2^2 + reordered_train_train$Response2
  lmod3 <- lm(Response2 ~ .-Response, data = reordered_train_train)

  lmod3pred <- predict(lmod3,newdata = reordered_train_test)
  z <- lmod3pred
  z[z<0] <- 0
  y <- sqrt(z)
  a <- (asinh(y) - 0.5)/2.2
  b <- (a+1)/2
  c <- b*(max(resptr)-min(resptr))
  d <- c + min(resptr)
  e <- abs(d)
  f <- 7*((e-min(e))/(max(e)-min(e)))+1

  print(paste(aa,":",RMSE(f,reordered_train_test$Response)))

}
#-------------

names(lmod2)
lmod2coeff <- lmod2$coefficients[c(2,21:length(lmod2$coefficients))]
summary(abs(lmod2coeff))
impcols_threshold <- sort(abs(lmod2coeff))[20]
impcols <- names(lmod2coeff)[lmod2coeff>=impcols_threshold]
names(reordered_train[,impcols])
names(reordered_train)
impcols <- append(impcols,c("No_Of_Keywords","BMI","Ins_Age","Product_Info_4"))

## RPART ##
rpart_impcols <- rpart(Response ~ .,
                       data = reordered_train_train[,c(impcols,"Response")])
rpart.plot(rpart_impcols)
rpartpred <- predict(rpart_impcols,reordered_train_test)

RMSE(reordered_train_test$Response,rpartpred)

## RANDOM FOREST ##

rf_model <- randomForest(Response ~ .,
                       data = reordered_train_train[,c(impcols,"Response")])
rfpred <- predict(rf_model,reordered_train_test)
RMSE(reordered_train_test$Response,rfpred)

## SVM ##

model_svm <- svm(Response ~ ., kernel = "radial", cost = 8,
                 data = reordered_train_train[,c(impcols,"Response")],
                 scale = F)
svmpred <- predict(model_svm,reordered_train_test)
svmpred[svmpred < 1] = 1
RMSE(reordered_train_test$Response,svmpred)

#PREDICTING ON TRAIN

RMSE(reordered_train_train$Response,lmod2pred)
RMSE(reordered_train_train$Response,rpartpred)
RMSE(reordered_train_train$Response,rfpred)
RMSE(reordered_train_train$Response,predict(model_svm,reordered_train_train))
