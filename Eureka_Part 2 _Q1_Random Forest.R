if("pacman" %in% rownames(installed.packages()) == FALSE) {install.packages("pacman")} 
pacman::p_load("caret","ROCR","lift","glmnet","MASS","e1071","mice","gdata") 
library(pROC)
library(dplyr)
library(caret)
library(mice)
library(gdata)

#Importing Data
eureka<-read.csv("C:\\Users\\urvipalvankar\\Urvi\\Master of Management Analytics\\831 - Marketing Analytics\\Mid Term Assignment\\eureka.csv",na.strings=c(""," ","NA"), header=TRUE,stringsAsFactors = TRUE)

#Discovering data
str(eureka)
dim(eureka)

#correcting target variable - replacing 2 and 3 with 1
eureka$converted_in_7days<-ifelse(eureka$converted_in_7days>1,1,eureka$converted_in_7days)

#Looking at the missing data
missing_data_columnwise<-sort(colSums(is.na(eureka)),decreasing = TRUE)
missing_data_columnwise
#md.pattern(eureka)
#all missing data are from the same rows

#option 1 create features for the missing values
# Create a custom function to fix missing values ("NAs") and preserve the NA info as surrogate variables
fixNAs<-function(data_frame){
  # Define reactions to NAs
  integer_reac<-0
  factor_reac<-"FIXED_NA"
  character_reac<-"FIXED_NA"
  date_reac<-as.Date("1900-01-01")
  # Loop through columns in the data frame and depending on which class the variable is, apply the defined reaction and create a surrogate
  
  for (i in 1 : ncol(data_frame)){
    if (class(data_frame[,i]) %in% c("numeric","integer")) {
      if (any(is.na(data_frame[,i]))){
        data_frame[,paste0(colnames(data_frame)[i],"_surrogate")]<-
          as.factor(ifelse(is.na(data_frame[,i]),"1","0"))
        data_frame[is.na(data_frame[,i]),i]<-integer_reac
      }
    } else
      if (class(data_frame[,i]) %in% c("factor")) {
        if (any(is.na(data_frame[,i]))){
          data_frame[,i]<-as.character(data_frame[,i])
          data_frame[,paste0(colnames(data_frame)[i],"_surrogate")]<-
            as.factor(ifelse(is.na(data_frame[,i]),"1","0"))
          data_frame[is.na(data_frame[,i]),i]<-factor_reac
          data_frame[,i]<-as.factor(data_frame[,i])
          
        } 
      } else {
        if (class(data_frame[,i]) %in% c("character")) {
          if (any(is.na(data_frame[,i]))){
            data_frame[,paste0(colnames(data_frame)[i],"_surrogate")]<-
              as.factor(ifelse(is.na(data_frame[,i]),"1","0"))
            data_frame[is.na(data_frame[,i]),i]<-character_reac
          }  
        } else {
          if (class(data_frame[,i]) %in% c("Date")) {
            if (any(is.na(data_frame[,i]))){
              data_frame[,paste0(colnames(data_frame)[i],"_surrogate")]<-
                as.factor(ifelse(is.na(data_frame[,i]),"1","0"))
              data_frame[is.na(data_frame[,i]),i]<-date_reac
            }
          }  
        }       
      }
  } 
  return(data_frame) 
}

eureka_fixed<-fixNAs(eureka) 

#checking for missing data
#md.pattern(eureka_fixed)

#Discovering data
str(eureka_fixed)
dim(eureka_fixed)

#Converting a few columns to factor format
cols <- c(8,16:18,22,26,40,42,44,46,48,50,52,54,56,58,61)
eureka_fixed[,cols] <- lapply(eureka_fixed[,cols], factor)
str(eureka_fixed)

#converting source medium factor column
eureka_fixed$sourceMedium <- as.character(eureka_fixed$sourceMedium)
eureka_fixed$sourceMedium <- unlist(lapply(strsplit(as.character(eureka_fixed$sourceMedium), "/ "), '[[', 2))
eureka_fixed$sourceMedium <- as.factor(eureka_fixed$sourceMedium)

#removing client id, region and date from the dataset
eureka_fixed<-eureka_fixed[,-6]
eureka_fixed<-eureka_fixed[,-11]

#combining rare categories
combinerarecategories<-function(data_frame,mincount){ 
  for (i in 1 : ncol(data_frame)){
    a<-data_frame[,i]
    replace <- names(which(table(a) < mincount))
    levels(a)[levels(a) %in% replace] <-paste("Other",colnames(data_frame)[i],sep=".")
    data_frame[,i]<-a }
  return(data_frame) }

table(eureka_fixed$region)
eureka_fixed<-combinerarecategories(eureka_fixed,200)
str(eureka_fixed)


#Creating Training and Testing set
set.seed(456)
inTrain <- createDataPartition(y = eureka_fixed$converted_in_7days,
                               p = 0.7, list = FALSE)
training <- eureka_fixed[ inTrain,]
testing <- eureka_fixed[ -inTrain,]


#Random Forest Model - Question 1 Part 2
control <- trainControl(method = 'repeatedcv',
                        number = 10,
                        repeats = 1,
                        search = 'grid',
                        classProbs = TRUE)

tunegrid <- expand.grid(.mtry = c(5,10))
modellist <- list()

# add one more element to ntree and node size - tuning
start<-Sys.time()
for (ntree in c(20,50,150)){
  for (nodesize in c(3,5,7)){
    set.seed(123)
    fit <- train(make.names(converted_in_7days)~.,
                 data = training,
                 method = 'rf',
                 metric = 'ROC',
                 tuneGrid = tunegrid,
                 trControl = control,
                 ntree = ntree,
                 nodesize = nodesize)
    key <- toString(ntree+nodesize)
    modellist[[key]] <- fit
  }
}

#check for specificity
time_end<-Sys.time()

#Results for model
results <- resamples(modellist)
summary(results)
#confusion Matrix for the training dataset
confusionMatrix(fit)
#Variable Importance
varImp(fit)

#Confusion Matrix 
forest_probabilities<-predict(fit,newdata=testing,type="prob") 
forest_classification<-rep("1",212798)
forest_classification[forest_probabilities[,2]<0.5]="0" 
forest_classification<-as.factor(forest_classification)
confusionMatrix(forest_classification,testing$converted_in_7days, positive="1") 

#ROC Curve
forest_ROC_prediction <- prediction(forest_probabilities[,2], testing$converted_in_7days) 
forest_ROC <- performance(forest_ROC_prediction,"tpr","fpr")
plot(forest_ROC) 

#AUC Calculation and plotting LIFT curve
AUC.tmp <- performance(forest_ROC_prediction,"auc")
forest_AUC <- as.numeric(AUC.tmp@y.values)
forest_AUC
plotLift(forest_probabilities[,2],  testing$converted_in_7days, cumulative = TRUE, n.buckets = 10) 
