if("pacman" %in% rownames(installed.packages()) == FALSE) {install.packages("pacman")} 
pacman::p_load("caret","ROCR","lift","glmnet","MASS","e1071","mice","gdata","MLmetrics","dplyr")

library(MLmetrics)
library(pROC)
library(dplyr)
library(caret)
library(mice)
library(gdata)
library(dplyr)

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
#sampling data
#Creating Training and Testing set
set.seed(98777)

inTrain <- createDataPartition(y = eureka_fixed$converted_in_7days,
                               p = 496530/709327, list = FALSE)


eureka_matrix<-data.matrix(select(eureka_fixed, -converted_in_7days))
dim(eureka_matrix)

x_train <- eureka_matrix[ inTrain,]
x_test <- eureka_matrix[ -inTrain,]


#splitting for label columns
training<-eureka_fixed[inTrain,]
testing<-eureka_fixed[-inTrain,]

y_train <-training$converted_in_7days
y_test <-testing$converted_in_7days

table(testing$converted_in_7days)
#hyper parameter tuning
xgb_grid = expand.grid(
  nrounds = c(100, 200, 300), 
  max_depth = c(2, 5, 10), 
  eta = c(0.0025, 0.05,0.1), 
  gamma = 0, 
  colsample_bytree = 1, 
  min_child_weight = 1,
  subsample = 1
)

# pack the training control parameters
start_time<-Sys.time()
xgb_trcontrol = trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 5,
  search = 'grid',
  classProbs = TRUE,
  sampling = "smote",
  summaryFunction = twoClassSummary
)

#Building xgboost model
model_XGboost = train(
  x = x_train,
  y = make.names(y_train),
  trControl = xgb_trcontrol,
  tuneGrid = xgb_grid,
  method = "xgbTree",
  metric = "ROC",
)

endtime<-Sys.time()
#confusion Matrix - Training Dataset
confusionMatrix(model_XGboost)
#Model Results
model_XGboost$results
#Variable Importance Plot
varImp(model_XGboost)
#Best Tuned model out of the cross validation
model_XGboost$bestTune

#predicting on testing set
XGboost_prediction<-predict(model_XGboost,newdata=x_test, type="prob")
xgb_classification<-rep("1",212796)
xgb_classification[XGboost_prediction[,2]<0.4]="0" 
xgb_classification<-as.factor(xgb_classification)
confusionMatrix(xgb_classification,y_test, positive="1") 

xgb_ROC_prediction <- prediction(XGboost_prediction[,2], y_test) 
xgb_ROC <- performance(xgb_ROC_prediction,"tpr","fpr")
plot(xgb_ROC) 

#AUC Calculation and plotting LIFT curve
AUC.tmp <- performance(xgb_ROC_prediction,"auc")
xgb_AUC <- as.numeric(AUC.tmp@y.values)
xgb_AUC
plotLift(XGboost_prediction[,2],  testing$converted_in_7days, cumulative = TRUE, n.buckets = 10) 

#F1 Score calculation
f1_score<-F1_Score(y_test,xgb_classification,positive=NULL)

#Recall Score Calculation
recall_score<-Recall(y_test,xgb_classification,positive=NULL)