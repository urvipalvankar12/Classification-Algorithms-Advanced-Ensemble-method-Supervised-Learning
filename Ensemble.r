if("pacman" %in% rownames(installed.packages()) == FALSE) {install.packages("pacman")} 
pacman::p_load("mlbench","pROC","dplyr","caret","ROCR","lift","glmnet","MASS","e1071","mice","gdata","MLmetrics","caretEnsemble","gbm")

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
set.seed(111)
inTrain <- createDataPartition(y = eureka_fixed$converted_in_7days,
                               p = 0.7, list = FALSE)


eureka_matrix<-data.matrix(dplyr::select(eureka_fixed, -converted_in_7days))
dim(eureka_matrix)

X_train <- eureka_matrix[ inTrain,]
x_test <- eureka_matrix[ -inTrain,]


#splitting for label columns
training<-eureka_fixed[inTrain,]
testing<-eureka_fixed[-inTrain,]

y_train <-training$converted_in_7days
y_test <-testing$converted_in_7days

#Tuning Hyperparameter
set.seed(123)
grid.xgboost <- expand.grid(.nrounds = 100,
                            .eta = 0.1,                
                            .gamma = 0,
                            .max_depth = 5,
                            .colsample_bytree = 1,                
                            .subsample = 1, 
                            .min_child_weight = 1)

grid.rf <- expand.grid(.mtry = 3)

#Control function
ctrl <- trainControl(method="repeatedcv",
                     number=10,
                     repeats=5,
                     savePredictions = "final",
                     classProbs = TRUE,
                     verboseIter = TRUE,
                     summaryFunction = twoClassSummary,
                     sampling="smote")


#RF and XGBOOST
set.seed(222)
model_list <- caretList(X_train,
                        make.names(y_train),
                        trControl = ctrl,
                        tuneList = list(
                          xgbTree = caretModelSpec(method="xgbTree",
                                                   tuneGrid = grid.xgboost),
                          rf = caretModelSpec(method = "rf",
                                              tuneGrid = grid.rf)),
                          metric="ROC")

output = resamples(model_list)
summary(output)

#predicting probabilities using RF and XGBOOST
library("caTools")
model_preds <- lapply(model_list, predict, newdata=x_test, type="prob")
model_preds <- lapply(model_preds, function(x) x[,1])
model_preds <- data.frame(model_preds)

#Ensemble Method - Stacking
glm_ensemble <- caretStack(
  model_list,
  method="glm",
  metric="ROC",
  trControl=trainControl(
    method="boot",
    number=10,
    savePredictions="final",
    classProbs=TRUE,
    summaryFunction=twoClassSummary
  )
)

#Predicting Probablities using Ensemble and then adding it to the same dataframe
model_preds2 <- model_preds
model_preds2$ensemble <- predict(glm_ensemble, newdata=x_test, type="prob")

#AUC calc
colAUC(model_preds2, y_test)

#Confusion Matrix for ensemble method
ensemble_classification<-rep("1",212796)
ensemble_classification[model_preds2$ensemble>0.995]="0"
ensemble_classification<-as.factor(ensemble_classification)
caret::confusionMatrix(ensemble_classification, y_test, positive="1")
