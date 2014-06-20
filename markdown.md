Machine learning algorithm to predict activity quality 
========================================================
## Executive Summary 
I used Random Forest method for prediction algorithm model to predict 20 different test cases with 95% accuracy. It is also very accuracy comparing with other methods such as Trees with 54%.   

## Build the model
After trying some methods for predicting, I chose to use the Ramdon Forest that is the best accuracy method for predictive algorithm in this case. 

## Exploratory data analysis
Loading the training data set

```r
data = read.csv("pml-training.csv", na.strings = c("NA", ""))
```


Remove NA columns

```r
dt <- data[, which(apply(data, 2, function(x) {
    sum(is.na(x))
}) == 0)]
```


Cross validation: split the training set

```r
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
inTrain <- createDataPartition(y = dt$classe, p = 0.1, list = FALSE)
training <- dt[inTrain, ]
testing <- dt[-inTrain, ]
```


Delete the unuseful predictors

```r
mydt <- training[, -grep("timestamp|X|user_name|new_window", names(training))]
mydt1 <- testing[, -grep("timestamp|X|user_name|new_window", names(testing))]
```

 
## Expect the out of sample error
Train the model

```r
modFit <- train(classe ~ ., data = mydt, method = "rf")
```

```
## Loading required package: randomForest
## randomForest 4.6-7
## Type rfNews() to see new features/changes/bug fixes.
```

```r
modFit
```

```
## Random Forest 
## 
## 1964 samples
##   53 predictors
##    5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## 
## Summary of sample sizes: 1964, 1964, 1964, 1964, 1964, 1964, ... 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy  Kappa  Accuracy SD  Kappa SD
##   2     0.9       0.9    0.01         0.01    
##   30    0.9       0.9    0.01         0.01    
##   50    0.9       0.9    0.01         0.02    
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 27.
```


The outcome is really good with 93% accuracy on train set while outcome of Trees method in this case is poor performent of 54% accuracy.

## Estimate the error appropriately with cross-validation
Test the model

```r
pre <- predict(modFit, mydt1)
confusionMatrix(pre, mydt1$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 5018  135    0    1    0
##          B    3 3169  138    2   60
##          C    0  104 2925   85   25
##          D    0    9   16 2803   39
##          E    1    0    0    3 3122
## 
## Overall Statistics
##                                         
##                Accuracy : 0.965         
##                  95% CI : (0.962, 0.968)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.955         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.999    0.927    0.950    0.969    0.962
## Specificity             0.989    0.986    0.985    0.996    1.000
## Pos Pred Value          0.974    0.940    0.932    0.978    0.999
## Neg Pred Value          1.000    0.983    0.989    0.994    0.991
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.284    0.179    0.166    0.159    0.177
## Detection Prevalence    0.292    0.191    0.178    0.162    0.177
## Balanced Accuracy       0.994    0.957    0.968    0.982    0.981
```


The outcome with cross-validation is even better with 96.98% accuracy. 

## Use model to predict 20 different test cases
Process the test set

```r
data1 = read.csv("pml-testing.csv", na.strings = c("NA", ""))
dt1 <- data1[, which(apply(data1, 2, function(x) {
    sum(is.na(x))
}) == 0)]
test <- dt1[, -grep("timestamp|X|user_name|new_window", names(dt1))]
```


Predict new values

```r
predict(modFit, newdata = test)
```

```
##  [1] B A B A A E D D A A B C B A E E A B B B
## Levels: A B C D E
```


After submitting 20 test cases for marking, I got 19/20 points (95% accuracy as same as expectation of the out of sample error) 

## Problems on model approach
- Random forests are difficult to interpret but often very accurate
- Care should be taken to avoid overfitting
- Very low





