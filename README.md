# credit-risk-classification

Machine learning techniques were used to train and evaluate a model based on loan risk.

## Overview of the Analysis

The purpose of the analysis is to identify the creditworthiness of borrowers using the historical lending activities from a peer-to-peer lending services company.

Based on the provided financial data the model is able to predict the loan status. The data includes:
- loan_size,
- interest_rate,
- borrower_income,
- debt_to_income, 
- num_of_accounts, 
- derogatory_marks, 
- total_debt.

The dataset incuded 77,536 records in total, with 75,036 healthy loan records (loan status 0) and 2,500 high risk loans (loan status 1). 

The analysis includes the following steps:
1. Splitting dataset into `train` and `test`. 
2. Building a logistic regression model using `LogisticRegression` module from scikit-learn.
3. Fitting the train data set into the model.
4. Using the fitted model to predict loan status on the test dataset.
5. Evaluating model performance by:
    - calculating the accuracy score of the model, 
    - generating a confusion matrix, 
    - printing the classification report.
6. Resampling the training dataset using  `RandomOverSampler` module from imbalanced-learn and rerunning steps 2-5 on a resampled data. 


## Results

* Machine Learning Model 1 (trained on the originally split train dataset):
  * Balanced Accuracy Score: 94%
  * Precision scores:
    - healthy loan (0) 100%,
    - high risk loans (1) 87%.
  * Recall scores:
    - healthy loan (0) 100%,
    - high risk loans (1) 89%.


* Machine Learning Model 2 (trained on resampled dataset):
  * Balanced Accuracy Score: 99%
  * Precision scores:
    - healthy loan (0) 100%,
    - high risk loans (1) 87%.
  * Recall scores:
    - healthy loan (0) 100%,
    - high risk loans (1) 100%.


## Summary

Learning Model 2 achieves better prediction results achieving Balanced Accuracy Score 99%, showing 100% accuracy and increasing the recall score for high risk loans to 100%, meaning that the model captures all high risk loans.

However, if the goal of the model is to ensure that less false positive outcomes for high risk loans is predicted, neither of the models trully serve this purpose, as the precision rate for high risk loans in both models is less than 90%;; meaning that the model is more strict in predicting the high risk status of a loan.