# LoanDefaulterClassification_MehulKapoor

# Loan Default Prediction

## 1. Introduction

### Overview:
This project focuses on building a data mining project that would help the financial institution that provide loans to predict potential loan defaulters. I have gathered data from Kaggle and used machine learning models to predict the outcomes for the classification type data. I've followed the entire data mining process, which involved collecting, cleaning, reducing, training, and testing data, followed by evaluating the models.

### Problem Statement:
Financial institutions provide loan services to individuals and companies, they earn profits by charging interests from them in return. These financial institutions face risk when the individuals are unable to pay the loan. This project leverages data mining techniques to identify the potential loan defaulters.

## 2. Methodology

### Data Collection:
- For this Project I have gathered dataset from Kaggle.
- Dataset Source: [Kaggle - Loan Default Dataset](https://www.kaggle.com/datasets/nikhil1e9/loan-default/data)

#### Data Understanding:
- The dataset contains 255,347 rows and 18 columns in total.

| Sr | Column Name     | Data Type | Description                                        |
|----|-----------------|-----------|----------------------------------------------------|
| 0  | LoanID          | string    | A unique identifier for each loan.                 |
| 1  | Age             | integer   | The age of the borrower.                          |
| 2  | Income          | integer   | The annual income of the borrower.                 |
| 3  | LoanAmount      | integer   | The amount of money being borrowed.                |
| 4  | CreditScore     | integer   | The credit score of the borrower, indicating their creditworthiness. |
| 5  | MonthsEmployed  | integer   | The number of months the borrower has been employed. |
| 6  | NumCreditLines  | integer   | The number of credit lines the borrower has open.  |
| 7  | InterestRate    | float     | The interest rate for the loan.                   |
| 8  | LoanTerm        | integer   | The term length of the loan in months.             |
| 9  | DTIRatio        | float     | The Debt-to-Income ratio, indicating the borrower's debt compared to their income. |
| 10 | Education       | string    | The highest level of education attained by the borrower (PhD, Master's, Bachelor's, High School). |
| 11 | EmploymentType  | string    | The type of employment status of the borrower (Full-time, Part-time, Self-employed, Unemployed). |
| 12 | MaritalStatus   | string    | The marital status of the borrower (Single, Married, Divorced). |
| 13 | HasMortgage     | string    | Whether the borrower has a mortgage (Yes or No).   |
| 14 | HasDependents   | string    | Whether the borrower has dependents (Yes or No).  |
| 15 | LoanPurpose     | string    | The purpose of the loan (Home, Auto, Education, Business, Other). |
| 16 | HasCoSigner     | string    | Whether the loan has a co-signer (Yes or No).     |
| 17 | Default         | integer   | The binary target variable indicating whether the loan defaulted (1) or not (0). |

#### Data Cleaning:
The dataset was already cleaned. It did not have any Null values or outliers.

#### Data Reduction and Feature Selection:
I have used Feature selection techniques to select the best columns and remove the columns that have the least impact on the target feature/column.
- chi-square test: to select the best categorical variables when the target variable is also categorical.
- ANOVA: to select the best Numerical variables when the target variable is categorical.

In total I have dropped the following columns:
'InterestRate', 'NumCreditLines', 'DTIRatio', 'LoanPurpose', 'MaritalStatus

#### Data Imbalance:
We deal with data imbalance to reduce the chances of creating a biased outcome and prevent model from being biased towards the majority class. I used the following model to deal with target variable imbalance:

SMOTEENN: Combine over- sampling of the minority class and under-sampling of the majority class using SMOTE and Edited Nearest Neighbours.

Before Re-Sampling:

**Figure 1: Before resampling**
- Non-Defaulters: 225694
- Defaulters: 29653

After Re-Sampling:

**Figure 2: After resampling**
- Non-Defaulters: 114877
- Defaulters: 108301

### Model Building:
The data is randomly split into training and testing data into 80:20 ratio using the train_test_split() function from the scikit-learn library.

The training data is then trained with help of following Machine learning Models:

- Random Forest:
  A random forest is an ensemble method that fits a number of decision trees and uses averaging to improve the predictive accuracy of the classification data.
  Used GridSearchCV function to choose the best hyper parameters for the random forest model.

- SVC:
  SVM finds the best hyperplane that can separate two classes in the best possible manner. Support Vector Machine (SVM) is a powerful supervised machine learning algorithm used for classification and regression tasks.
  Used default hyperparameters for this as the hyperparameter tuning was taking hours to execute.

- XGBoost:
  XGBoost, which stands for Extreme Gradient Boosting, is a scalable, distributed gradient-boosted decision tree (GBDT) machine learning library. Just like random forest it uses decision trees to predict the outcome. It boosts the weak decision trees to reduce the loss function in the next decision tree, provides parallel tree boosting and is the leading machine learning library for regression, classification, and ranking problems.

- VotingClassifier (Ensemble of Random Forest and XGboost):
  A Voting Classifier is an ensemble learning method in machine learning which Utilizes multiple models to make predictions.
  Used soft voting hyperparameter to calculate the average probability of each class and then declares the winner having the highest weighted probability.

#### Evaluation Metrics:
We used evaluation metrics the check the performance of the machine learning models.
The following evaluation metrics were used to compare how accurately was the dataset trained, by comparing actual and the predicted test data output:

- Accuracy:
  Accuracy simply measures how often the classifier correctly predicts. We can define accuracy as the ratio of the number of correct predictions and the total number of predictions.

- Precision:
  It explains how many of the correctly predicted cases actually turned out to be positive.

- Recall:
  It explains how many of the actual positive cases we were able to predict correctly with our model.

- F1 Score:
  It gives a combined idea about Precision and Recall metrics. It is maximum when Precision is equal to Recall. It balances the trade-off between recall and precision.


