# Cars CO2 Emission Estimator

## Modeling the Regression relationship between Cars and their CO2 emission

&nbsp; &nbsp; &nbsp; &nbsp;

The contents of this project are divided into following topics which are listed as follows:- 




## Table of Contents


1.	Introduction
2.	License information
3.	Python libraries
4.	The problem statement
5.	Linear Regression
6.	About the dataset
7.	Exploratory Data Analysis
8.  Statistical Data Analysis
9.  Cleaning Data
10.	Mechanics of Linear Regression
11.	Hyperparameter Tuning
12.	Regression metrics for model performance
i.	    RMSE
ii.	    R2 Score
13.	Interpretation and Conclusion
14.	Residual Analysis
15.	Checking for Overfitting or Underfitting the data
16.	Linear Regression - Model Assumptions
17.	References

&nbsp; &nbsp; &nbsp; &nbsp;

## 1.	Introduction

In this project, I build a Ridge Regression model to study the linear relationship between Cars and their various parameters and CO2 Emissions for them. I study the varios parameter and their rellationship with our target variable (CO2 Emission) and implementation a Ridge Regression model to predict the same in Python programming language using Scikit-learn. Scikit-learn is the popular machine learning library of Python programming language. 


&nbsp; &nbsp; &nbsp; &nbsp;

## 2.	License information

The work done in this Jupyter notebook is made available under the Creative Commons Zero v1.0 Universal  License 

You are free to:

•	Share—copy and redistribute the material in any medium or format
•	Adapt—remix, transform, and build upon the material.

I have licensed this Jupyter notebook for general public. The work done in this project is for learning and demonstration purposes. 
  
&nbsp; &nbsp; &nbsp; &nbsp;


## 3.	Python libraries

I have Anaconda Python distribution installed on my system. It comes with most of the standard Python libraries I need for this project. The basic Python libraries used in this project are:-

 •	Numpy – It provides a fast numerical array structure and operating functions.
 
 •	pandas – It provides tools for data storage, manipulation and analysis tasks.
 
 •	Scikit-Learn – The required machine learning library in Python.
 
 •  Matplotlib – It is the basic plotting library in Python. It provides tools for making plots. 
 
 •  Seaborn – It is the plotting library in Python. It provides clean and user friendly tools for making plots. 
 
 •  Scipy – It provides scientific tools for calculations. 
 
 •  Statsmodels – It  provides statistical tools for calculations, analysis and modelling. 
 
 •  Pickle – It serializes the Model so that it need not be trained every time. 


&nbsp; &nbsp; &nbsp; &nbsp;

## 4.	The problem statement

The aim of building a machine learning model is to solve a problem and to define a metric to measure model performance. So, first of all I have to define the problem to be solved in this project.
As described earlier, the problem is to study the linear relationship between car parameters including but not limited to manufactuirng company, fuel consumption, etc and CO2 Emission by cars and predict the tailpipe emissions of carbon dioxide (in grams per kilometre) for combined city and highway driving. I have used two performance metrics RMSE (Root Mean Square Value) and R2 Score value to compute our model performance.


&nbsp; &nbsp; &nbsp; &nbsp;

## 5.	Linear Regression

Linear Regression is a statistical technique which is used to find the linear relationship between dependent and one or more independent variables. This technique is applicable for Supervised Learning Regression problems where we try to predict a continuous variable.
Linear Regression can be further classified into two types – Simple and Multiple Linear Regression. In this project, I employ Ridge Regression technique thus exploiting the mlticollinearity present in this dataset


&nbsp; &nbsp; &nbsp; &nbsp;

## 6.	About the Dataset

The data set has been imported from the Natural Resources Canada website with the following url-
x
https://open.canada.ca/data/en/dataset/98f1a129-f628-4ce4-b24d-6f16bf24dd64#wb-auto-6

Datasets provide model-specific fuel consumption ratings and estimated carbon dioxide emissions for new light-duty vehicles for retail sale in Canada.

Note that these datasets were cleaned and then concatenated to a new dataset 'ready_to_use_data.csv' in the attached python script.

&nbsp; &nbsp; &nbsp; &nbsp;


## 7.	Exploratory Data Analysis

First, I import the datasets of fuel consumption ratings and estimated carbon dioxide emissions for new light-duty vehicles for various years into dataframes with the standard read_csv () function of pandas library and assign it to  variables. Then, I drop NaN values from each dataset and concatenate them row-wise. 
Next, I conducted exploratory data analysis to get a feel for the data.
I checked the dimensions of dataframe with the shape attribute of the dataframe. I viewed the top rows of the dataframe with the pandas head() method. I viewed the dataframe summary with the pandas info() method and descriptive statistics with the describe() method and hecked the correlation coefficient for numerical variables using the pandas corr() method. 
Once I had an idea of the basic statistical sense of data, I plotted each individual parameter against ur target varible 'co2_emissions': Scatter Plot using the eaborn scatterplot() method for Numerical aparameters and Box Plot and Bar Graphs using the seaborn boxplot() and barplot() methods respectively.

![](Images/Fuel%20Consumption%20(combined)%20vs%20CO2%20Emission.png)

![](Images/Fuel%20Type.png)

&nbsp; &nbsp; &nbsp; &nbsp;


## 8. Statistical Analysis

Once I developed an inutition about various parameters and how their are co-related to the target variable, I calculated Pearson Correlation Coefficeint and F-Statistic to understand the Statistical significance of these variables

&nbsp; &nbsp; &nbsp; &nbsp;


## 9. Cleaning Data

On understanding the data, its visual intuition and statistical significance, I prepared a final dataset 'ready_to_use_data.csv' by ridding unrelated parameters and encoding the Categorical Variables using Dummy Variables.

&nbsp; &nbsp; &nbsp; &nbsp;


## 10.	Mechanics of Linear Regression

The mechanics of Linear Regression model starts with splitting the dataset into two sets – the training set and the test set. We instantiate the regressor RR with hyperparameter(alpha) value 0.1 and fit it on the training set with the fit method. In this step, the model learned the correlations between the training data (x_train, y_train). 
Now the model is ready to make predictions on the test data (x_test). Hence, I predict on the test data using the predict method. 

&nbsp; &nbsp; &nbsp; &nbsp;


## 11.	Hyperparameter Tuning
I have predicted the CO2 Emissions values on test split of the dataset by writing code

    yhat = RR.predict(x_test_pr)

To get the score of prediction model, I write

    RR.score(x_train_pr, y_train)

To find the best hyperparameter value for Regression model, I perform GridSearchCV with some assisstance from RepeatedKFold both from sklearn.model_selection module. 
I found that the best hyperparmter value for alpha is 0.1


&nbsp; &nbsp; &nbsp; &nbsp;


## 12.	Regression Metrics for model performance

Now, it is the time to evaluate model performance. For regression problems, there are two ways to compute the model performance. They are RMSE (Root Mean Square Error) and R-Squared Value. These are explained below:-  

###	i.	RMSE

RMSE is the standard deviation of the residuals. So, RMSE gives us the standard deviation of the unexplained variance by the model. It can be calculated by taking square root of Mean Squared Error.
RMSE is an absolute measure of fit. It gives us how spread the residuals are, given by the standard deviation of the residuals. The more concentrated the data is around the regression line, the lower the residuals and hence lower the standard deviation of residuals. It results in lower values of RMSE. So, lower values of RMSE indicate better fit of data. 

###	ii.	R2 Score

R2 Score is another metric to evaluate performance of a regression model. It is also called coefficient of determination. It gives us an idea of goodness of fit for the linear regression models. It indicates the percentage of variance that is explained by the model. Mathematically, 


R2 Score = Explained Variation/Total Variation


In general, the higher the R2 Score value, the better the model fits the data. Usually, its value ranges from 0 to 1. So, we want its value to be as close to 1. Its value can become negative if our model is wrong.


&nbsp; &nbsp; &nbsp; &nbsp;

## 13.	Interpretation and Conclusion

The RMSE value has been found to be 1.3631. It means the standard deviation for our prediction is 1.3631. So, sometimes we expect the predictions to be off by more than 1.3631 and other times we expect less than 1.3631. So, the model is a good fit to the data. 

In business decisions, the benchmark for the R2 score value is 0.7. It means if R2 score value >= 0.7, then the model is good enough to deploy on unseen data whereas if R2 score value < 0.7, then the model is not good enough to deploy. Our R2 score value has been found to be 0.9995. It means that this model explains 99.5 % of the variance in our dependent variable. So, the R2 score value confirms that the model is a good enough to deploy because it providea good fit to the data.

&nbsp; &nbsp; &nbsp; &nbsp;


## 14.	Residual Analysis

A linear regression model may not represent the data appropriately. The model may be a poor fit to the data. So, we should validate our model by defining and examining residual plots.

The difference between the observed value of the dependent variable (y) and the predicted value (ŷi) is called the residual and is denoted by e. The scatter-plot of these residuals is called residual plot.

If the data points in a residual plot are randomly dispersed around horizontal axis and an approximate zero residual mean, a linear regression model may be appropriate for the data. Otherwise a non-linear model may be more appropriate.

Our model suggests less Residuals and hence a good fit to the data.

&nbsp; &nbsp; &nbsp; &nbsp;


## 15.	Checking for Overfitting and Underfitting

Upon plotting a histogram using the seaborn hisplot() method on the Actual Values of dataset and values oredicted by the model, it appears that the model fits the data well, neither too shallow to be called "Underfit", nor too crisp such as to be considered "Overfit".

![](Images/Predicted%20values%20vs%20Actual%20Values.png)

This is confirmed on plotting a Scatterplot for both the variables.

![](Images/Residual%20Analysis.png)

&nbsp; &nbsp; &nbsp; &nbsp;


## 16.	Linear Regression - Model Assumptions


The Linear Regression Model is based on several assumptions which are listed below:-


i.	  Linear relationship

ii.  	Multivariate normality

iii.	No or little multicollinearity

iv. 	No auto-correlation

v.	  Homoscedasticity


### i.	Linear relationship

The relationship between response and feature variables should be linear. This linear relationship assumption can be tested by plotting a scatter-plot between response and feature variables.


### ii.	Multivariate normality

The linear regression model requires all variables to be multivariate normal. A multivariate normal distribution means a vector in multiple normally distributed variables, where any linear combination of the variables is also normally distributed.


### iii.No or little multicollinearity

It is assumed that there is little or no multicollinearity in the data. Multicollinearity occurs when the features (or independent variables) are highly correlated.


### iv.	No auto-correlation

Also, it is assumed that there is little or no auto-correlation in the data. Autocorrelation occurs when the residual errors are not independent from each other.


### v.	Homoscedasticity

Homoscedasticity describes a situation in which the error term (that is, the noise in the model) is the same across all values of the independent variables. It means the residuals are same across the regression line. It can be checked by looking at scatter plot.

&nbsp; &nbsp; &nbsp; &nbsp;


## 17.	 References


The concepts and ideas in this project have been taken from the following websites and books:-


 i.   Machine learning notes by Andrew Ng
 
 ii.  https://en.wikipedia.org/wiki/Linear_regression
 
 iii. https://scikit-learn.org/stable/
 
 iv.  https://scikit-learn.org/stable/
 
 v.   Python for Machine Learning e-course by IBM on Coursera
