## -----------------------------------------------------------------------------------------------------------------------------------
## Title: wineQualityWithSVM class
## Author: Begüm Şara Ünal
## Description: This class estimates the wine quality using the Linear Support Vector Machine (SVM) algorithm and 
## recommends wine to the user from the data set using the SVM algorithm based on the values entered by the user. 
## The class uses the red wine quality data from kaggle. (Kaggle link is below.) Also, there are graphs for various purposes in this class.
## The link of the dataset taken from Kaggle: https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009?resource=download
## ------------------------------------------------------------------------------------------------------------------------------------

#import libraries to use 
import pandas as pd #to read data
import seaborn as sns #to show charts etc.
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

##---------------------------------------------------------------------------------------------------
## Summary: Loads dataset from csv document. Then, it is assigned to a DataFrame with the name wine.
##---------------------------------------------------------------------------------------------------
wine = pd.read_csv('winequality-red.csv')

##-------------------------------------------------
## Summary: Print the initial rows and check data
##-------------------------------------------------
print(wine.head())

##----------------------------------------------------
## Summary: Print information about the data columns
##---------------------------------------------------
print(wine.info())


##-------------------------------------------------
## Summary: Print and check the size of data
##-------------------------------------------------
print(wine.shape)


##-------------------------------------------------------------------------------------------------
## Summary: Creates and displays a graph where the 'quality' of all data is different in color.
##-------------------------------------------------------------------------------------------------
sns.pairplot(wine, hue='quality')
plt.show()

##-------------------------------------------------------------------------------------------------
## Summary: #Seperates indepentent variables (inputs) and target variables (outputs). 
## The x variable is created by leaving the "quality" column from the DataFrame with the arguments. 
## The variable y is set as the "quality" column.
##-------------------------------------------------------------------------------------------------
x = wine.drop("quality", axis =1)
y = wine["quality"]

##---------------------------------------------------------------------------------------------------
## Summary: Using the train_test_split() function, the dataset is split into training and test sets
##---------------------------------------------------------------------------------------------------
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

## ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
## test_size --> The test_size parameter determines how much of the data set will be reserved as the test set. 
## In this case, 20% of the data set is reserved as the test set. In general, the size of the test set can be changed, 
## depending on the size of the data set.
## ----------------------------------------------------------------------------------------------------------------------
## random_state --> The random_state parameter is a seed value used during random splitting of the dataset. 
## This way, the random split of the dataset remains the same on each run. 
## So random_state=42 is a seed value chosen to get the same random split on each run.
## ----------------------------------------------------------------------------------------------------------------------
## The reasons for determining these parameters are to ensure the repeatability of the model 
## and to create an initial set of tests. Thus, you can obtain consistent results when evaluating 
## the performance of the model. The random_state value can be arbitrarily changed, but the advantage of using 
## a specific value is that the same random split can be repeated.
## /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

##--------------------------------------------------------------------------------------------------------------------------
## Summary: This part performs the data scaling. Scaling features ensures that each feature is at the same scale. 
## Most machine learning algorithms want features to be at the same scale. 
## Therefore, scaling features is one of the data processing steps.
##-------------------------------------------------------------------------------------------------------------------------

## /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
## `StandardScaler` class --> We scale features using the `StandardScaler` class. `StandardScaler` is a commonly used class
## for scaling features. The scaling operation converts the mean of each feature to 0 and its standard deviation to 1.
## /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
scaler = StandardScaler()
#The training process calculates the mean and standard deviation of each feature.
x_train = scaler.fit_transform(x_train)  #We train (fit) the 'StandardScaler' using the `X_train` dataset (This operation returns the scaled version of the properties.)
                                         
x_test = scaler.transform(x_test)        #We scale both the training and test datasets using the `transform` method.
                                         #After this process, the `X_train` and `X_test` datasets are ready for use with their features at scale. 
                                         
#*** This helps the model work better without being affected by the different scales of the features and allows the algorithm to observe all features with equal weight.***


##---------------------------------------------------
## Summary: To use svm import necessary libraries
##--------------------------------------------------- 
from sklearn import svm
##---------------------------------------------------------------------------------------------------
## Summary: Creates model with using KNeighborsClassifier()
##---------------------------------------------------------------------------------------------------
clf = svm.SVC(gamma='auto',kernel='linear', C=1.0) #C is default parameter and best choice by experience is 1.0

##---------------------------------------------------------------------------------------------------
## Summary: Trains the model with using fit() method and fit features (x) and labels (y)
##---------------------------------------------------------------------------------------------------
clf.fit(x_train,y_train)

##---------------------------------------------------------------------------------------------------
## Summary: The SVM model makes predictions on test data using the predict() function.
##---------------------------------------------------------------------------------------------------
clf.predict(x_test)

##---------------------------------------------------------------------------------------------------
## Summary: The success of the model is calculated using the score() function and 
## printed to the screen with print().
##---------------------------------------------------------------------------------------------------
print(f'Model Accuracy With Using Linear Support Vector Machine (SVM) : {clf.score(x_train,y_train):.3f}')

##---------------------------------------------------------------------------------------------------------------
## Summary: Makes a graph of datas , gets coefficients with coef_ ,and print with using print() 
###-------------------------------------------------------------------------------------------------------------
w = clf.coef_[0]
print(w)

#a is a learning rate 
a = -w[0] / w[1]

#xx is our line
xx = np.linspace(-3,6) #min and max features values 
yy = a * xx - clf.intercept_[0] / w[1]

#plot line
h0 = plt.plot(xx,yy, 'k',label = "non-weighted divide") #'k-' is for black line 

plt.scatter(x_train[:,0],x_train[:,1],c=y_train)
# x[:,0] is first element, x[:,1] is first if element of the second element ,and c=y to color different plot differently
plt.legend()
plt.show() #shows the graph 

import numpy as np
##---------------------------------------------------------------------------------------------------
## Summary: Using the SVM model, the index of the data closest to the user's input is found 
## and the suggested data is printed.
##---------------------------------------------------------------------------------------------------
fixed_acidity = float(input("Fixed Acidity: "))
volatile_acidity = float(input("Volatile Acidity:"))
citric_acid = float(input("Citric Acid :"))
residual_sugar = int(float(input("Residual Sugar")))
chlorides = float(input("Chlorides :" ))
free_sulfur_dioxide = float(input("Free Sulphur Dioxide: "))
total_sulfur_dioxide = float(input("Total Sulfur Dioxide: "))
density = float(input("Density: "))
pH = float(input("PH Value: "))
sulphates = float(input("Sulphates: "))
alcohol = float(input("Alcohol: "))

##---------------------------------------------------------------------------------------------------
## Summary: Creates an array with user inputs
##---------------------------------------------------------------------------------------------------
user_input = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                        chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
                        density, pH, sulphates, alcohol]])


##---------------------------------------------------------------------------------------------------
## Summary: Scale features 
##---------------------------------------------------------------------------------------------------
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
scaled_input = scaler.transform(user_input)

##---------------------------------------------------------------------------------------------------
## Summary: The SVM model is created using the SVC() class and trained with the fit() function.
##---------------------------------------------------------------------------------------------------
svm = svm.SVC()
svm.fit(x_scaled, y)
##---------------------------------------------------------------------------------------------------
## Summary: Predict the user's input
##---------------------------------------------------------------------------------------------------
predicted_label = svm.predict(scaled_input)

##---------------------------------------------------------------------------------------------------
## Summary: Find the data closest to the predicted label
##---------------------------------------------------------------------------------------------------
recommended_index = np.where(y.values == predicted_label)[0][0]

##---------------------------------------------------------------------------------------------------
## Summary: Print the suggested data
##---------------------------------------------------------------------------------------------------
#print the suggested data
recommended_data = wine.iloc[recommended_index]
print("Suggested Data on Linear Support Vector Machine (SVM) : ")
print(recommended_data)

## ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
## iloc --> iloc is an indexing method used in Pandas library to access data in dataframes (DataFrame). 
## The abbreviation "iloc" stands for "integer location".
## ----------------------------------------------------------------------------------------------------------------------
## ----------------------------------------------------------------------------------------------------------------------
## The statement data.iloc[recommended_index] selects the row with the index recommended_index in the data dataframe. 
## Returns a Pandas Series representing this row containing all columns starting with the first column. 
## That is, recommended_data is a Series object that represents a particular row in the dataframe.
## ----------------------------------------------------------------------------------------------------------------------
## ----------------------------------------------------------------------------------------------------------------------
## In this case, the recommended_data variable represents a Series object containing the nearest neighbor's data 
## estimated by the KNN algorithm.
## ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////





