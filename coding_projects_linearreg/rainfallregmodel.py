#importing panda (data manipulation), numpy (vectorization, indexing and mathematical functions),
# matplot (visualization software), seaborn (statistical visualization) and scilearnkit (machine learning)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#Importing the chi2 function from scikit learn to perform chi squared test
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

#Importing the metrics function from scikit learn to evaluate the model
from sklearn import metrics
from sklearn.metrics import classification_report


#loading the dataset
df= pd.read_csv('rainfall.csv')
#reads first 5 heads of the dataset
print(df.head(5))


#makes a dataframe from only the months so we can work with the data easily
df = pd.read_csv('rainfall.csv',usecols=["sep","oct","nov","dec","jan","feb","mar","apr","may"])
#print(df)

#makes a column in the dataframe that sums all the rainfall in the months
df["total_rainfall"]=df["sep"]+df["oct"]+df["nov"]+df["dec"]+df["jan"]+df["feb"]+df["mar"]+df["apr"]+df["may"]

#If the rainfall is less than 500, we get 0 which will represent no flooding. Otherwise, 1 which is flooding
def evalrain (x):
    if x < 500:
        return 0
    else:
        return 1

#Applys function to each entry of the total rainfall column, then adds it to a new column called flooding
result = df ["total_rainfall"].apply(evalrain)
df["flooding"]=result
#print(df.head(5))
#could drop total rainfall here



#adding the names of the areas back into the data set
df2= pd.read_csv('rainfall.csv',usecols=["name"])

df["name"]=df2["name"]
print(df.head(5))

#a chi square test and select the training features we want to train the model on

#defines X and Y of chi squared test
X = df.iloc[:,1:9]
Y = df.iloc[:,-1]

#applys chi squared test to the data and selects for the top 3 features
best_features = SelectKBest(score_func=chi2, k=3)
fit = best_features.fit(X,Y)

#Creates dataframe objects for the scores of each feature and also a dataframe for the features
df_scores= pd.DataFrame(fit.scores_)
df_columns= pd.DataFrame(X.columns)

#This merges the dataframe objects into one data frame and then sorts them by how well the feature scored
features_scores= pd.concat([df_columns, df_scores], axis=1)
features_scores.columns= ['Features', 'Score']
features_scores.sort_values(by = 'Score')
   
#Prints the features and their scores in descending order
print(features_scores.sort_values(by = 'Score'))


###Building regressional model 

#defines X and Y of the model

#Top 3 features
X = df[["feb","dec","jan"]]
#Target output
Y = df[["flooding"]]


#Splitting dataset into a training and testing set

X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.4,random_state=100)


#creating a logical regression body

logreg= LogisticRegression()
#fit the regression body with the training data
logreg.fit(X_train,y_train)


#predict the likelihood of rainfall occuring using the fitted logistic regression model

y_pred=logreg.predict(X_test)
print (X_test) #test dataset
print (y_pred) #predicted values



#Evaluating the model using a classification report and a ROC curve:

#evaluates the accuracy of the model using the test data
print("Accuracy: ",metrics.accuracy_score(y_test, y_pred))

#evaluates the recall score of the model using the test data
print("Recall: ",metrics.recall_score(y_test, y_pred, zero_division=1))

#evaluates the precision score of the model 
print("Precision:",metrics.precision_score(y_test, y_pred, zero_division=1))

#prints out the classification report of the model
print("CL Report:",metrics.classification_report(y_test, y_pred, zero_division=1))


#Plotting an ROC curve


#defining metrics of the ROC curve
#takes log regression body and the test data to predict the probabilities of the test data
y_pred_proba= logreg.predict_proba(X_test) [::,1]

#calculates the false positive and true positive rate of the model using the probabilities of the test data
false_positive_rate, true_positive_rate, _ = metrics.roc_curve(y_test, y_pred_proba)

#Calculating the area under the curve to evaluate how well the model performed
auc= metrics.roc_auc_score(y_test, y_pred_proba)

#Plotting the curve
plt.plot(false_positive_rate, true_positive_rate,label="AUC="+str(auc))

#titling the graph, x and y axis
plt.title('ROC Curve')
plt.ylabel('True Positive Rate')
plt.xlabel('false Positive Rate')
plt.legend(loc=4)

#showing the graph

plt.show()