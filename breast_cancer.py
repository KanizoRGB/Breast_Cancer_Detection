import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
cancer_data = load_breast_cancer() 
#The object returned( which we stored in the cancer_data variable) is an object similar to a Python dictionary

print(cancer_data.keys())
#Use the command above to see the  keys

print(cancer_data['DESCR'])
 #The dataset contains 30 features, and two clases i.e (malignant/cancerous) and (Benign/Not cancerous)

#In the dataset, some features are calculated based on other columns. The process of figuring out what other additional features to calculate is called feature engineering

cancer_data['data'] 
# #pull the feature data out of the cancer object. It is a numpy array with shape(569,30)

print(cancer_data['feature_names']) 
# #pull out the feature names. They are stored with the 'feature_names' key

df = pd.DataFrame(cancer_data['data'],columns=cancer_data['feature_names']) 
#We put the data in a dataframe using pandas and make it more human readable by adding the column names

df['target'] = cancer_data['target']
#Put the target data into our data frame which can be found with the 'target' key from our object
#This increases the number of columns from 30 to 31

print(cancer_data['target_names'])
#Determine what the target data means
#It gives an array ['malignant' 'benign'] meaning 0 is for malignant(cancerous) and 1 is benign(no cancer)


X = df[cancer_data.feature_names].values
y= df['target'].values
#Build a features matrix and a target arrary so that we can build a Logistic regression model



model = LogisticRegression(solver='liblinear')
model.fit(X,y)
#Create a logistic regression object and use the fit method to build the model

print("prediction for datapoint 0:",model.predict([X[0]]))
#Check what the model predicts for the first data point in our dataset. Recall that predict takes a 2D array so we must put the data point in a list


print(model.score(X,y))
#Check how your model performs over the whole dataset

# print(df.head())