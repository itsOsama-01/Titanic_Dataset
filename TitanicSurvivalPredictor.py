import pandas as pd

#importing dataset
data = pd.read_csv('titanic3.csv')

#Cleaning Data
data = data.drop(["name","ticket","cabin","boat","body"],axis=1)

cols = ["sex","embarked","home.dest"]
for col in cols:
    data[col].fillna("NA",inplace=True)

columns = ["pclass","survived","age","sibsp","parch","fare"]

for column in columns:
    data[column].fillna(data[column].median(),inplace=True)

#Encoding columns with non-integer values
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
for col in cols:
    data[col] = le.fit_transform(data[col])

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#Splitting Data
y=data.values[:,1]
x=data.values[:, data.columns != "survived"]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=10)

#Applying Logistic Regression
Regressor = LogisticRegression(max_iter=400)
Regressor.fit(x_train,y_train)

#Applying Decision Tree algorithm
Classifier = DecisionTreeClassifier(criterion="gini")
Classifier.fit(x_train,y_train)

#Calculating accuracy of the above models
from sklearn.metrics import accuracy_score

y_predictC = Classifier.predict(x_test)
y_predictR = Regressor.predict(x_test)


print("Accuracy of the Decision Tree is: ",accuracy_score(y_test,y_predictC))
print("Accuracy of the Logistic Regression model is: ",accuracy_score(y_test,y_predictR))