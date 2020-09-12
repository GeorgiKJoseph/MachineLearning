import pandas as pd
import numpy as np
import sklearn
from matplotlib import style
from sklearn import linear_model
import matplotlib.pyplot as pyplot
import pickle

# Inputing data from student-mat.csv to data
data = pd.read_csv("student-mat.csv", sep=";")

# Taking only 6 attributes
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

# Spliting values to be predicted from main array
# Here G3
predict = "G3"
x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

# Spliting Training data and test data
# Here takes 10% of total data as test data
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y, test_size=0.1)

# Using Linear model
linear = linear_model.LinearRegression()

'''
# Apply linear regression
linear.fit(x_train,y_train)

# Save the MODEL so that we don't have to retrain model everytime
# using pickel
with open("studentModel.pickel", "wb") as f:
    pickle.dump(linear,f)
'''

'''
# Prints the accuracy of result
# Score changes in every run because training dataset selected by
#             sklear.model_selection.train_test_split() is random
acc = linear.score(x_test,y_test)
print("Accuracy: ",round(acc*100,2),"%")
'''


# Get previously trained model from studentModel.pickel
pickelIn = open("studentModel.pickel", "rb")
linear = pickle.load(pickelIn)

# Prints coefficients of the line generated
print("coeff: ", linear.coef_)

# Prints Y intercept of the line
print("interc: ", linear.intercept_)

# Printing predictions of test data
prediction = linear.predict(x_test)
for i in range(len(prediction)):
    print(prediction[i], y_test[i], x_test[i])

# Visualizing by matplotlib
p = 'G1'
style.use("ggplot")
pyplot.scatter(data[p],data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()


