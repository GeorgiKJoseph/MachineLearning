import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

data = pd.read_csv("car.data")
print(data.head())

le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))

predict = "class"

X = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(cls)

xtrain, xtest, ytrain, ytest = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

model = KNeighborsClassifier(n_neighbors=9)
model.fit(xtrain, ytrain)
acc = model.score(xtest, ytest)
print(acc)

predict = model.predict(xtest)
names = ["unacc", "acc", "good", "vgood"]
for i in range(len(predict)):
    print("{:<{}} {:<{}}".format('Pred: ' + names[predict[i]], 12, 'Act: '+names[ytest[i]], 12))
    #print("Pred: ", names[predict[i]],"Act: ", names[ytest[i]])
