import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pd.read_csv("student-mat.csv",sep=";")
data = data[["G1", "G2", "G3"]]

predict = "G3"

x = np.array(data.drop(columns=predict))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
"""
best = 0
for _ in range(50):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
    
    linear = linear_model.LinearRegression()

    linear.fit(x_train,y_train)
    acc = linear.score(x_test,y_test)
    print(acc)
    if acc > best:
        with open("grademodel.pickle","wb") as f:
            pickle.dump(linear, f)
        best = acc
"""
pickle_in = open("grademodel.pickle","rb")
linear = pickle.load(pickle_in)

print("Coefficient:",linear.coef_)
print("Intercept",linear.intercept_)
predictions = linear.predict(x_test)
for x in range(len(predictions)):
    print(predictions[x],x_test[x],y_test[x])

p="G1"
style.use("ggplot")
pyplot.scatter(data[p],data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("final grade")
pyplot.show()