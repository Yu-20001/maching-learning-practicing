# LinearRegression_02

code from:https://www.youtube.com/watch?v=3AQ_74xrch8&list=PLzMcBGfZo4-mP7qA9cagf68V06sko5otr&index=4

由LinearRegression 01改動，增加自動儲存最準確的模型及畫圖的功能，以下嘗試分析新的程式碼。

```
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
```
```
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style
```
新import的函式庫，pyplot、style用來畫圖，pickle用來存model。
```
data = pd.read_csv("student-mat.csv",sep=";")
data = data[["G1", "G2", "G3"]]

predict = "G3"

x = np.array(data.drop(columns=predict))
y = np.array(data[predict])
```
```
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
```
在迴圈外重新定義一組測試集、訓練集，以供訓練好(註解掉後)的模型使用。
```
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
```
以linear_model的.score得到的準確度為比較標準，執行50次for迴圈，產生50次linear_model，並把最佳準確度的model用pickel儲存下來。
>  with open("grademodel.pickle","wb") as f:
            pickle.dump(linear, f)
        best = acc

直接在.py存放的資料夾中新增一grademodel.pickel檔案(準確度最佳的模型)，pickle.dump()儲存模型，把當前的linear模型存成grademodel.pickel。
在執行一次後，我們即可得到一個經過50次訓練中最佳準確率的模型的pickle檔案，於是不需要再訓練模型，所以將這段註解掉。
```
pickle_in = open("grademodel.pickle","rb")
linear = pickle.load(pickle_in)
```
pickel.load()載入剛剛訓練好的模型
```
print("Coefficient:",linear.coef_)
print("Intercept",linear.intercept_)
predictions = linear.predict(x_test)
for x in range(len(predictions)):
    print(predictions[x],x_test[x],y_test[x])
```
```

style.use("ggplot")
```
style.use()用來設定圖面的style，例如顏色等，
```
p="G1"
pyplot.scatter(data[p],data["G3"])
```
pyplot.scatter(,)用來畫散布圖，括號的前方為x軸值的意義，後為y軸值的意義，
```
pyplot.xlabel(p)
pyplot.ylabel("final grade")
```
標示x、y坐標軸的名稱
```
pyplot.show()
```
展示產生的圖
