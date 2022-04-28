# KNN
code from: https://www.techwithtim.net/tutorials/machine-learning-python/k-nearest-neighbors-3/

KNN是一種適合用於分類的演算法，在這次的資料中，因為包含非數字的部分，導致無法計算，故我們需要先將其轉為數字，再套入KNN模組，方可做我們想要的預測模型。

以下試著逐行分析程式碼(KNN.py)：
```
import pandas
import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing
```
在這次的程式碼中，除了在linear regression中用的pandas、numpy之外，我們需要import KNN模組，又因為本次的資料集並非可以直接使用，需要預先處理才可套用KNN model，所以需要import preprocessing。
```
data = pandas.read_csv("car.data")
```
利用pandas讀data set
```
le = preprocessing.LabelEncoder()
```
透過觀察data set發現，資料中有非數字的部分，因為我們想使用KNN，而KNN需透過計算各項attribute的差後，算出距離最近的N個點，若attribute不是數字的話，便無法計算，透過LabelEncoder()，即可以把原本string的attribute自動編碼為數字，方便KNN使用。
```
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))
```
data["欄位名稱"]是在data frame中指定特定的一行，之後用把他轉為list型給(也可用np.array()將他轉為numpy array)le.fit_transform()使用，就可以開始轉換，完成後，我們就會得到一整數的numpy array，同時可以具備原本文字資料的意義(e.g. vhigh → 3、 low → 1)。
※因為python中class屬保留字，故class用cls代替。
```
predict = "class"
```
表示我們要預測的是class這個欄位。
```
x = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(cls)
```
透過zip將輸入資料統整為一個list
```
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)
```
透過train_test_split將資料切成測試集與訓練集，設定測試集的比例0.1
```
model = KNeighborsClassifier(n_neighbors= 7)
```
設定KNN model，N值為7，代表輸入資料會根據與他最相近的7個點的位置遭到分類，距離的計算方法使用x中的各項attribute計算。
```
model.fit(x_train, y_train)
```
使用x_train, y_train建立model
```
acc = model.score(x_test, y_test)
print(acc)
```
觀察產生model的準確率
```
predicted = model.predict(x_test)
```
使用測試集，正式開始預測。
```
names = ["acc","good","unacc","vgood"]
for x in range(len(predicted)):
    print("predicted_y:", names[predicted[x]],"input_x: ",x_test[x],"actual_y: ",names[y_test[x]])
    n = model.kneighbors([x_test[x]], 7, True)
    print("N:", n)
```
此部分目的為輸出預測結果。
參考輸出：
> predicted_y: unacc input_x:  (0, 1, 1, 0, 1, 1) actual_y:  unacc
N: (array([[1., 1., 1., 1., 1., 1., 1.]]), array([[ 755, 1056, 1395,  983,  229, 1043, 1390]], dtype=int64))

predicted_y為我們預測的class，input_x為我們輸入的各項attribute(e.g. buying, maint, ....)，actual_y是實際上的class。
"N:"輸出我們用來分類的資料點，第一個array為輸入點跟各點的距離，第二個array為該資料點的編號。