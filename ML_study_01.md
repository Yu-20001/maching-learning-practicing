# LinearRegression_01
linear regression 線性回歸

code from:
https://www.techwithtim.net/tutorials/machine-learning-python/linear-regression-2/

以下嘗試分析程式碼:
```
import pandas as pd 
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
```
引用pandas(用dataframe), numpy(numpy array), sklearn(數學模型)

```data = pd.read_csv("student-mat.csv",sep=";")```
將csv檔(資料包)轉為dataframe資料結構，且設定用來分割的符號為";"
```data = data[["G1", "G2", "G3"]]```
用list選取dataframe中想要的資料，我們選擇用G1、G2預測G3的成績
```predict = "G3"```
設定我們要預測的項目為G3
```
x = np.array(data.drop([predict],1))
y = np.array(data[predict])
```
x為training data, y是我們預測的結果
用data.drop去除掉G3並存為numpy的array(因為train_test_split需要用array)
```
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
```
使用train_test_split(array*,test_size,random_state)
將x分為x_train、x_test，y分為y_train、y_test(訓練集、測試集)，test_size設定測試集佔全部樣本的比例，未設定random_state則預設為None
```linear = linear_model.LinearRegression()```
建立線性回歸的模型
```linear.fit(x_train,y_train)```
用trainning data生成回歸直線
```acc = linear.score(x_test,y_test)```
用.score()觀察模型的準確度
```
print(acc)
print("Coefficient:",linear.coef_)
```
印出回歸曲線各項的參數(g1、g2)
```
print("Intercept",linear.intercept_)
```
印出回歸曲線的截距
```
predictions = linear.predict(x_test)
for x in range(len(predictions)):
    print(predictions[x],x_test[x],y_test[x])
```
實際使用測試集的資料(x_test)，做出回歸曲線，顯示預測結果(G3)、input、實際結果以供比較。


note:
```x = np.array(data.drop([predict],1))```
這行出現了以下的warning
FutureWarning: In a future version of pandas all arguments of DataFrame.drop except for the argument 'labels' will be keyword-only
將這行改為
```x = np.array(data.drop(column=predict))```
