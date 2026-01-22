import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense
from sklearn.preprocessing import MinMaxScaler

data=np.array([10,12,13,15,18,20,22,25,28,30]).reshape(-1,1)
scaler=MinMaxScaler()
data=scaler.fit_transform(data)

X=[]
y=[]
for i in range(len(data)-3):
    X.append(data[i:i+3])
    y.append(data[i+3])
X=np.array(X)
y=np.array(y)

model=Sequential()
model.add(LSTM(50,activation='relu',input_shape=(3,1)))
model.add(Dense(1))
model.compile(optimizer='adam',loss='mse')
model.fit(X,y,epochs=200,verbose=0)

test=np.array([25,28,30]).reshape(1,3,1)
test=scaler.transform(test.reshape(-1,1)).reshape(1,3,1)
pred=model.predict(test)
pred=scaler.inverse_transform(pred)
print(pred[0][0])
