import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from Interface.loginpage.Models.NeuralNetwrok import NeuralNetwork


neural = tf.keras.models.load_model("neural.keras")
dft = pd.read_csv("BuildingModels/Data/Power.csv")
nodes = dft.iloc[:,0].values
Xt = dft.iloc[:,1:-1].values
scaler = StandardScaler()
Xt = scaler.fit_transform(Xt)
# print(max(Xt[0]), max(Xt[1]), max(Xt[2]))
yt = dft.iloc[:,-1].values
neural.evaluat(Xt,yt)
y_pred = neural.predicting(Xt)

m = confusion_matrix(y_pred, yt)
print(m)