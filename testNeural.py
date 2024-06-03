import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import tensorflow as tf
from Interface.loginpage.Models.NeuralNetwrok import NeuralNetwork
import networkx as nx
import matplotlib.pyplot as plt


neural = tf.keras.models.load_model("neural.keras")
dft = pd.read_csv("BuildingModels/Data/Facebook.csv")
nodes = dft.iloc[:,0].values
Xt = dft.iloc[:,1:-1].values
scaler = StandardScaler()
Xt = scaler.fit_transform(Xt)
# print(max(Xt[0]), max(Xt[1]), max(Xt[2]))
yt = dft.iloc[:,-1].values
neural.evaluat(Xt, yt)
y_pred = neural.predicting(Xt)

m = confusion_matrix(y_pred, yt)
print(m)
# dft = pd.read_csv("BuildingModels/Data/Facebook.csv")

neural = tf.keras.models.load_model("neural.keras")
dft = pd.read_csv("BuildingModels/Data/Power.csv")
nodes = dft.iloc[:,0].values
Xt = dft.iloc[:,1:-1].values
scaler = StandardScaler()
Xt = scaler.fit_transform(Xt)
# print(max(Xt[0]), max(Xt[1]), max(Xt[2]))
yt = dft.iloc[:,-1].values
# neural.train(Xt,yt, 16, 50)
neural.evaluat(Xt,yt)
y_pred = neural.predicting(Xt)

m = confusion_matrix(y_pred, yt)
print(m)