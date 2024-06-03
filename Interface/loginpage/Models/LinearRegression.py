import joblib
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import confusion_matrix


model = joblib.load('C:/Users/bella/Desktop/PFE/Interface/static/Models/Linear_regression.h5')

dft = pd.read_csv("BuildingModels/Data/Football.csv")
Xt = dft.iloc[:,1:-1].values
scaler = StandardScaler()
Xt = scaler.fit_transform(Xt)
print(max(Xt[0]), max(Xt[1]), max(Xt[2]))
yt = dft.iloc[:,-1].values
y_pred = model.predict(Xt)

# print(y_pred)
m = confusion_matrix(y_pred, yt)
print(m)
dft = pd.read_csv("BuildingModels/Data/Power.csv")
Xt = dft.iloc[:,1:-1].values
scaler = StandardScaler()
Xt = scaler.fit_transform(Xt)
print(max(Xt[0]), max(Xt[1]), max(Xt[2]))
yt = dft.iloc[:,-1].values
y_pred = model.predict(Xt)

# print(y_pred)
m = confusion_matrix(y_pred, yt)
print(m)