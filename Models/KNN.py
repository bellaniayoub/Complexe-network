from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib
import pandas as pd

class KNNClassifier:
    def __init__(self, k=3):
        self.k = k
        self.model = KNeighborsClassifier(n_neighbors=k)

    def fit(self, X_train, y_train):
        histo = self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
    def extract(self):
        joblib.dump(self.model, "KNN.joblib")

# Example usage:
if __name__ == "__main__":

    # Example dataset
    df = pd.read_csv("BuildingModels/Data/AllData.csv")
    X = df.iloc[:, 1:-1].values
    scaler = StandardScaler()
    X=scaler.fit_transform(X)
    print(max(X[0]), max(X[1]), max(X[2]))
    y = df.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=42)
    # Initialize and fit KNN classifier
    knn_classifier = joblib.load("KNN.joblib")
    knn_classifier.fit(X_train,y_train)
    dft = pd.read_csv("BuildingModels/Data/Facebook.csv")
    Xt = dft.iloc[:,1:-1].values
    Xt = scaler.fit_transform(Xt)
    yt = dft.iloc[:,-1].values
    predictions = knn_classifier.predict(Xt)
    # print("Predictions:", predictions)


    m = confusion_matrix(predictions, yt)
    print(m)

    # Example test data

    # Make predictions
    predictions = knn_classifier.predict(X_test)
    # print("Predictions:", predictions)

    m = confusion_matrix(predictions, y_test)
    print(m)

    dft = pd.read_csv("BuildingModels/Data/Dolphins.csv")
    Xt = dft.iloc[:,1:-1].values
    Xt = scaler.fit_transform(Xt)
    yt = dft.iloc[:,-1].values
    
    predictions = knn_classifier.predict(Xt)
    # print("Predictions:", predictions)


    m = confusion_matrix(predictions, yt)
    print(m)
    dft = pd.read_csv("BuildingModels/Data/Karate.csv")
    Xt = dft.iloc[:,1:-1].values
    Xt = scaler.fit_transform(Xt)
    yt = dft.iloc[:,-1].values
    
    predictions = knn_classifier.predict(Xt)
    # print("Predictions:", predictions)


    m = confusion_matrix(predictions, yt)
    print(m)
    dft = pd.read_csv("BuildingModels/Data/Facebook.csv")
    Xt = dft.iloc[:,1:-1].values
    Xt = scaler.fit_transform(Xt)
    yt = dft.iloc[:,-1].values
    X_train, X_test, y_train, y_test = train_test_split(Xt, yt)
    knn_classifier.fit(X_train, y_train)
    predictions = knn_classifier.predict(Xt)
    # print("Predictions:", predictions)


    m = confusion_matrix(predictions, yt)
    print(m)
    dft = pd.read_csv("BuildingModels/Data/GrQc.csv")
    Xt = dft.iloc[:,1:-1].values
    Xt = scaler.fit_transform(Xt)
    yt = dft.iloc[:,-1].values
    
    predictions = knn_classifier.predict(Xt)
    # print("Predictions:", predictions)


    m = confusion_matrix(predictions, yt)
    print(m)

    knn_classifier.extract()

