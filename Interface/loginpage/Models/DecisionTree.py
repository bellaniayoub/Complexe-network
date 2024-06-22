import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib

class DecisionTreeModel:
    def __init__(self):
        self.model = DecisionTreeClassifier(max_depth=2)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy:", accuracy)
        
    def extract(self, path="DecisionTree.pkl"):
        joblib.dump(self, path)


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split # type: ignore
    from sklearn.metrics import confusion_matrix
    # try:
    #     from imblearn.over_sampling import RandomOverSampler
    # except ImportError as e:
    #     print("Error importing imblearn:", e)
    #neural = NeuralNetwork(3,1,1)

    neural = DecisionTreeModel()

    #neural.compile_neural("sgd", 'binary_crossentropy', 'accuracy')
    df = pd.read_csv("BuildingModels/Data/AllData.csv")
    # df = df.drop("Unnamed: 0", axis=1)

    X = df.iloc[:, 1:-1].values
    scaler = StandardScaler()
    X=scaler.fit_transform(X)
    y = df.iloc[:, -1].values

    # over = RandomOverSampler()
    # X, y = over.fit_resample(X,y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=42)
    neural.train(X_train,y_train)

    neural.evaluate(X_test, y_test)
    y_pred = neural.predict(X_test)
    m = confusion_matrix(y_pred, y_test)
    print(m)
    print("\n\n")
    # dft = pd.read_csv("Dolphine.csv")
    # # dft = dft.drop("Unnamed: 0", axis=1)
    # Xt = dft.iloc[:,:-1].values
    # Xt = scaler.fit_transform(Xt)
    # yt = dft.iloc[:,-1].values
    # y_pred = neural.predict(Xt)
    # m = confusion_matrix(y_pred, yt)
    # print(m)


    df = pd.read_csv("BuildingModels/Data/Dolphins.csv")
    # df = df.drop("Unnamed: 0", axis=1)

    X = df.iloc[:, 1:-1].values
    X=scaler.fit_transform(X)
    y = df.iloc[:, -1].values
    neural.evaluate(X, y)

    y_pred = neural.predict(X)
    #print(max(y_pred))
    m = confusion_matrix(y_pred, y)
    print(m)

    print("Facebook")

    df = pd.read_csv("BuildingModels/Data/GrQc.csv")
    # df = df.drop("Unnamed: 0", axis=1)

    X = df.iloc[:, 1:-1].values
    X=scaler.fit_transform(X)
    y = df.iloc[:, -1].values
    neural.evaluate(X, y)

    y_pred = neural.predict(X)
    #print(max(y_pred))
    m = confusion_matrix(y_pred, y)
    print(m)

    print("Power")
    df = pd.read_csv("BuildingModels/Data/Power.csv")
    # df = df.drop("Unnamed: 0", axis=1)

    X = df.iloc[:, 1:-1].values
    X=scaler.fit_transform(X)
    y = df.iloc[:, -1].values
    neural.evaluate(X, y)

    y_pred = neural.predict(X)
    #print(max(y_pred))
    m = confusion_matrix(y_pred, y)
    print(m)

    df = pd.read_csv("BuildingModels/Data/Karate.csv")
    # df = df.drop("Unnamed: 0", axis=1)

    X = df.iloc[:, 1:-1].values
    X=scaler.fit_transform(X)
    y = df.iloc[:, -1].values
    neural.evaluate(X, y)

    y_pred = neural.predict(X)
    #print(max(y_pred))
    m = confusion_matrix(y_pred, y)
    print(m)

    df = pd.read_csv("BuildingModels/Data/Football.csv")
    # df = df.drop("Unnamed: 0", axis=1)

    X = df.iloc[:, 1:-1].values
    X=scaler.fit_transform(X)
    y = df.iloc[:, -1].values
    neural.evaluate(X, y)

    y_pred = neural.predict(X)
    #print(max(y_pred))
    m = confusion_matrix(y_pred, y)
    print(m)

    df = pd.read_csv("BuildingModels/Data/Science.csv")
    # df = df.drop("Unnamed: 0", axis=1)

    X = df.iloc[:, 1:-1].values
    X=scaler.fit_transform(X)
    y = df.iloc[:, -1].values
    neural.evaluate(X, y)

    y_pred = neural.predict(X)
    #print(max(y_pred))
    m = confusion_matrix(y_pred, y)
    print(m)
    neural.extract()

    # dft = pd.read_csv("BuildingModels/Dolphine.csv")
    # dft = dft.drop("Unnamed: 0", axis=1)
    # Xt = dft.iloc[:,:-1]
    # yt = dft.iloc[:,-1]
    # loaded_model = tf.keras.models.load_model('neural.h5')

    # loaded_model.evaluat(X, y) 
