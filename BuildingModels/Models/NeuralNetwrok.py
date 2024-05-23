import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
import tensorflow as tf

class NeuralNetwork(tf.keras.Model):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.input_layer = tf.keras.layers.Input(shape=(input_size,))
        self.hidden_layer = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.output_layer = tf.keras.layers.Dense(output_size, activation='sigmoid')

    def call(self, inputs):
        x = self.hidden_layer(inputs)
        return self.output_layer(x)
    
    def compile_neural(self, optimizer, loss, metric):
        """
            Compiling the model on the optimizer and loss and metric
        """
        self.compile(optimizer=optimizer, loss=loss, metrics=[metric])

    def train(self, X, y, batch, epochs):
        """
            Trainning the model
        """
        self.fit(X,y, batch, epochs)

    def evaluat(self, X, y):
        """
            evaluating the model and printing the loss and accuracy
        """
        error, accuracy = self.evaluate(X, y)
        print(error, accuracy)


    def predicting(self, X):
        """
            predict data and calssify it
        """
        y = self.predict(X)
        y = [1 if i>0.3 else 0 for i in y]
        return y
    
    def extract(self, path="neural.h5"):
        try:
            self.save(path)
            print(f"Model saved successfully to {path}")
        except Exception as e:
            print(f"Error occurred while saving the model: {e}")


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split # type: ignore
    from sklearn.metrics import confusion_matrix
    # try:
    #     from imblearn.over_sampling import RandomOverSampler
    # except ImportError as e:
    #     print("Error importing imblearn:", e)
    #neural = NeuralNetwork(3,1,1)

    neural = NeuralNetwork(3,2,1)

    neural.compile_neural("sgd", 'binary_crossentropy', 'accuracy')
    df = pd.read_csv("BuildingModels/Data_Facebook.csv")
    df = df.drop("Unnamed: 0", axis=1)

    X = df.iloc[:, :-1].values
    scaler = StandardScaler()
    X=scaler.fit_transform(X)
    y = df.iloc[:, -1].values

    # over = RandomOverSampler()
    # X, y = over.fit_resample(X,y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=42)
    neural.train(X_train,y_train, 8, 50)

    neural.evaluat(X_test, y_test)
    y_pred = neural.predicting(X_test)
    m = confusion_matrix(y_pred, y_test)
    print(m)
    print("\n\n")
    dft = pd.read_csv("BuildingModels/Dolphine.csv")
    dft = dft.drop("Unnamed: 0", axis=1)
    Xt = dft.iloc[:,:-1].values
    Xt = scaler.fit_transform(Xt)
    yt = dft.iloc[:,-1].values
    y_pred = neural.predicting(Xt)
    m = confusion_matrix(y_pred, yt)
    print(m)


    df = pd.read_csv("BuildingModels/Data_Facebook.csv")
    df = df.drop("Unnamed: 0", axis=1)

    X = df.iloc[:, :-1].values
    X=scaler.fit_transform(X)
    y = df.iloc[:, -1].values
    neural.evaluat(X, y)

    y_pred = neural.predicting(X)
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
    


