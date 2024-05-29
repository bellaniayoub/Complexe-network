import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
import tensorflow as tf

class NeuralNetwork(tf.keras.Model):
    @tf.keras.utils.register_keras_serializable()
    class NeuralNetwork(tf.keras.Model):
        pass

    def __init__(self, input_size=3 , hidden_size=8, output_size=1):
        super(NeuralNetwork, self).__init__()
        self.input_size = input_size,
        self.hidden_size = hidden_size,
        self.output_size = hidden_size,
        self.input_layer = tf.keras.layers.Input(shape=(input_size,))
        self.hidden_layer1 = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.hidden_layer2 = tf.keras.layers.Dense(int(hidden_size/2), activation='sigmoid')

        # self.hidden_layer2 = tf.keras.layers.Dense(2*hidden_size, activation='relu')
        self.output_layer = tf.keras.layers.Dense(output_size, activation='sigmoid')

    def call(self, inputs):
        x = self.hidden_layer1(inputs)
        y = self.hidden_layer2(x)
        return self.output_layer(y)
    
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
        self.evaluate(X, y)
        # print(error, accuracy)


    def predicting(self, X):
        """
            predict data and calssify it
        """
        y = self.predict(X)
        print(max(y), min(y))
        y = [1 if i>0.5 else 0 for i in y]
        return y
    
    def extract(self, path="neural.h5"):
        try:
            self.save(path)
            print(f"Model saved successfully to {path}")
        except Exception as e:
            print(f"Error occurred while saving the model: {e}")
    def get_config(self):
        return {
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size,
        }

    @classmethod
    def from_config(cls, config):
        config.pop('trainable', None)
        config.pop('dtype')
        return cls(**config)


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split # type: ignore
    from sklearn.metrics import confusion_matrix
    # from sklearn.utils import __all__
    # from imblearn.over_sampling import RandomOverSampler
    # try:
    #     from imblearn.over_sampling import RandomOverSampler
    # except ImportError as e:
    #     print("Error importing imblearn:", e)
    #neural = NeuralNetwork(3,1,1)

    neural = NeuralNetwork(3,8,1)
    opt = tf.keras.optimizers.Adam(learning_rate = 0.001)
    loss = "binary_crossentropy"
    accuracy = "accuracy"
    neural.compile_neural(opt, loss, accuracy)
    df = pd.read_csv("BuildingModels/Data/AllData.csv")
    # df.to_csv("BuildingModels/Data_t.csv", index=False)

    # i'm gonna extract the data of the nodes and train the model on facebook data


    X = df.iloc[:, 1:-1].values
    scaler = StandardScaler()
    X=scaler.fit_transform(X)
    # print(max(X[0]), max(X[1]), max(X[2]))
    y = df.iloc[:, -1].values

    # over = RandomOverSampler()
    # X, y = over.fit_resample(X,y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=42)
    neural.train(X_train,y_train, 16, 100)
    print("Test:")
    neural.evaluat(X_test, y_test)
    y_pred = neural.predicting(X_test)
    m = confusion_matrix(y_pred, y_test)
    print(m)
    # print("\n\n")
    print("Facebook:")
    dft = pd.read_csv("BuildingModels/Data/GrQc.csv")
    Xt = dft.iloc[:,1:-1].values
    Xt = scaler.fit_transform(Xt)
    # print(max(Xt[0]), max(Xt[1]), max(Xt[2]))
    yt = dft.iloc[:,-1].values
    neural.evaluat(Xt,yt)
    y_pred = neural.predicting(Xt)

    # print(y_pred)
    m = confusion_matrix(y_pred, yt)
    print(m)
    print("Karate:")
    dft = pd.read_csv("BuildingModels/Data/karate.csv")
    Xt = dft.iloc[:,1:-1].values
    Xt = scaler.fit_transform(Xt)
    # print(max(Xt[0]), max(Xt[1]), max(Xt[2]))
    yt = dft.iloc[:,-1].values
    neural.evaluat(Xt,yt)
    y_pred = neural.predicting(Xt)

    # print(y_pred)
    m = confusion_matrix(y_pred, yt)
    print(m)

    print("Dolphins:")
    dft = pd.read_csv("BuildingModels/Data/Dolphins.csv")
    Xt = dft.iloc[:,1:-1].values
    Xt = scaler.fit_transform(Xt)
    # print(max(Xt[0]), max(Xt[1]), max(Xt[2]))
    yt = dft.iloc[:,-1].values
    neural.evaluat(Xt,yt)
    y_pred = neural.predicting(Xt)

    # print(y_pred)
    m = confusion_matrix(y_pred, yt)
    print(m)

    print("Footaball:")
    dft = pd.read_csv("BuildingModels/Data/Football.csv")
    Xt = dft.iloc[:,1:-1].values
    Xt = scaler.fit_transform(Xt)
    print(max(Xt[0]), max(Xt[1]), max(Xt[2]))
    yt = dft.iloc[:,-1].values
    neural.evaluat(Xt,yt)
    y_pred = neural.predicting(Xt)

    # print(y_pred)
    m = confusion_matrix(y_pred, yt)
    print(m)


    # dfm = pd.read_csv("BuildingModels/Data_Facebook.csv")
    # X = df.iloc[:, 1:-1].values
    # X=scaler.fit_transform(X)
    # print(max(X[0]), max(X[1]), max(X[2]))
    # y = df.iloc[:, -1].values
    # # over = RandomOverSampler()

    # neural.evaluat(X, y)

    # y_pred = neural.predicting(X)
    # #print(max(y_pred))
    # m = confusion_matrix(y_pred, y)
    # print(m)
    neural.extract("neural.keras")

    # dft = pd.read_csv("BuildingModels/Data/Dolphins.csv")
    # # dft = dft.drop("Unnamed: 0", axis=1)
    # Xt = dft.iloc[:,1:-1]
    # yt = dft.iloc[:,-1]
    # loaded_model = tf.keras.models.load_model('neural.keras')

    # loaded_model.evaluat(Xt, yt) 


    


