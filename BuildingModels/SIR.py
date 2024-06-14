from matplotlib import pyplot as plt
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
import tensorflow as tf
import networkx as nx
import random
import joblib

@tf.keras.utils.register_keras_serializable()
class NeuralNetwork(tf.keras.Model):

    def __init__(self, input_size=3 , hidden_size=8, output_size=1):
        super(NeuralNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
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

    def train(self, X, y, batch, epochs, validation):
        """
            Trainning the model
        """
        hist = self.fit(X,y, batch, epochs, validation_data= validation)
        print(hist)
        return hist
    def showProgress(self, history):


        # history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

        # Plot training & validation accuracy values
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')

        # Plot training & validation loss values
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')

        plt.show()
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
        # config.pop('trainable', None)
        # config.pop('dtype')
        return cls(**config)

def sir_model(G, beta, gamma, initial_infected, max_steps=100):
    """
    Implements the SIR model for the spread of influence in a complex network.

    Args:
        G (NetworkX Graph): The network graph.
        beta (float): The infection rate.
        gamma (float): The recovery rate.
        initial_infected (list): A list of initially infected nodes.
        max_steps (int): The maximum number of steps to simulate.

    Returns:
        dict: A dictionary containing the number of susceptible, infected, and recovered nodes at each time step.
    """
    # Initialize node states
    states = {node: 'S' for node in G.nodes()}
    for node in initial_infected:
        states[node] = 'I'

    # Initialize time series
    time_series = {
        'S': [sum(state == 'S' for state in states.values())],
        'I': [sum(state == 'I' for state in states.values())],
        'R': [sum(state == 'R' for state in states.values())]
    }

    # Simulation loop
    for step in range(max_steps):
        new_states = states.copy()
        for node, state in states.items():
            if state == 'I':
                neighbors = list(G.neighbors(node))
                for neighbor in neighbors:
                    if states[neighbor] == 'S' and random.random() < beta:
                        new_states[neighbor] = 'I'
                if random.random() < gamma:
                    new_states[node] = 'R'
        states = new_states

        # Update time series
        time_series['S'].append(sum(state == 'S' for state in states.values()))
        time_series['I'].append(sum(state == 'I' for state in states.values()))
        time_series['R'].append(sum(state == 'R' for state in states.values()))

    return time_series
if __name__=="__main__":
    # Create a NetworkX graph
    file = "BuildingModels/facebook_combined.txt"
    df = pd.read_csv(file, sep=' ', names=["source", "target"], usecols=[0,1])
    G = nx.from_pandas_edgelist(df, "source", "target")
    # Set the infection rate and recovery rate
    beta = 0.5
    gamma = 0.3
    data = pd.read_csv("BuildingModels/Data/Facebook.csv")
    model = tf.keras.models.load_model("C:/Users/bella/Desktop/PFE/Interface/static/Models/neural.keras")
    X = data.iloc[:,1:-1].values
    sclare = StandardScaler()
    X = sclare.fit_transform(X)
    y_pred = model.predicting(X)
    nodes = data["Node"].values
    n_pred = []
    for n, y in zip(nodes, y_pred):
        if(y==1):
            n_pred.append(n)
        
    print(len(n_pred))
    data = data[data["influence"]==1]
    # Set the initially infected nodes
    initial_infected = data.iloc[:,0].values
    print(len(initial_infected))

    # Run the SIR model
    topsis_time_series = sir_model(G, beta, gamma, initial_infected, 40)
    your_model_time_series = sir_model(G, beta, gamma, n_pred, 40)

    # Plot the time series
    import matplotlib.pyplot as plt

# Assuming you have the time series data for both models in the following format:
# topsis_time_series = {'S': [...], 'I': [...], 'R': [...]}
# your_model_time_series = {'S': [...], 'I': [...], 'R': [...]}

# Create a figure and axis
fig, ax = plt.subplots(figsize=(8, 6))

# Plot the Susceptible population
ax.plot(topsis_time_series['S'], label='TOPSIS Susceptible', linestyle='--')
ax.plot(your_model_time_series['S'], label='Your Model Susceptible')

# Plot the Infected population
ax.plot(topsis_time_series['I'], label='TOPSIS Infected', linestyle='--')
ax.plot(your_model_time_series['I'], label='Your Model Infected')

# Plot the Recovered population
ax.plot(topsis_time_series['R'], label='TOPSIS Recovered', linestyle='--')
ax.plot(your_model_time_series['R'], label='Your Model Recovered')

# Set plot title and axis labels
ax.set_title('Comparison of SIR Models')
ax.set_xlabel('Time Step')
ax.set_ylabel('Number of Nodes')

# Add a legend
ax.legend()

# Show the plot
plt.show()
