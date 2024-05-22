import numpy as np
import pandas as pd
import tensorflow as tf
from imblearn.over_sampling import RandomOverSampler
from tensorflow import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def Scaling(X):
    scaler = StandardScaler()
    X=scaler.fit_transform(X)
    return X
if __name__=="__main__":
    # Generate some sample data
    # For simplicity, let's generate random binary classification data
    # Replace this with your actual dataset
    df = pd.read_csv("Data_t.csv")

    #df = df.drop("Unnamed: 0")

    X = df.iloc[:,:-1].values
    y = df.iloc[:, -1].values
    over = RandomOverSampler()
    X, y = over.fit_resample(X, y)
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the neural network model
    model = keras.Sequential([
        keras.layers.Dense(32, activation='relu', input_shape=(X.shape[1],)),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')  # Output layer with sigmoid activation for binary classification
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)


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