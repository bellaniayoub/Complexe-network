from django.shortcuts import render, redirect
from django.http import HttpResponse
import joblib
import pandas as pd
import numpy as np
from django.shortcuts import render
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
# from tensorflow.keras.models import load_model # type: ignore
from sklearn.discriminant_analysis import StandardScaler
import tensorflow as tf
import networkx as nx
from .Models.NeuralNetwrok import NeuralNetwork
from .Models.DecisionTree import DecisionTreeModel
from django.contrib.staticfiles.finders import find
import matplotlib.pyplot as plt

# from django.conf import settings
# from django.templatetags.static import static
def base(request):
    return render(request,'base.html')
def home(request):
    return render(request,'new.html')

def DrawGraph(data, node_list):
    """Get the Graph from the input data"""
    G = nx.from_pandas_edgelist(data)
    # Calculate node degrees
    degrees = dict(G.degree())

    # Calculate edge weights
    edge_weights = {(u, v): 1 / (1 + G.degree(u) + G.degree(v)) for u, v in G.edges()}

    # Set node colors based on degree and node list
    node_colors = []
    for node in G.nodes():
        if node in node_list:
            node_colors.append('red')  # Nodes in the list are red
        else:
            node_colors.append('skyblue')  # Other nodes are blue

    # Set edge colors based on weight
    edge_colors = [edge_weights[(u, v)] for u, v in G.edges()]

    # Draw the graph
    fig, ax = plt.subplots(figsize=(20, 20))
    nx.draw(G, pos=nx.spring_layout(G, k=0.15, iterations=50), node_color=node_colors, edge_color=edge_colors, node_size=[v*5 for v in degrees.values()], with_labels=False, ax=ax)

    # Set colorbar for node degrees
    sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=min(degrees.values()), vmax=max(degrees.values())))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Node Degree', rotation=270, labelpad=20)

    # Set colorbar for edge weights
    sm = plt.cm.ScalarMappable(cmap='Reds', norm=plt.Normalize(vmin=min(edge_weights.values()), vmax=max(edge_weights.values())))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Edge Weight', rotation=270, labelpad=20)
    # Save the figure to a file
    plt.savefig("image.png", bbox_inches='tight')

    # Close the figure to free up memory
    plt.close(fig)


def getData(data):

    """Get the Graph from the input data"""
    graph = nx.from_pandas_edgelist(data)
    """Calculate the features"""
    degree = nx.degree_centrality(graph)
    closenness = nx.closeness_centrality(graph)
    betweenness = nx.betweenness_centrality(graph)

    """Calculate the nodes"""
    nodes = graph.nodes()
    """Drawing the graph"""
    # DrawGraph(graph)
    """Transform the data into a numpy array"""
    Data = pd.DataFrame({"Node":[],"Degree":[],"Closeness":[],"betwennes":[]})
    for n, d, c, b in zip(nodes,list(degree.values()),list(closenness.values()),list(betweenness.values())):
        included = [n,d,c,b]
        Data.loc[len(Data)] = included
    
    return Data

def predict(data, model_name):
    nodes = data.iloc[:, 0]
    X = data.iloc[:,1:].values
    scaler = StandardScaler()
    X=scaler.fit_transform(X)
    """ Loading the model """
    if model_name == "neural_network":
        model_path = find("Models/neural.keras")
        with tf.keras.utils.custom_object_scope({'NeuralNetwork': NeuralNetwork}):
            model = tf.keras.models.load_model(model_path)

    if model_name == "arbre_decision":
        path = find("Models/DecisionTree.joblib")
        model = joblib.load(path)
    if model_name == "k-ppv":
        path = find("Models/KNN.joblib")
        model = joblib.load(path)
    if model_name == "regression_linear":
        """After definning the model"""
        path = find("Models/Linear_regression.h5")
        model = joblib.load(path)
    
    
    if model_name!='neural_network':
        prediction = model.predict(X)
    else:
        prediction = model.predicting(X)
    predict = [n for n, p in zip(nodes,prediction) if p==1]
    return predict

def generate_neural_network_pdf(result):
    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = 'attachment; filename="neural_network_result.pdf"'

    # Créer un canvas pour le PDF
    pdf_canvas = canvas.Canvas(response, pagesize=letter)
    pdf_canvas.drawString(100, 750, f'Résultat du Réseau de neurones : {result}')
    pdf_canvas.save()

    return response
def traiter(request):
    if request.method == 'POST' and request.FILES['file']:
        csv_file = request.FILES['file']
        data = pd.read_csv(csv_file, names=["source", "target"], sep=' ', usecols=[0,1])
        traitement_data = getData(data)
        result = predict(traitement_data, request.POST['model'])
        DrawGraph(data, result)
        context = {
            "prediction":result
        }
        # return generate_neural_network_pdf(result)
    return render(request, 'showResult.html', context)

