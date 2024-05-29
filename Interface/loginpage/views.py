import os
from django.shortcuts import render, redirect
from django.http import HttpResponse
import joblib
import pandas as pd
import numpy as np
from django.shortcuts import render
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
# from tensorflow.keras.models import load_model # type: ignore
import tensorflow as tf
import networkx as nx
from .Models.NeuralNetwrok import NeuralNetwork

# from django.conf import settings
# from django.templatetags.static import static
def base(request):
    return render(request,'base.html')
def home(request):
    return render(request,'new.html')

def getData(data):

    """Get the Graph from the input data"""
    graph = nx.from_pandas_edgelist(data)
    """Calculate the features"""
    degree = nx.degree_centrality(graph)
    closenness = nx.closeness_centrality(graph)
    betweenness = nx.betweenness_centrality(graph)

    """Calculate the nodes"""
    nodes = graph.nodes()
    """Transform the data into a numpy array"""
    Data = np.array([degree.values(),closenness.values(), betweenness.values()])

    return nodes ,Data








def predict(data, model_name):
    nodes, data = getData(data)
    print(os.getcwd())
    """ Loading the model """
    if model_name == "neural_network":
        model_path = "neural.h5"
        with tf.keras.utils.custom_object_scope({'NeuralNetwork': NeuralNetwork}):
            model = tf.keras.models.load_model(model_path)

    if model_name == "arbre_decision":
        model = joblib.load("DecisionTree.pkl")
    if model_name == "k-ppv":
        model = joblib.load("../static/Models/KNN.joblib")
    if model_name == "regression_linear":
        """After definning the model"""
        pass
    
    
    prediction = model.predict(data)

    return nodes , prediction

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
        data = pd.read_csv(csv_file, names=["source", "target"], sep=' ')

        nodes, result = predict(data, request.POST['model'])
        context = {
            "nodes":nodes,
            "prediction":result
        }
        # return generate_neural_network_pdf(result)
    return render(request, 'showResult.html', context)


