import json
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
# import matplotlib
# matplotlib.use('Agg')  # Use a non-interactive backend
# import matplotlib.pyplot as plt
# from io import BytesIO
import base64
import plotly.graph_objects as go

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
    node_colors = ['red' if node in node_list else 'skyblue' for node in G.nodes()]
    pos = nx.spring_layout(G)
    # Create Plotly graph
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')
    node_x = []
    node_y = []
    for node in G.nodes():
        node_x.append(pos[node][0])
        node_y.append(pos[node][1])

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='Blues',
            reversescale=True,
            color=[],
            size=[v*5 for v in degrees.values()],
            colorbar=dict(
                thickness=15,
                title='Node Degree',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))

    node_trace.marker.color = list(degrees.values())

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        showlegend=False,
                        hovermode='closest',
                        title="Network Graph",
                        margin=dict(b=20, l=5, r=5, t=40),
                        annotations=[dict(
                            text="",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002)],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )

    return fig.to_json()



def getData(data):

    """Get the Graph from the input data"""
    graph = nx.from_pandas_edgelist(data)
    """Calculate the features"""
    degree = nx.degree_centrality(graph)
    closenness = nx.closeness_centrality(graph)
    betweenness = nx.betweenness_centrality(graph)

    """Calculate the nodes"""
    nodes = [{'id': node, **data} for node, data in graph.nodes(data=True)]
    links = [{'source': source, 'target': target, **data} for source, target, data in graph.edges(data=True)]

    # Convert the data to JSON strings
    nodes_json = json.dumps(nodes)
    links_json = json.dumps(links)
    """Drawing the graph"""
    # DrawGraph(graph)
    """Transform the data into a numpy array"""
    Data = pd.DataFrame({"Node":[],"Degree":[],"Closeness":[],"betwennes":[]})
    for n, d, c, b in zip(nodes,list(degree.values()),list(closenness.values()),list(betweenness.values())):
        included = [n,d,c,b]
        Data.loc[len(Data)] = included
    
    return Data, nodes_json, links_json
# def get_network_image(data, node_list):

#     # Generate the graph image
#     fig, _= plt.subplots(figsize=(20, 20))
#     buffer = BytesIO()
#     DrawGraph(data, node_list)
#     fig.savefig(buffer, format='jpeg')
#     buffer.seek(0)
#     image_png = buffer.getvalue()
#     buffer.close()
#     plt.close(fig)
#     # Encode the image data as a base64 string
#     image_base64 = base64.b64encode(image_png).decode('utf-8')

#     print(image_base64[-1])
#     # Pass the image data to the template
#     return image_base64
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
        print("DEbut")
        csv_file = request.FILES['file']
        data = pd.read_csv(csv_file, names=["source", "target"], sep=' ', usecols=[0,1])
        traitement_data, nodes, links = getData(data)
        result = predict(traitement_data, request.POST['model'])
        # json = DrawGraph(data, result)
        context = {
            "nodes":nodes,
            "links":links,
            "prediction":json.dumps(result)
        }
        print("i'm here")
        # return generate_neural_network_pdf(result)
    return render(request, 'showResult.html', context)