from django.shortcuts import render, redirect
from django.http import HttpResponse
import pandas as pd
from django.shortcuts import render
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def base(request):
    return render(request,'base.html')
def home(request):
    return render(request,'new.html')
def neural_network(request):
    return render(request, 'neural_network.html')

def linear_regression(request):
    return render(request, 'linear_regression.html')

def k_nearest_neighbors(request):
    return render(request, 'k_nearest_neighbors.html')

def decision_tree(request):
    return render(request, 'decision_tree.html')


def process_neural_network(data):
    # Ajoutez ici le traitement spécifique de votre algorithme de réseau de neurones
    # Par exemple, prédictions, entraînement, etc.
    return "Résultat du traitement"

def generate_neural_network_pdf(result):
    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = 'attachment; filename="neural_network_result.pdf"'

    # Créer un canvas pour le PDF
    pdf_canvas = canvas.Canvas(response, pagesize=letter)
    pdf_canvas.drawString(100, 750, f'Résultat du Réseau de neurones : {result}')
    pdf_canvas.save()

    return response
def neural_network(request):
    if request.method == 'POST' and request.FILES['file']:
        csv_file = request.FILES['file']
        data = pd.read_csv(csv_file)
        result = process_neural_network(data)
        return generate_neural_network_pdf(result)
    return render(request, 'neural_network.html')

def process_linear_regression(data):
    # Ajoutez ici le traitement spécifique de votre algorithme de régression linéaire
    return "Résultat du traitement"
def generate_linear_regression_pdf(result):
    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = 'attachment; filename="neural_network_result.pdf"'

    # Créer un canvas pour le PDF
    pdf_canvas = canvas.Canvas(response, pagesize=letter)
    pdf_canvas.drawString(100, 750, f'Résultat du Réseau de neurones : {result}')
    pdf_canvas.save()

    return response

def linear_regression(request):
    if request.method == 'POST' and request.FILES['file']:
        csv_file = request.FILES['file']
        data = pd.read_csv(csv_file)
        result = process_linear_regression(data)
        return generate_linear_regression_pdf(result)
    return render(request, 'linear_regression.html')


def process_k_nearest_neighbors(data):
    # Ajoutez ici le traitement spécifique de votre algorithme K-ppv
    return "Résultat du traitement"
def generate_k_nearest_neighbors_pdf(result):
    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = 'attachment; filename="neural_network_result.pdf"'

    # Créer un canvas pour le PDF
    pdf_canvas = canvas.Canvas(response, pagesize=letter)
    pdf_canvas.drawString(100, 750, f'Résultat du Réseau de neurones : {result}')
    pdf_canvas.save()

    return response

def k_nearest_neighbors(request):
    if request.method == 'POST' and request.FILES['file']:
        csv_file = request.FILES['file']
        data = pd.read_csv(csv_file)
        result = process_k_nearest_neighbors(data)
        return generate_k_nearest_neighbors_pdf(result)
    return render(request, 'k_nearest_neighbors.html')


def process_decision_tree(data):
    # Ajoutez ici le traitement spécifique de votre algorithme d'arbre de décision
    return "Résultat du traitement"
def generate_decision_tree_pdf(result):
    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = 'attachment; filename="neural_network_result.pdf"'

    # Créer un canvas pour le PDF
    pdf_canvas = canvas.Canvas(response, pagesize=letter)
    pdf_canvas.drawString(100, 750, f'Résultat du Réseau de neurones : {result}')
    pdf_canvas.save()

    return response

def decision_tree(request):
    if request.method == 'POST' and request.FILES['file']:
        csv_file = request.FILES['file']
        data = pd.read_csv(csv_file)
        result = process_decision_tree(data)
        return generate_decision_tree_pdf(result)
    return render(request, 'decision_tree.html')

