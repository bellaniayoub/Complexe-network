"""loginpage URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    #path('admin/', admin.site.urls),
    #path('base', views.base, name='base'),
    path('', views.home, name='home'),
    path('neural_network/', views.neural_network, name='neural_network'),
    path('linear_regression/', views.linear_regression, name='linear_regression'),
    path('k_nearest_neighbors/', views.k_nearest_neighbors, name='k_nearest_neighbors'),
    path('decision_tree/', views.decision_tree, name='decision_tree'),
]
