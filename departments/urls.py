from django.contrib import admin
from django.urls import path, include
from .views import AddNewClusterView

urlpatterns = [
    path('newCluster/', AddNewClusterView.as_view(), name='new-cluster'),
]