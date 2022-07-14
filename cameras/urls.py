from django import views
from django.contrib import admin
from django.urls import path, include
from .views import StatusView, AddNewCamera, addCameraList, livefeed, CameraFeed, EditCameraDetails

urlpatterns = [
    path('newCamera/', AddNewCamera.as_view(), name='new-camera'),
    path('addCameraList/', addCameraList),
    path('video/<int:cameraId>', livefeed, name='video-feed'),
    path('cameraFeed/<int:cameraId>/', CameraFeed.as_view(), name='feed-page'),
    path('editCamera/<int:cameraId>/', EditCameraDetails.as_view(), name='edit-page'),
    path('<str:department>/', StatusView.as_view(), name='status-view'),
]
