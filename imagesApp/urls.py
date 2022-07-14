from django.contrib import admin
from django.urls import path, include
import imagesApp.views as imageAppViews


urlpatterns = [
    path('get-all-persons/', imageAppViews.getAllPersons, name='get-all-persons'),
    path('get-more-info-on-person/<int:personId>', imageAppViews.getMoreInfoOnPerson, name='get-more-info-on-person'),
]