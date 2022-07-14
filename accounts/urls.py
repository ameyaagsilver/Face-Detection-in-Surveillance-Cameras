from django.urls import include, path
from django.contrib.auth.views import LoginView

urlpatterns = [
    path('', LoginView.as_view(), name='login'),
]