from django.urls import path
from . import views

app_name = 'home'
urlpatterns = [
    path('', views.IndexView, name='home'),
    path('InfoExtraction/', views.InfoExtraction, name='infoextraction')
]