from django.urls import path
from . import views

app_name = 'prediction_app'

urlpatterns = [
    path('', views.index, name='index'),
    path('predict/', views.predict, name='predict'),
    path('results//', views.results, name='results'),
    path('train/', views.train_model, name='train_model'),
    path('predictions/', views.prediction_list, name='prediction_list'),
]
