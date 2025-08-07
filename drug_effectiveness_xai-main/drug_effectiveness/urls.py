"""
URL configuration for drug_effectiveness project.
"""
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('prediction_app.urls')),
]
