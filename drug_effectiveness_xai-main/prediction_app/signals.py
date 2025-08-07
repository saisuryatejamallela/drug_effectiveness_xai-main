from django.db.models.signals import post_migrate
from django.dispatch import receiver
from django.contrib.auth.models import User
from django.conf import settings
import os

@receiver(post_migrate)
def create_superuser(sender, **kwargs):
    """Create a superuser after migration if not already exists"""
    if sender.name == 'prediction_app':
        if not User.objects.filter(username='admin').exists():
            User.objects.create_superuser('admin', 'admin@example.com', 'admin')
            print("Superuser created.")
