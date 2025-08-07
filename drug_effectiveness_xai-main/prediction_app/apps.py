from django.apps import AppConfig


class PredictionAppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'prediction_app'
    
    def ready(self):
        # Import the signal handlers
        import prediction_app.signals
