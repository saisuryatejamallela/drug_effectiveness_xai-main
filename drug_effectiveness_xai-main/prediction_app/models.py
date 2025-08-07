from django.db import models

class DrugPrediction(models.Model):
    age_group = models.CharField(max_length=20)
    sex = models.CharField(max_length=10)
    condition = models.CharField(max_length=100)
    drugs = models.TextField()
    symptoms = models.TextField()
    effectiveness_score = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Prediction for {self.condition} - Score: {self.effectiveness_score:.2f}"
