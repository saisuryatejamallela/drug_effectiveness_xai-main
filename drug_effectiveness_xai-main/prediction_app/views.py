from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.http import HttpResponse
from .forms import PredictionForm
from .models import DrugPrediction
from .ml_models.model import DrugEffectivenessModel
import os
from django.conf import settings

def index(request):
    """Home page view"""
    return render(request, 'prediction_app/index.html')

def predict(request):
    """View for making predictions"""
    if request.method == 'POST':
        form = PredictionForm(request.POST)
        if form.is_valid():
            # Get form data
            age_group = form.cleaned_data['age_group']
            sex = form.cleaned_data['sex']
            condition = form.cleaned_data['condition']
            drugs = form.cleaned_data['drugs']
            symptoms = form.cleaned_data['symptoms']
            
            try:
                # Initialize the model and make prediction
                model = DrugEffectivenessModel()
                effectiveness = model.predict(age_group, sex, condition, drugs, symptoms)
                
                # Save the prediction to the database
                prediction = DrugPrediction.objects.create(
                    age_group=age_group,
                    sex=sex,
                    condition=condition,
                    drugs=drugs,
                    symptoms=symptoms,
                    effectiveness_score=effectiveness
                )
                
                # Redirect to results page
                return redirect('prediction_app:results', prediction_id=prediction.id)
            
            except Exception as e:
                messages.error(request, f"Error making prediction: {str(e)}")
                return render(request, 'prediction_app/predict.html', {'form': form})
    else:
        form = PredictionForm()
    
    return render(request, 'prediction_app/predict.html', {'form': form})

def results(request, prediction_id):
    """View for displaying prediction results"""
    prediction = get_object_or_404(DrugPrediction, id=prediction_id)
    
    try:
        # Generate explanation
        model = DrugEffectivenessModel()
        explanation_img = model.explain_prediction(
            prediction.age_group,
            prediction.sex,
            prediction.condition,
            prediction.drugs,
            prediction.symptoms
        )
        
        context = {
            'prediction': prediction,
            'explanation_img': explanation_img
        }
        
        return render(request, 'prediction_app/results.html', context)
    
    except Exception as e:
        messages.error(request, f"Error generating explanation: {str(e)}")
        return redirect('prediction_app:index')

def train_model(request):
    """View for training the model (admin only)"""
    if not request.user.is_superuser:
        messages.error(request, "You don't have permission to train the model")
        return redirect('prediction_app:index')
    
    if request.method == 'POST':
        if 'dataset' in request.FILES:
            # Save the uploaded dataset
            dataset = request.FILES['dataset']
            dataset_path = os.path.join(settings.MEDIA_ROOT, 'dataset.csv')
            os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
            
            with open(dataset_path, 'wb+') as destination:
                for chunk in dataset.chunks():
                    destination.write(chunk)
            
            try:
                # Train the model
                model = DrugEffectivenessModel()
                rmse, r2 = model.train(dataset_path)
                
                messages.success(request, f"Model trained successfully. RMSE: {rmse:.4f}, R2: {r2:.4f}")
                return redirect('prediction_app:index')
            
            except Exception as e:
                messages.error(request, f"Error training model: {str(e)}")
        else:
            messages.error(request, "No dataset file uploaded")
    
    return render(request, 'prediction_app/train.html')

def prediction_list(request):
    """View for listing all predictions"""
    predictions = DrugPrediction.objects.all().order_by('-created_at')
    return render(request, 'prediction_app/prediction_list.html', {'predictions': predictions})
