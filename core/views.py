from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
import json
import pandas as pd
import numpy as np
from .ml_utils import MLManager

ml_manager = MLManager()

def index(request):
    """Renderiza el frontend"""
    return render(request, 'index.html')

@csrf_exempt
def get_preview_data(request):
    """Devuelve 10 filas y las columnas correlacionadas"""
    df, best_features = ml_manager.load_and_select_features()
    
    # Tomamos solo 10 filas para mostrar
    preview = df.head(10).copy()
    
    # Marcamos qué columnas son las seleccionadas
    data = {
        'columns': list(preview.columns),
        'rows': preview.to_dict(orient='records'),
        'correlated_features': best_features
    }
    return JsonResponse(data)

@csrf_exempt
def train_api(request):
    if request.method == 'POST':
        body = json.loads(request.body)
        percentage = float(body.get('percentage', 80))
        
        # Entrenar
        metrics = ml_manager.train_model(percentage)
        
        return JsonResponse(metrics)
    return JsonResponse({'error': 'POST required'}, status=400)

@csrf_exempt
def calculate_sigmoid(request):
    """
    Calcula la curva sigmoidea basada en el desplazamiento del slider (bias)
    """
    shift_x = request.GET.get('shift', 0)
    x, y = ml_manager.get_sigmoid_data(shift_x)
    
    return JsonResponse({
        'x': x,
        'y': y
    })