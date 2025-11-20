from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from .ml_utils import MLManager

ml_manager = MLManager()

def index(request):
    return render(request, 'index.html')

@csrf_exempt
def get_preview_data(request):
    # Usamos la versión ligera
    df = ml_manager.get_preview_data_light()
    # Intentamos cargar el modelo para saber las features
    _, features = ml_manager.load_model_from_mongo()
    
    data = {
        'columns': list(df.columns),
        'rows': df.to_dict(orient='records'),
        'correlated_features': features if features else ['No cargado', 'No cargado']
    }
    return JsonResponse(data)

@csrf_exempt
def train_api(request):
    # YA NO ENTRENAMOS EN VIVO. 
    # Este endpoint ahora recarga el modelo de Mongo y devuelve sus metadatos.
    metrics = ml_manager.predict_dummy()
    return JsonResponse(metrics)

@csrf_exempt
def calculate_sigmoid(request):
    shift_x = request.GET.get('shift', 0)
    x, y = ml_manager.get_sigmoid_data(shift_x)
    return JsonResponse({'x': x, 'y': y})