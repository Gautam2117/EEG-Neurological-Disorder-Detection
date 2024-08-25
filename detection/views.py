from django.shortcuts import render
from django.http import JsonResponse
import tensorflow as tf
import numpy as np
from django.views.decorators.csrf import csrf_exempt
import json

from tensorflow.keras.optimizers import Adam

model = tf.keras.models.load_model(r'D:\140724\alzhimer_eeg_ep\alzheimer_predictor\parkinsons_model_no_opt.h5')
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

@csrf_exempt
def predict(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            features = np.array(data['features'], dtype=float).reshape(1, -1)  # Ensure it's float and correctly shaped
            prediction = model.predict(features)
            result = int(prediction[0][0] > 0.5)
            return JsonResponse({'prediction': result})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
    return JsonResponse({'error': 'Invalid request'}, status=400)

def upload(request):
    return render(request, 'detection/upload.html')