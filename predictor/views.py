from django.shortcuts import render, redirect
from .models import ImageUpload
from .forms import ImageUploadForm
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow_addons as tfa

model = tf.keras.models.load_model(
    "D:\\140724\\alzheimer_inception_lstm_model_new",
    custom_objects={"Addons>F1Score": tfa.metrics.F1Score(num_classes=4)}
)

def index(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            img = form.save()
            img_path = '.' + img.image.url
            img = image.load_img(img_path, target_size=(128, 128))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0

            # Make prediction
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions, axis=1)[0]
            
            # Print the raw prediction array and the predicted class index
            print("Raw predictions:", predictions)
            print("Predicted class index:", predicted_class)

            classes = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
            predicted_label = classes[predicted_class]
            print("Predicted label:", predicted_label)

            return render(request, 'predictor/result.html', {'predicted_label': predicted_label})
    else:
        form = ImageUploadForm()
    return render(request, 'predictor/index.html', {'form': form})
