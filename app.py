import os
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import base64
from io import BytesIO


app = Flask(__name__)


try:
    model = tf.keras.models.load_model('cat_dog_model.h5')
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None


UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


IMG_SIZE = (150, 150)


def preprocess_image(img_path):
    """
    Loads, resizes, and normalizes an image for model prediction.
    """
    img = Image.open(img_path).convert('RGB')
    img = img.resize(IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    img_array /= 255.0
    return img_array


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
      
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

       
        img_pil = Image.open(filepath)
        buffered = BytesIO()
        img_pil.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        img_data_url = f"data:image/jpeg;base64,{img_str}"

     
        img_array = preprocess_image(filepath)

        
        prediction = model.predict(img_array)
        probability = prediction[0][0]

        if probability > 0.5:
            result = 'Dog 🐕'
            confidence = probability * 100
        else:
            result = 'Cat 🐈'
            confidence = (1 - probability) * 100

       
        os.remove(filepath)

        return jsonify({
            'result': result,
            'confidence': f'{confidence:.2f}%',
            'image_url': img_data_url  
        })

if __name__ == '__main__':
    app.run(debug=False)