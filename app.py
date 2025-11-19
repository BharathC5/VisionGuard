from flask import Flask, request, jsonify,render_template
from flask_cors import CORS
import numpy as np
from PIL import Image
import io
import base64
import traceback
import tensorflow as tf

app = Flask(__name__, static_url_path='/static')
CORS(app)

# Modify model loading to print confirmation
try:
    model = tf.keras.models.load_model('model/DR4.h5')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print(traceback.format_exc())

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/home', methods=['POST'])
def predict_retinopathy():
    try:
        # Detailed logging
        print("Received prediction request")
        
        # Receive base64 image
        image_base64 = request.json['image']
        print("Image base64 received")
        
        # Remove data URL prefix if present
        if 'base64,' in image_base64:
            image_base64 = image_base64.split('base64,')[1]
        
        # Decode base64 to image
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_bytes))
        print("Image processed successfully")
        
        # Preprocess image
        # Resize and normalize
        image = image.resize((224, 224))  # Adjust size to match model input
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        
        # Predict
        prediction = model.predict(image_array)
        stage = np.argmax(prediction)
        
        print(f"Prediction result: {stage}")
        
        return jsonify({
            'stage': int(stage),
            'confidence': float(np.max(prediction))
        })
    
    except Exception as e:
        # Print full error details
        print("Prediction Error:")
        print(traceback.format_exc())
        
        return jsonify({
            'error': str(e),
            'details': traceback.format_exc()
        }), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)