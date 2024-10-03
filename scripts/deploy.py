#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

app = Flask(__name__) #create a flask instance

# Load the trained model
model = load_model('pneumonia_detection.h5')

# Define a route for making predictions
@app.route('/predict', methods=['POST']) #define the respose port via POST method
def predict():
    try:
        # Get the uploaded image
        image = request.files['image']
        image = load_img(image, target_size=(128, 128))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)

        # Make a prediction
        prediction = model.predict(image)
        prediction = np.round(prediction).astype(int)

        # Return the prediction as JSON
        return jsonify({'prediction': prediction[0][0]})

    except Exception as e:
        return jsonify({'error': str(e)})
    
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

