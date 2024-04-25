from flask import Flask, render_template, request
from skimage.transform import resize
import numpy as np
import pickle
import requests
from PIL import Image
from io import BytesIO
import base64

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
Categories = ['Nấm', 'Bình thường', 'Mụn', 'Vẩy nến']

@app.route('/')
def index():
    return render_template('index.html', show_loading=request.referrer == request.url)

@app.route('/predict', methods=['POST'])
def predict():
    url = request.form['image_url']

    try:
        if url.startswith('data:image'):
            # Extract base64 image data
            _, encoded_image = url.split(',', 1)
            decoded_image = base64.b64decode(encoded_image)

            # Load image using PIL
            img = Image.open(BytesIO(decoded_image))
        else:
            # Load image from URL
            response = requests.get(url)
            img = Image.open(BytesIO(response.content))

        # Convert image to numpy array
        img_array = np.array(img)

        # Resize the image
        img_resized = resize(img_array, (100, 100, 2))

        # Flatten the image array
        img_flattened = img_resized.flatten()

        img_to_predict = np.array([img_flattened])

        predicted_class = model.predict(img_to_predict)
        probability = model.predict_proba(img_to_predict)

        # Make prediction
        predicted_class_name = Categories[predicted_class[0]]
        probabilities_list = [(Categories[i], probability[0][i]) for i in range(len(Categories))]
        accuracy_ratio = probability[0][predicted_class[0]]
        
        return render_template('result.html', predicted_category=predicted_class_name, img_url=url, accuracies=probabilities_list, correct=accuracy_ratio)

    except Exception as e:
        error_message = f"Error processing image: {str(e)}"
        return render_template('error.html', error_message=error_message)

if __name__ == '__main__':
    app.run(debug=True)
