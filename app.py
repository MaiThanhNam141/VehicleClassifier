from flask import Flask, render_template, request
from skimage.transform import resize
import numpy as np
import pickle
import requests
from PIL import Image
from io import BytesIO
import base64

app = Flask(__name__)

# Kiểm tra nếu model.pkl tồn tại
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

Categories = ['Xe buýt', 'Máy bay', 'Xe hơi', 'Xe đạp', 'Xe máy']

@app.route('/')
def index():
    return render_template('index.html', show_loading=request.referrer == request.url)

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return render_template('error.html', error_message="Model is not available.")
    
    url = request.form['image_url']
    
    try:
        if url.startswith('data:image'):
            _, encoded_image = url.split(',', 1)
            decoded_image = base64.b64decode(encoded_image)
            img = Image.open(BytesIO(decoded_image))
        else:
            response = requests.get(url)
            img = Image.open(BytesIO(response.content))
        
        # Chuyển đổi sang RGB (nếu cần)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img_array = np.array(img)
        img_resized = resize(img_array, (100, 100, 3))
        img_flattened = img_resized.flatten()
        img_to_predict = np.array([img_flattened])
        
        predicted_class = model.predict(img_to_predict)
        probability = model.predict_proba(img_to_predict)
        predicted_class_name = Categories[predicted_class[0]]
        probabilities_list = [(Categories[i], probability[0][i]) for i in range(len(Categories))]
        accuracy_ratio = probability[0][predicted_class[0]]
        
        return render_template('result.html', predicted_category=predicted_class_name, img_url=url, accuracies=probabilities_list, correct=accuracy_ratio)
    
    except Exception as e:
        return render_template('error.html', error_message=f"Error processing image: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
