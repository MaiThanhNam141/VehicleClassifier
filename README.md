# VehicleClassifier

VehicleClassifier is a web-based image classification system that uses a Support Vector Machine (SVM) model to predict different types of vehicles. The project is built with Flask for the web interface and utilizes `skimage` and `scikit-learn` for image processing and machine learning.

## 🚀 Features
- Upload or provide a URL for an image.
- Classify vehicles into categories: `Bus`, `Plane`, `Car`, `Bicycle`, `Motorcycle`.
- Display classification results with accuracy percentages.
- Visualize prediction probabilities using pie and bar charts.
- Allow users to provide feedback and improve the model.

## 🛠️ Technologies Used
- **Python** (Flask, scikit-learn, skimage, numpy, pandas, pickle)
- **Frontend** (HTML, CSS, JavaScript, Chart.js)
- **Machine Learning** (Support Vector Machine with GridSearchCV)

## 📂 Project Structure
```
VehicleClassifier/
│── static/
│   ├── images/             # Training images
│   ├── style.css           # CSS styles
│   ├── loading.gif         # Loading animation
│── templates/
|	├── error.html			# Log the error to the screen
│   ├── index.html          # Main page for image upload
│   ├── result.html         # Results display page
│── app.py                  # Flask web server
│── VehicleClassifier.py    # Model training and saving script
│── model.pkl               # Trained SVM model
│── README.md               # Project documentation
│── requirements.txt        # Dependencies list
```

## 📥 Installation
1. Clone the repository:
   ```bash
   git clone this repo
   cd VehicleClassifier
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Train the model (optional if `model.pkl` exists):
   ```bash
   python3 VehicleClassifier.py
   ```
4. Run the Flask application:
   ```bash
   python3 app.py
   ```
5. Open your browser and go to:
   ```
   http://127.0.0.1:5000
   ```

## 📸 How It Works
1. Upload an image or enter an image URL.
2. Click `Submit` to classify the image.
3. View the prediction results along with pie and bar charts displaying confidence levels.
4. If incorrect, provide feedback to improve the model.

## 📜 License
This project is open-source and available under the MIT License.