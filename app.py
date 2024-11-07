import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Define the model paths
model_save_path = 'C:/Users/ramakrishna/OneDrive/Desktop/Agros_Flask'
crop_model_path = os.path.join(model_save_path, 'crop_model.pkl')
fertilizer_model_path = os.path.join(model_save_path, 'fertilizer_model.pkl')
label_encoder_path = os.path.join(model_save_path, 'label_encoder.pkl')

# Load the models and label encoder
crop_model = joblib.load(crop_model_path)
fertilizer_model = joblib.load(fertilizer_model_path)
label_encoder = joblib.load(label_encoder_path)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/prediction', methods=['POST'])
def prediction():
    Nitrogen = request.form['Nitrogen']
    Phosphorus = request.form['Phosphorus']
    Potassium = request.form['Potassium']
    
    # Render the prediction form with the values
    return render_template('prediction.html', Nitrogen=Nitrogen, Phosphorus=Phosphorus, Potassium=Potassium)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the form data
    Nitrogen = float(request.form['Nitrogen'])
    Phosphorus = float(request.form['Phosphorus'])
    Potassium = float(request.form['Potassium'])
    Temperature = float(request.form['Temperature'])
    Humidity = float(request.form['Humidity'])
    ph = float(request.form['ph'])
    Rainfall = float(request.form['Rainfall'])
    Soil_Type = request.form['Soil_Type']
    Moisture = float(request.form['Moisture'])

    # Prepare input data for crop prediction
    crop_features = pd.DataFrame({
        'Nitrogen': [Nitrogen],
        'Phosphorus': [Phosphorus],
        'Potassium': [Potassium],
        'Temperature': [Temperature],
        'Humidity': [Humidity],
        'ph': [ph],
        'Rainfall': [Rainfall]
    })

    # Prepare input data for fertilizer prediction
    fertilizer_features = pd.DataFrame({
        'Nitrogen': [Nitrogen],
        'Phosphorus': [Phosphorus],
        'Potassium': [Potassium],
        'Soil_Type': [Soil_Type],
        'Moisture': [Moisture]
    })

    # Make predictions
    predicted_crop = crop_model.predict(crop_features)
    predicted_fertilizer = fertilizer_model.predict(fertilizer_features)

    # Decode predictions
    decoded_crop = label_encoder.inverse_transform(predicted_crop)
    decoded_fertilizer = predicted_fertilizer[0]  # Assuming it's already in the right format

    # Render the results page with predictions
    return render_template('results.html', predicted_crop=decoded_crop[0], predicted_fertilizer=decoded_fertilizer)

# New endpoint for crop information
@app.route('/api/crop_info/<crop>', methods=['GET'])
def get_crop_info(crop):
    return jsonify({'crop': crop, 'info': 'No information available for this crop.'})

# New endpoint for fertilizer information
@app.route('/api/fertilizer_info/<fertilizer>', methods=['GET'])
def get_fertilizer_info(fertilizer):
    return jsonify({'fertilizer': fertilizer, 'info': 'No information available for this fertilizer.'})

if __name__ == '__main__':
    app.run(debug=True)