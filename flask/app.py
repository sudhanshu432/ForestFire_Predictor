import pickle
from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

# Import ridge regressor model and standard scaler pickle
ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))

# Route for home page
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/Predictor')
def Predictor():
    return render_template('Predictor.html')

# Route to handle prediction
@app.route('/predictdata', methods=['POST'])
def predict_datapoint():
    if request.method == 'POST':
        # Collect input values
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        # Standardize the data using the loaded scaler
        new_data_scaled = standard_scaler.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])

        # Predict the result using the loaded ridge regression model
        result = ridge_model.predict(new_data_scaled)

        # Pass the result to the template for rendering in the popup
        return render_template('Predictor.html', result=result[0])

    return render_template('Predictor.html')

# Additional routes
@app.route('/ProjectFlow')
def ProjectFlow():
    return render_template('ProjectFlow.html')

@app.route('/resources')
def resources():
    return render_template('resources.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/feedback')
def feedback():
    return render_template('feedback.html')

# Run the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0")
