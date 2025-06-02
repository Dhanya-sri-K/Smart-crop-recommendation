from flask import Flask, render_template, request, jsonify
import base64
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import pandas as pd
from geopy.geocoders import Nominatim
import joblib
import json
import os
from datetime import datetime
import requests
import pickle
import sklearn
import warnings
import asyncio
from gemini_helper import scrape_crop_details
import sys
warnings.filterwarnings('ignore')


app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
OPENWEATHER_API_KEY = '641e6ac4a1c179ba87fea73b4e3ba8b7'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Soil type mapping
SOIL_TYPES = {
    0: "Alluvial",
    1: "Black",
    2: "Clayey",
    3: "Latterite",
    4: "Red",
    5: "Sandy"
}

def get_state_code(state):
    """Convert state name to state code"""
    state_codes = {
        "Andhra Pradesh": 1,
        "Arunachal Pradesh": 2,
        "Assam": 3,
        "Bihar": 4,
        "Chhattisgarh": 5,
        "Goa": 6,
        "Gujarat": 7,
        "Haryana": 8,
        "Himachal Pradesh": 9,
        "Jharkhand": 10,
        "Karnataka": 11,
        "Kerala": 12,
        "Madhya Pradesh": 13,
        "Maharashtra": 14,
        "Manipur": 15,
        "Meghalaya": 16,
        "Mizoram": 17,
        "Nagaland": 18,
        "Odisha": 19,
        "Punjab": 20,
        "Rajasthan": 21,
        "Sikkim": 22,
        "Tamil Nadu": 23,
        "Telangana": 24,
        "Tripura": 25,
        "Uttar Pradesh": 26,
        "Uttarakhand": 27,
        "West Bengal": 28,
        "Andaman and Nicobar Islands": 29,
        "Dadra and Nagar Haveli and Daman and Diu": 30,
        "Chandigarh": 31,
        "Delhi": 32,
        "Jammu and Kashmir": 33,
        "Lakshadweep": 34,
        "Puducherry": 35,
        "Ladakh": 36
    }
    return state_codes.get(state, 0)

def get_season(month):
    """Determine season based on month"""
    if month in [11, 12, 1, 2]:
        return 2  # Winter
    elif month in [6, 7, 8, 9]:
        return 1  # Monsoon
    elif month in [3, 4]:
        return 3  # Summer
    return 4  # Post-Monsoon

def load_model_with_version_check(model_path):
    """Load a scikit-learn model with version compatibility handling"""
    try:
        return joblib.load(model_path)
    except ValueError as e:
        print(f"Initial loading failed: {e}")
        try:
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        except Exception as pickle_error:
            print(f"Pickle loading failed: {pickle_error}")
            try:
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    if hasattr(model_data, '_sklearn_version'):
                        print(f"Model was saved with scikit-learn version: {model_data._sklearn_version}")
                    return model_data
            except Exception as final_error:
                raise Exception(f"Failed to load model with all methods: {final_error}")

# Load models at startup
try:
    print("Loading models...")
    SOIL_MODEL_PATH = 'D:\\COLLEGE\\Sem-3\\AI\\project\\my_model.h5'
    soil_model = load_model(SOIL_MODEL_PATH)
    
    CROP_MODEL_PATH = "finalized_model1.sav"
    crop_recommendation_model = load_model_with_version_check(CROP_MODEL_PATH)
    print("Models loaded successfully")
except Exception as e:
    print(f"Error loading models: {e}")
    raise

def predict_soil_type(image_path):
    """Predict soil type using the loaded .h5 model"""
    try:
        img = image.load_img(image_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x/255.0
        
        prediction = soil_model.predict(x)
        soil_type_idx = np.argmax(prediction)
        confidence = float(prediction[0][soil_type_idx] * 100)
        
        print(f"Soil prediction vector: {prediction}")
        print(f"Predicted soil type: {SOIL_TYPES[soil_type_idx]} with {confidence:.2f}% confidence")
        
        return {
            'soil_type': SOIL_TYPES[soil_type_idx],
            'confidence': confidence,
            'soil_type_code': soil_type_idx + 1
        }
    except Exception as e:
        print(f"Error in soil prediction: {e}")
        return None

import requests

import requests
from datetime import datetime, timedelta

def get_weather_and_rain_data(lat, lon):
    """Fetch current weather and daily/monthly rainfall data from OpenWeather API."""
    current_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
    forecast_url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
    
    try:
        current_response = requests.get(current_url)
        current_response.raise_for_status()
        current_data = current_response.json()
        
        forecast_response = requests.get(forecast_url)
        forecast_response.raise_for_status()
        forecast_data = forecast_response.json()
        
        

        # Get current rainfall if available
        total_rain = 0
        if 'rain' in current_data:
            total_rain += current_data['rain'].get('1h', 0) or current_data['rain'].get('3h', 0)

        # Accumulate rainfall for the day from forecast data
        daily_rainfall = 0
        today = datetime.utcnow().date()
        for entry in forecast_data['list']:
            timestamp = datetime.utcfromtimestamp(entry['dt'])
            if timestamp.date() == today and 'rain' in entry:
                daily_rainfall += entry['rain'].get('3h', 0)
        
        # Use forecasted daily rainfall if current rain is not available
        total_rain = total_rain if total_rain > 0 else daily_rainfall
        print(total_rain)
        print(daily_rainfall)
       
        
        return {
            'temperature': current_data['main']['temp'],
            'humidity': current_data['main']['humidity'],
            'description': current_data['weather'][0]['description'],
            'rainfall': total_rain *1000    # Rainfall in mm per day
        }

    except requests.RequestException as e:
        print(f"Error fetching data: {e}")
        return None


def predict_crop(input_data):
    """Predict crop based on input data"""
    try:
        # Convert input data to array format
        input_values = list(input_data.values())
        input_array = np.array(input_values).reshape(1, -1)
        print(input_values)
        
        # X_new = np.array([23,145,16.67,27.5,2,2])
        # print(X_new)
        # X_new = X_new.reshape(1,-1)
        # print(X_new)
        # Make prediction
        prediction = crop_recommendation_model.predict(input_array)
        print(prediction)
        pred_crop_name = prediction[0]
        
        # Load crop information
        with open("Prediction.json") as fp:
            crop_info = json.load(fp)
        
        
        # Get detailed information for predicted crop
        crop_details = crop_info[pred_crop_name]
        
        return {
            'status': 'success',
            'crop_name': pred_crop_name,
            'details': crop_details
        }
    except Exception as e:
        print(f"Error in crop prediction: {e}")
        return {
            'status': 'error',
            'message': str(e)
        }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        soil_image = request.files['soil_image']
        coordinates = request.form['coordinates']
        date = datetime.strptime(request.form['date'], '%Y-%m-%d')
        rainfall_input_type = request.form['rainfall_input_type']
        
        # Save soil image
        image_path = os.path.join(UPLOAD_FOLDER, soil_image.filename)
        soil_image.save(image_path)
        
        # Parse coordinates
        lat, lon = map(float, coordinates.split(','))
        
        # Get location data
        locator = Nominatim(user_agent="crop_recommender")
        location = locator.reverse(f"{lat}, {lon}")
        state = location.raw['address'].get('state', '')
        
        # Get soil prediction
        soil_prediction = predict_soil_type(image_path)
        if soil_prediction is None:
            raise Exception("Failed to predict soil type")
        
        # Get rainfall data based on input type
        if rainfall_input_type == 'manual':
            try:
                weather_data = {
                    'temperature': float(request.form['temperature']),
                    'humidity': float(request.form['humidity']),
                    'description': 'Manual Input',
                    'rainfall': float(request.form['manual_rainfall'])
                }
            except ValueError:
                raise Exception("Invalid manual input values")
        else:
            # Get weather data from API
            weather_data = get_weather_and_rain_data(lat, lon)
            if weather_data is None:
                raise Exception("Failed to fetch weather data")
        
        # Get ground water data
        ground_water = float(pd.read_csv("Cat_Crop.csv").loc[
            pd.read_csv("Cat_Crop.csv")["States"]==get_state_code(state), 
            "Ground Water"
        ].iloc[0])
        
        # Prepare input data for prediction
        input_data = {
            "States": get_state_code(state),
            "Rainfall": weather_data['rainfall'],
            "Ground Water": ground_water,
            "Temperature": weather_data['temperature'],
            "Soil_type": soil_prediction['soil_type_code'],
            "Season": get_season(date.month)
        }
        
        # Get prediction
        prediction_result = predict_crop(input_data)
        if prediction_result['status'] == 'error':
            return render_template('index.html', error=prediction_result['message'])
        
        crop_name = prediction_result['details']['Crops']
        
        # Get detailed crop information from Gemini API
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        crop_details = loop.run_until_complete(scrape_crop_details(crop_name))
        loop.close()
        
        # Prepare response data
        result = {
            'crop_name': prediction_result['details']['Crops'],
            'current_weather': weather_data,
            'soil_analysis': soil_prediction,
            'details': prediction_result['details'],
            'additional_info': crop_details
        }
        
        return render_template('index.html', prediction=result)
        
    except Exception as e:
        print(f"Error in prediction route: {e}")
        return render_template('index.html', error=str(e))

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
if __name__ == "__main__":
    app.run(debug=True)