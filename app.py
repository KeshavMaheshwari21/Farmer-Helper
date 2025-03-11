from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

# Load dataset once
df = pd.read_csv('static/Data_change.csv')

# Function to load ML model
def load_model():
    with open("static/crop_price_model.pkl", "rb") as model_file:
        return pickle.load(model_file)

# Function to load encoder
def load_encoder():
    with open("static/crop_price_encoder.pkl", "rb") as encoder_file:
        return pickle.load(encoder_file)

# Load the trained model and encoder
model = load_model()
encoder = load_encoder()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/crop_price_predict')
def crop_price_predict():
    states = df['State'].dropna().unique().tolist()
    return render_template('crop_price_predict.html', states=states)

@app.route('/get_options', methods=['POST'])
def get_options():
    """Dynamically returns dropdown values based on user selection"""
    data = request.json
    state = data.get('state')

    if not state:
        return jsonify({'error': 'State is required'}), 400

    filtered_df = df[df['State'] == state]

    options = {
        'districts': filtered_df['District'].dropna().unique().tolist(),
        'markets': filtered_df['Market'].dropna().unique().tolist(),
        'commodities': filtered_df['Commodity'].dropna().unique().tolist(),
        'varieties': filtered_df['Variety'].dropna().unique().tolist(),
        'grades': filtered_df['Grade'].dropna().unique().tolist(),
    }
    return jsonify(options)

@app.route('/predict', methods=['POST'])
def predict():
    """Predicts commodity price based on user input"""
    data = request.form

    required_fields = ['state', 'district', 'market', 'commodity', 'variety', 'grade']
    for field in required_fields:
        if field not in data or not data[field].strip():
            return jsonify({'error': f"{field} is required"}), 400

    new_data = pd.DataFrame({
        'State': [data['state']],
        'District': [data['district']],
        'Market': [data['market']],
        'Commodity': [data['commodity']],
        'Variety': [data['variety']],
        'Grade': [data['grade']],
    })

    # Transform the new data using the encoder
    try:
        new_data_encoded = encoder.transform(new_data)
        predicted_price = model.predict(new_data_encoded)[0]
        return jsonify({'predicted_price': f"₹{predicted_price:.2f}"})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
