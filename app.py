from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

new_df = pd.read_csv('static/Data_change.csv')

with open("models/kmeans_model.pkl", "rb") as f:
    kmeans = pickle.load(f)

with open("models/standardscaler.pkl", "rb") as f:
    std = pickle.load(f)

df = pd.read_csv('models/filtering_data.csv')

item_images = {
    0: [{'name': 'Pigeonpeas (अरहर/तूर दाल)', 'image': 'images/pigeonpeas.jpg'},{'name': 'Moth Beans (मोठ/मोठ दाल)', 'image': 'images/mothbeans.jpg'},
        {'name': 'Mung Bean (मूंग)', 'image': 'images/mungbean.jpg'},{'name': 'Black Gram (उड़द दाल)', 'image': 'images/blackgram.jpg'},
        {'name': 'Lentil (मसूर दाल)', 'image': 'images/lentil.jpg'},{'name': 'Mango (आम)', 'image': 'images/mango.jpg'},
        {'name': 'Orange (संतरा)', 'image': 'images/orange.jpg'},{'name': 'Papaya (पपीता)', 'image': 'images/papaya.jpg'}],
    1: [{'name': 'Maize (मक्का/भुट्टा)', 'image': 'images/maize.jpg'},{'name': 'Lentil (मसूर दाल)', 'image': 'images/lentil.jpg'},
        {'name': 'Banana (केला)', 'image': 'images/banana.jpg'},{'name': 'Papaya (पपीता)', 'image': 'images/papaya.jpg'},
        {'name': 'Coconut (नारियल)', 'image': 'images/coconut.jpg'},{'name': 'Cotton (कपास)', 'image': 'images/cotton.jpg'},
        {'name': 'Jute (पटसन/जूट)', 'image': 'images/jute.jpg'},{'name': 'Coffee (कॉफी)', 'image': 'images/coffee.jpg'}],
    2: [{'name': 'Grapes (अंगूर)', 'image': 'images/grapes.jpg'},{'name': 'Apple (सेब)', 'image': 'images/apple.jpg'}],
    3: [{'name': 'Pigeonpeas (अरहर/तूर दाल)', 'image': 'images/pigeonpeas.jpg'},{'name': 'Pomegranate (अनार)', 'image': 'images/pomegranate.jpg'},
        {'name': 'Orange (संतरा)', 'image': 'images/orange.jpg'},{'name': 'Papaya (पपीता)', 'image': 'images/papaya.jpg'},
        {'name': 'Coconut (नारियल)', 'image': 'images/coconut.jpg'}],
    4: [{'name': 'Rice (चावल)', 'image': 'images/rice.jpg'},{'name': 'Pigeonpeas (अरहर/तूर दाल)', 'image': 'images/pigeonpeas.jpg'},
        {'name': 'Papaya (पपीता)', 'image': 'images/papaya.jpg'},{'name': 'Coconut (नारियल)', 'image': 'images/coconut.jpg'},
        {'name': 'Jute (पटसन/जूट)', 'image': 'images/jute.jpg'},{'name': 'Coffee (कॉफी)', 'image': 'images/coffee.jpg'}],
    5: [{'name': 'Pigeonpeas (अरहर/तूर दाल)', 'image': 'images/pigeonpeas.jpg'},{'name': 'Moth Beans (मोठ/मोठ दाल)', 'image': 'images/mothbeans.jpg'},
        {'name': 'Lentil (मसूर दाल)', 'image': 'images/lentil.jpg'},{'name': 'Mango (आम)', 'image': 'images/mango.jpg'}],
    6: [{'name': 'Watermelon (तरबूज)', 'image': 'images/watermelon.jpg'},{'name': 'Muskmelon (खरबूजा)', 'image': 'images/muskmelon.jpg'}],
    7: [{'name': 'Chickpea (Chickpea)', 'image': 'images/chickpea.jpg'},{'name': 'Kidney Beans (राजमा)', 'image': 'images/kidneybeans.jpg'},
        {'name': 'Pigeonpeas (अरहर/तूर दाल)', 'image': 'images/pigeonpeas.jpg'},{'name': 'Lentil (मसूर दाल)', 'image': 'images/lentil.jpg'}]
}

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
    states = new_df['State'].dropna().unique().tolist()
    return render_template('crop_price_predict.html', states=states)

@app.route('/get_options', methods=['POST'])
def get_options():
    """Dynamically returns dropdown values based on user selection"""
    data = request.json
    state = data.get('state')

    if not state:
        return jsonify({'error': 'State is required'}), 400

    filtered_df = new_df[new_df['State'] == state]

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
    
@app.route('/crop_recommendation')
def crop_recommendation():
    return render_template('crop_recommendation.html')

@app.route('/crop_recommendation_output')
def crop_recommendation_output():
    return render_template('crop_recommendation_output.html')

@app.route('/submit', methods=['POST'])
def submit():
    # Retrieve form data
    n = float(request.form.get('n'))
    p = float(request.form.get('p'))
    k = float(request.form.get('k'))
    temperature = float(request.form.get('temperature'))
    humidity = float(request.form.get('humidity'))
    ph = float(request.form.get('ph'))
    rainfall = float(request.form.get('rainfall'))

    # Prepare input data for prediction
    input_data = np.array([[n, p, k, temperature, humidity, ph, rainfall]])

    # Transform the input data
    transformed_input_data = std.transform(input_data)

    # Make prediction using the KMeans model
    cluster = kmeans.predict(transformed_input_data)[0]

    crops = item_images.get(cluster, [{'name': 'Unknown', 'image': 'images/default.jpg'}] * 5)

    # Pass the predicted cluster to the output page
    return render_template('crop_recommendation_output.html', crops=crops)

if __name__ == '__main__':
    app.run(debug=True)
