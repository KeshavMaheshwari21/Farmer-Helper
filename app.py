from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
import pickle
import os
import json
import numpy as np
import tensorflow as tf
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv
import re
import requests
from flask import send_file
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import time

app = Flask(__name__)

OPENWEATHER_API_KEY = "7a91ad5c225eee9832f9efc6a1812fe6"

new_df = pd.read_csv('models/Data_change.csv')

with open("models/kmeans_model.pkl", "rb") as f:
    kmeans = pickle.load(f)

with open("models/standardscaler.pkl", "rb") as f:
    std = pickle.load(f)

df = pd.read_csv('models/filtering_data.csv')

item_images = {
    0: [{'name': 'Pigeonpeas (अरहर/तूर दाल)', 'image': 'images/pigeonpeas.jpg'},{'name': 'Moth Beans (मोठ/मोठ दाल)', 'image': 'images/mothbeans.jpg'},
        {'name': 'Mung Bean (मूंग)', 'image': 'images/mungbean.jpg'},{'name': 'Black Gram (उड़द दाल)', 'image': 'images/blackgram.jpg'},
        {'name': 'Lentil (मसूर दाल)', 'image': 'images/lentil.jpg'},{'name': 'Mango (आम)', 'image': 'images/mango.jpg'},
        {'name': 'Orange (संतरा)', 'image': 'images/orange.jpg'},{'name': 'Papaya (पपीता)', 'image': 'images/papaya.jpeg'}],
    1: [{'name': 'Maize (मक्का/भुट्टा)', 'image': 'images/maize.jpg'},{'name': 'Lentil (मसूर दाल)', 'image': 'images/lentil.jpg'},
        {'name': 'Banana (केला)', 'image': 'images/banana.jpg'},{'name': 'Papaya (पपीता)', 'image': 'images/papaya.jpeg'},
        {'name': 'Coconut (नारियल)', 'image': 'images/coconut.jpg'},{'name': 'Cotton (कपास)', 'image': 'images/cotton.jpg'},
        {'name': 'Jute (पटसन/जूट)', 'image': 'images/jute.jpg'},{'name': 'Coffee (कॉफी)', 'image': 'images/coffee.jpeg'}],
    2: [{'name': 'Grapes (अंगूर)', 'image': 'images/grapes.jpg'},{'name': 'Apple (सेब)', 'image': 'images/apple.jpg'}],
    3: [{'name': 'Pigeonpeas (अरहर/तूर दाल)', 'image': 'images/pigeonpeas.jpg'},{'name': 'Pomegranate (अनार)', 'image': 'images/pomegranate.jpg'},
        {'name': 'Orange (संतरा)', 'image': 'images/orange.jpg'},{'name': 'Papaya (पपीता)', 'image': 'images/papaya.jpeg'},
        {'name': 'Coconut (नारियल)', 'image': 'images/coconut.jpg'}],
    4: [{'name': 'Rice (चावल)', 'image': 'images/rice.jpeg'},{'name': 'Pigeonpeas (अरहर/तूर दाल)', 'image': 'images/pigeonpeas.jpg'},
        {'name': 'Papaya (पपीता)', 'image': 'images/papaya.jpeg'},{'name': 'Coconut (नारियल)', 'image': 'images/coconut.jpg'},
        {'name': 'Jute (पटसन/जूट)', 'image': 'images/jute.jpg'},{'name': 'Coffee (कॉफी)', 'image': 'images/coffee.jpeg'}],
    5: [{'name': 'Pigeonpeas (अरहर/तूर दाल)', 'image': 'images/pigeonpeas.jpg'},{'name': 'Moth Beans (मोठ/मोठ दाल)', 'image': 'images/mothbeans.jpg'},
        {'name': 'Lentil (मसूर दाल)', 'image': 'images/lentil.jpg'},{'name': 'Mango (आम)', 'image': 'images/mango.jpg'}],
    6: [{'name': 'Watermelon (तरबूज)', 'image': 'images/watermelon.jpg'},{'name': 'Muskmelon (खरबूजा)', 'image': 'images/muskmelon.jpg'}],
    7: [{'name': 'Chickpea (चना)', 'image': 'images/chickpea.jpg'},{'name': 'Kidney Beans (राजमा)', 'image': 'images/kidneybeans.jpg'},
        {'name': 'Pigeonpeas (अरहर/तूर दाल)', 'image': 'images/pigeonpeas.jpg'},{'name': 'Lentil (मसूर दाल)', 'image': 'images/lentil.jpg'}]
}

item_text = {
    0: 'Pigeonpeas अरहर तूर दाल Moth Beans मोठ  दाल Mung Bean मूंग Black Gram उड़द दाल Lentil मसूर दाल Mango आम Orange संतरा Papaya पपीता',
    1: 'Maize मक्का/भुट्टा Lentil मसूर दाल Banana केला Papaya पपीता Coconut नारियल Cotton कपास Jute पटसन जूट Coffee कॉफी',
    2: 'Grapes अंगूर Apple सेब',
    3: 'Pigeonpeas अरहर तूर दाल Pomegranate अनार Orange संतरा Papaya पपीता Coconut नारियल',
    4: 'Rice चावल Pigeonpeas अरहर तूर दाल Papaya पपीता Coconut नारियल Jute पटसन जूट Coffee कॉफी',
    5: 'Pigeonpeas अरहर तूर दाल Moth Beans मोठ दाल Lentil मसूर दाल Mango आम',
    6: 'Watermelon तरबूज Muskmelon खरबूजा',
    7: 'Chickpea चना Kidney Beans राजमा Pigeonpeas अरहर तूर दाल Lentil मसूर दाल'
}
# Function to load ML model
def load_model():
    with open("models/crop_price_model.pkl", "rb") as model_file:
        return pickle.load(model_file)

# Function to load encoder
def load_encoder():
    with open("models/crop_price_encoder.pkl", "rb") as encoder_file:
        return pickle.load(encoder_file)

# Load the trained model and encoder
price_model = load_model()
price_encoder = load_encoder()

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
        new_data_encoded = price_encoder.transform(new_data)
        predicted_price = price_model.predict(new_data_encoded)[0]
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

    text = item_text.get(cluster)

    # Pass the predicted cluster to the output page
    return render_template('crop_recommendation_output.html', crops=crops , text = text)

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=API_KEY)

# Initialize Flask
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the pre-trained model
model_path = "models/plant_disease_prediction_model.h5"
disease_model = tf.keras.models.load_model(model_path)

# Load class indices
with open("models/class_indices.json", "r") as f:
    class_indices = json.load(f)
class_indices = {int(k): v for k, v in class_indices.items()}

# Function to preprocess the image
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to predict plant disease
def predict_image_class(image_path):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = disease_model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[predicted_class_index]
    return predicted_class_name



# Function to generate precautions using Gemini AI with error handling

def get_precautions(disease):
    prompt = f"""
    Provide a detailed explanation and precautions for the plant disease '{disease}' in both English and Hindi.

    **Format the response exactly like this:**
    
    Disease in English: <disease_name_english>
    Disease in Hindi: <disease_name_hindi>
    Explanation in English: <explanation_english>
    Explanation in Hindi: <explanation_hindi>
    Precautions in English: <precautions_english>
    Precautions in Hindi: <precautions_hindi>
    """

    try:
        model_gemini = genai.GenerativeModel("gemini-1.5-flash")
        response = model_gemini.generate_content(prompt)

        if not response or not response.text:
            raise ValueError("Empty response from Gemini AI.")

        # Debug: Print full response
        print("\n=== Raw Gemini Response ===\n")
        print(response.text)
        print("\n===========================\n")

        # Initialize keys
        details = {
            "disease_en": disease,
            "disease_hi": "अज्ञात रोग",
            "explanation_en": "",
            "explanation_hi": "",
            "precautions_en": "",
            "precautions_hi": "",
        }

        # Parsing logic for multi-line values
        current_key = None
        for line in response.text.split("\n"):
            line = line.strip()
            if line.lower().startswith("disease in english:"):
                details["disease_en"] = line.split(":", 1)[1].strip()
                current_key = None
            elif line.lower().startswith("disease in hindi:"):
                details["disease_hi"] = line.split(":", 1)[1].strip()
                current_key = None
            elif line.lower().startswith("explanation in english:"):
                details["explanation_en"] = line.split(":", 1)[1].strip()
                current_key = "explanation_en"
            elif line.lower().startswith("explanation in hindi:"):
                details["explanation_hi"] = line.split(":", 1)[1].strip()
                current_key = "explanation_hi"
            elif line.lower().startswith("precautions in english:"):
                details["precautions_en"] = line.split(":", 1)[1].strip()
                current_key = "precautions_en"
            elif line.lower().startswith("precautions in hindi:"):
                details["precautions_hi"] = line.split(":", 1)[1].strip()
                current_key = "precautions_hi"
            elif current_key:
                details[current_key] += " " + line.strip()

        return details

    except Exception as e:
        print("Error generating content from Gemini AI:", e)
        return {
            "disease_en": disease,
            "disease_hi": "अनुमानित रोग",
            "explanation_en": "No explanation available.",
            "explanation_hi": "इस समय कोई विवरण उपलब्ध नहीं।",
            "precautions_en": "No precautions available.",
            "precautions_hi": "इस समय कोई सावधानियाँ उपलब्ध नहीं हैं।",
        }


# Flask Routes
app.config["UPLOAD_FOLDER"] = "static/uploads"

def generate_pdf(image_path, details, pdf_filename):
    pdf_path = os.path.join(app.config["UPLOAD_FOLDER"], pdf_filename)

    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter

    # Add image
    img = Image.open(image_path)
    img_width, img_height = img.size
    aspect = img_height / img_width

    max_width = 400  # Max width for image
    max_height = max_width * aspect  # Maintain aspect ratio

    img_x = (width - max_width) / 2
    img_y = height - max_height - 50  # Leave some space at the top
    c.drawImage(image_path, img_x, img_y, max_width, max_height)

    # Add text
    text_y = img_y - 30
    c.setFont("Helvetica-Bold", 14)
    c.drawString(100, text_y, f"Disease: {details['disease_en']} ({details['disease_hi']})")

    text_y -= 20
    c.setFont("Helvetica", 12)
    c.drawString(100, text_y, "Explanation (English):")
    text_y -= 15
    c.setFont("Helvetica", 10)
    for line in details["explanation_en"].split(". "):
        c.drawString(100, text_y, line)
        text_y -= 15

    text_y -= 10
    c.setFont("Helvetica", 12)
    c.drawString(100, text_y, "Explanation (Hindi):")
    text_y -= 15
    c.setFont("Helvetica", 10)
    for line in details["explanation_hi"].split(". "):
        c.drawString(100, text_y, line)
        text_y -= 15

    text_y -= 10
    c.setFont("Helvetica", 12)
    c.drawString(100, text_y, "Precautions (English):")
    text_y -= 15
    c.setFont("Helvetica", 10)
    for line in details["precautions_en"].split(". "):
        c.drawString(100, text_y, line)
        text_y -= 15

    text_y -= 10
    c.setFont("Helvetica", 12)
    c.drawString(100, text_y, "Precautions (Hindi):")
    text_y -= 15
    c.setFont("Helvetica", 10)
    for line in details["precautions_hi"].split(". "):
        c.drawString(100, text_y, line)
        text_y -= 15

    c.save()
    return pdf_path

@app.route("/plant_disease_output", methods=["GET", "POST"])
def plant_disease_output():
    if request.method == "POST":
        if "image" not in request.files:
            return render_template("index.html", error="No file uploaded")

        file = request.files["image"]
        if file.filename == "":
            return render_template("index.html", error="No file selected")

        # Save uploaded file
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        # Predict disease
        predicted_disease = predict_image_class(filepath)

        # Get precautions and related details
        details = get_precautions(predicted_disease)

        # Generate PDF
        pdf_filename = f"{predicted_disease}_report.pdf"
        pdf_path = generate_pdf(filepath, details, pdf_filename)

        return render_template(
            "plant_disease_output.html",
            image=filepath,
            disease_en=details["disease_en"],
            disease_hi=details["disease_hi"],
            explanation_en=details["explanation_en"],
            explanation_hi=details["explanation_hi"],
            precautions_en=details["precautions_en"],
            precautions_hi=details["precautions_hi"],
            pdf_path=pdf_filename,  # Pass the PDF filename to the template
        )

    return render_template("plant_disease.html")

@app.route("/download/<filename>")
def download_file(filename):
    return send_file(os.path.join(app.config["UPLOAD_FOLDER"], filename), as_attachment=True)

@app.route('/weather_forecast')
def weather_forecast():
    return render_template('weather_forecast.html')

@app.route('/plant_disease')
def plant_disease():
    return render_template('plant_disease.html')

@app.route('/developers')
def developers():
    return render_template('developers.html')

@app.route('/getWeather', methods=['GET'])
def get_weather():
    lat = request.args.get('lat')
    lon = request.args.get('lon')

    if not lat or not lon:
        return jsonify({"error": "Missing latitude or longitude"}), 400

    weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&units=metric&appid={OPENWEATHER_API_KEY}"
    forecast_url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&units=metric&appid={OPENWEATHER_API_KEY}"

    try:
        weather_response = requests.get(weather_url)
        forecast_response = requests.get(forecast_url)

        weather_data = weather_response.json()
        forecast_data = forecast_response.json()

        if weather_response.status_code != 200 or forecast_response.status_code != 200:
            return jsonify({"error": "Failed to fetch weather data"}), 500

        forecast_list = []
        unique_dates = set()
        for forecast in forecast_data["list"]:
            date = forecast["dt_txt"].split(" ")[0]
            if date not in unique_dates:
                unique_dates.add(date)
                forecast_list.append({
                    "date": date,
                    "temp": forecast["main"]["temp"],
                    "humidity": forecast["main"]["humidity"],
                    "rain": forecast.get("rain", {}).get("3h", 0),  # Rainfall in last 3 hours
                    "wind_speed": forecast["wind"]["speed"],
                    "weather": forecast["weather"][0]["description"].capitalize(),
                    "icon": forecast["weather"][0]["icon"]
                })
            if len(forecast_list) == 7:
                break

        return jsonify({
            "city": weather_data.get("name", "Unknown Location"),
            "temperature": weather_data["main"]["temp"],
            "weather": weather_data["weather"][0]["description"].capitalize(),
            "wind_speed": weather_data["wind"]["speed"],
            "humidity": weather_data["main"]["humidity"],
            "forecast": forecast_list        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
genai.configure(api_key="AIzaSyCdHvM6djoFfXcAHcfxrPH5on6d7fZ5cqA")  # Replace with your actual API key

model = genai.GenerativeModel("gemini-1.5-flash")

conversation_history = []
last_message_time = time.time()

@app.route("/chat", methods=["POST"])
def chat():
    global conversation_history, last_message_time

    user_message = request.json.get("message", "").strip()
    
    if not user_message:
        return jsonify({"reply": "Please enter a message."})

    # Auto-clear history if inactive for 5 minutes (300 seconds)
    current_time = time.time()
    if current_time - last_message_time > 300:
        conversation_history.clear()

    last_message_time = current_time  # Update timestamp

    # Add user message to conversation history
    conversation_history.append({"role": "user", "content": user_message})

    # Define the prompt with history
    prompt = f"Conversation history: {conversation_history}. Reply briefly in the same tone and language as '{user_message}'. If explanation is needed, keep it short and to point and related to farming. If unrelated, respond with 'Please enter a farming-related prompt.'"

    try:
        # Call Gemini model to generate response
        response = model.generate_content(prompt)

        if response and hasattr(response, "text"):
            bot_reply = response.text.strip()
        else:
            bot_reply = "Sorry, I couldn't understand that."

        # Add bot response to conversation history
        conversation_history.append({"role": "assistant", "content": bot_reply})

        return jsonify({"reply": bot_reply})
    
    except Exception as e:
        print(f"Error: {e}")  # Log the error for debugging
        return jsonify({"reply": "Error processing request."})

if __name__ == '__main__':
    app.run(debug=True)
