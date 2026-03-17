from flask import Flask, request, jsonify, render_template
import pickle
import traceback
import os

app = Flask(__name__)

# Test model loading with error handling
try:
    model_path = os.path.join(os.path.dirname(__file__), 'best_sentiment_model.pkl')
    print(f"Loading model from: {model_path}")
    print(f"File exists: {os.path.exists(model_path)}")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/test-model')
def test_model():
    if model is None:
        return "❌ Model failed to load. Check terminal."
    
    try:
        test_text = "I love this app"
        prediction = model.predict([test_text])
        return f"✅ Model works! Test prediction: {prediction[0]}"
    except Exception as e:
        return f"❌ Prediction failed: {str(e)}<br><pre>{traceback.format_exc()}</pre>"

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        text = data.get('text', '')
        print(f"Received text: {text[:50]}...")  # Debug log
        
        prediction = model.predict([text])
        print(f"Prediction: {prediction[0]}")  # Debug log
        
        return jsonify({'sentiment': str(prediction[0])})
    except Exception as e:
        print(f"Predict error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
