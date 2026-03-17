from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load your trained model
with open('best_sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return "Sentiment Analysis API is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '')
    
    # Make sure to preprocess 'text' exactly how you did in your NLP.ipynb
    prediction = model.predict([text]) 
    
    return jsonify({'sentiment': str(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
