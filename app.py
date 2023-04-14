from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = load_model("mnet_model.h5")

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Load image from request
        file = request.files['file']
        img = Image.open(file.stream)
        img = img.resize((96,96))
        img = np.array(img) / 255.0
        img = img.reshape(1,96,96,3)

        # Make prediction
        pred = model.predict(img)
        pred_label = np.argmax(pred, axis=1)[0]
        emotion_list = ['angry', 'happy', 'sad', 'surprise']
        emotion = emotion_list[pred_label]

        # Return prediction result
        return jsonify({'result': emotion})
    except Exception as e:
        print(str(e))
        return jsonify({'error': 'Error occurred during prediction'}), 500

if __name__ == '__main__':
    app.run(debug=True)