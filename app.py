#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# useless
import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask import Flask, request, render_template, jsonify

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model
model = tf.keras.models.load_model('./model/lstm_model.h5')

# Set max_length (based on training data padding length)
max_length = 56293   # Replace with X_train.shape[1] from training notebook

# Preprocessing function to convert audio to the required model input format
def preprocess_audio(audio_path, max_length):
    # Load the audio file
    audio, sr = librosa.load(audio_path, sr=None)

    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=15).T  # Transpose to match input format

    # Pad MFCCs to match max_length used during training
    padded_mfccs = pad_sequences([mfccs], maxlen=max_length, dtype='float32', padding='post', truncating='post')

    return padded_mfccs

# Flask route for home page
@app.route('/')
def home():
    return render_template('index.html')  # Render the HTML template

# Flask route for handling audio file upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No audio or voice file uploaded"}), 400

    # Save and preprocess the audio file
    audio = request.files['audio']
    audio_path = os.path.join("uploads", audio.filename)
    audio.save(audio_path)

    # Preprocess audio for model input
    padded_sample = preprocess_audio(audio_path, max_length)

    # Predict with the pre-trained model
    prediction = model.predict(padded_sample)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = float(prediction[0][predicted_class])

    # Interpret prediction
    result = "Fake" if predicted_class == 1 else "Real"

    # Return the result as JSON
    return jsonify({
        "prediction": result,
        "confidence": confidence
    })
if __name__ == '__main__':
    # Ensure "uploads" folder exists to store audio files temporarily
    if not os.path.exists("uploads"):
        os.makedirs("uploads")

    # Run the Flask app
    app.run(debug=True)


# In[ ]:


# Install dependencies (if not installed)
get_ipython().system('pip install flask flask-cors librosa tensorflow numpy')

# Import necessary libraries
import os
import librosa
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests (frontend & backend)

# Load the trained model from Google Drive
MODEL_PATH = "/content/drive/MyDrive/Deepfake_Voice_Detection/model/lstm_model.h5"
model = load_model(MODEL_PATH)

# Define the fixed length for MFCC extraction (same as during training)
MAX_LENGTH = 500

# Function to preprocess the audio file before prediction
def preprocess_audio(audio_path):
    try:
        # Load the audio file
        audio, sr = librosa.load(audio_path, sr=16000, duration=10)

        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)

        # Pad or truncate to fixed length
        if mfccs.shape[1] < MAX_LENGTH:
            mfccs = np.pad(mfccs, ((0, 0), (0, MAX_LENGTH - mfccs.shape[1])), mode="constant")
        else:
            mfccs = mfccs[:, :MAX_LENGTH]

        mfccs = mfccs.T  # Reshape to (500, 13)
        return np.expand_dims(mfccs, axis=0)  # Add batch dimension (1, 500, 13)

    except Exception as e:
        print(f"Error processing audio: {e}")
        return None

# API Endpoint for Prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    audio_file = request.files['audio']
    file_path = "/content/temp_audio.wav"
    audio_file.save(file_path)

    # Preprocess the audio
    processed_audio = preprocess_audio(file_path)
    if processed_audio is None:
        return jsonify({"error": "Invalid audio file"}), 400

    # Make a prediction
    prediction = model.predict(processed_audio)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = float(prediction[0][predicted_class])

    # Return prediction as JSON
    result = {
        "prediction": "Fake" if predicted_class == 1 else "Real",
        "confidence": confidence
    }
    return jsonify(result)

# Run the Flask app
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)


# In[ ]:


# Install required libraries
get_ipython().system('pip install flask flask-ngrok flask-cors librosa tensorflow numpy')

# Import necessary libraries
import os
import librosa
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask_ngrok import run_with_ngrok  # ✅ Import ngrok

# ✅ Mount Google Drive to access the trained model
from google.colab import drive
drive.mount('/content/drive')

# ✅ Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend-backend communication
run_with_ngrok(app)  # ✅ Expose Flask app online

# ✅ Load the trained model from Google Drive
MODEL_PATH = "/content/drive/MyDrive/Deepfake_Voice_Detection/model/lstm_model.h5"

if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
else:
    print("❌ Model file not found! Check Google Drive path.")

# ✅ Fixed MFCC length for consistent input shape
MAX_LENGTH = 500

# ✅ Function to preprocess audio before prediction
def preprocess_audio(audio_path):
    try:
        audio, sr = librosa.load(audio_path, sr=16000, duration=10)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)

        # Pad or truncate to MAX_LENGTH
        if mfccs.shape[1] < MAX_LENGTH:
            mfccs = np.pad(mfccs, ((0, 0), (0, MAX_LENGTH - mfccs.shape[1])), mode="constant")
        else:
            mfccs = mfccs[:, :MAX_LENGTH]

        mfccs = mfccs.T  # Reshape to (500, 13)
        return np.expand_dims(mfccs, axis=0)  # Add batch dimension

    except Exception as e:
        print(f"❌ Error processing audio: {e}")
        return None

# ✅ API Endpoint for Prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    audio_file = request.files['audio']
    file_path = "/content/temp_audio.wav"
    audio_file.save(file_path)

    processed_audio = preprocess_audio(file_path)
    if processed_audio is None:
        return jsonify({"error": "Invalid audio file"}), 400

    # ✅ Make prediction
    prediction = model.predict(processed_audio)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = float(prediction[0][predicted_class])

    return jsonify({
        "prediction": "Fake" if predicted_class == 1 else "Real",
        "confidence": confidence
    })

# ✅ Start Flask App
if __name__ == '__main__':
    app.run()


# In[ ]:


from google.colab import drive
drive.mount('/content/drive', force_remount=True)
from tensorflow.keras.models import load_model

model_path = "/content/drive/MyDrive/Deepfake_Voice_Detection/model/lstm_model.h5"
model = load_model(model_path)  # Load the trained model
print("✅ Model loaded successfully!")


# In[ ]:


get_ipython().system('pip install flask-ngrok')

# Restart ngrok (force stop any existing connections)
get_ipython().system('pkill -f ngrok')


# In[ ]:


get_ipython().system('pip install pyngrok')


# In[ ]:


#using ngork


# Install required libraries
get_ipython().system('pip install flask flask-cors librosa tensorflow numpy pyngrok')

# Import necessary libraries
import os
import librosa
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from pyngrok import ngrok  # ✅ Import pyngrok instead of flask-ngrok
get_ipython().system('ngrok authtoken 2tdYnN1tXYRFad9PhpuwYTp1Ui8_6q85pcHB7cFREY3opay9')

# ✅ Mount Google Drive to access the trained model
from google.colab import drive
drive.mount('/content/drive')

# ✅ Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend-backend communication

# ✅ Load the trained model from Google Drive
MODEL_PATH = "/content/drive/MyDrive/Deepfake_Voice_Detection/model/lstm_model.h5"

if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
else:
    print("❌ Model file not found! Check Google Drive path.")

# ✅ Fixed MFCC length for consistent input shape
MAX_LENGTH = 500

# ✅ Function to preprocess audio before prediction
def preprocess_audio(audio_path):
    try:
        audio, sr = librosa.load(audio_path, sr=16000, duration=10)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)

        # Pad or truncate to MAX_LENGTH
        if mfccs.shape[1] < MAX_LENGTH:
            mfccs = np.pad(mfccs, ((0, 0), (0, MAX_LENGTH - mfccs.shape[1])), mode="constant")
        else:
            mfccs = mfccs[:, :MAX_LENGTH]

        mfccs = mfccs.T  # Reshape to (500, 13)
        return np.expand_dims(mfccs, axis=0)  # Add batch dimension

    except Exception as e:
        print(f"❌ Error processing audio: {e}")
        return None

# ✅ API Endpoint for Prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    audio_file = request.files['audio']
    file_path = "/content/temp_audio.wav"
    audio_file.save(file_path)

    processed_audio = preprocess_audio(file_path)
    if processed_audio is None:
        return jsonify({"error": "Invalid audio file"}), 400

    # ✅ Make prediction
    prediction = model.predict(processed_audio)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = float(prediction[0][predicted_class])

    return jsonify({
        "prediction": "Fake" if predicted_class == 1 else "Real",
        "confidence": confidence
    })

# ✅ Start Flask App with ngrok
port = 5000
public_url = ngrok.connect(port).public_url  # ✅ Get public URL for API
print(f"🌍 Ngrok Tunnel: {public_url}")

app.run(port=port)


# In[ ]:


get_ipython().system('ngrok authtoken 2tdYnN1tXYRFad9PhpuwYTp1Ui8_6q85pcHB7cFREY3opay9')


# In[ ]:


# using localhost


# Install dependencies (only once per session)
get_ipython().system('pip install flask flask-cors librosa tensorflow numpy')

# Import necessary libraries
import os
import librosa
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ✅ Mount Google Drive to access the model
from google.colab import drive
drive.mount('/content/drive')

# ✅ Initialize Flask app
app = Flask(__name__)
CORS(app)

# ✅ Load trained model from Google Drive
MODEL_PATH = "/content/drive/MyDrive/Deepfake_Voice_Detection/model/lstm_model.h5"
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

# ✅ Start Flask (Runs on Colab's internal port)
port = 5000

def start_localhost_tunnel():
    """Creates a public URL using localhost.run instead of ngrok"""
    os.system(f"ssh -R 80:localhost:{port} localhost.run")

# ✅ Run Flask App
from threading import Thread

Thread(target=start_localhost_tunnel, daemon=True).start()
app.run(port=port)


# In[13]:


get_ipython().system('jupyter nbconvert --to script /content/drive/MyDrive/Deepfake_Voice_Detection//app.ipynb')

