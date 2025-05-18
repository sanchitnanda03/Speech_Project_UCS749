
#  Speech Emotion Recognition Web App

A Streamlit-based web application for analyzing emotions and gender from speech using pre-trained deep learning models.

---

##  Features

-  Upload `.wav`, `.mp3`, or `.ogg` audio files
-  Predict emotions (3, 6, or 7-class) using MFCC-based models
-  Gender classification from voice
-  Visualizations: waveform, spectrogram, and polar plots
-  Built-in test audio for quick demo

---

##  Installation

Make sure Python 3.8–3.10 is installed on your machine.

### Step 1: Clone the repository

```bash
git clone https://github.com/yourusername/speech-emotion-app.git
cd speech-emotion-app
```

### Step 2: Create and activate a virtual environment (recommended)

```bash
python -m venv venv
# Activate the environment:
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install all required Python packages

```bash
pip install -r requirements.txt
```

---

##  Running the App

Launch the web app with:

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`.

---

## File & Folder Structure

```
speech-emotion-app/
│
├── app.py                     # Main Streamlit application
├── melspec_utils.py           # Utility for spectrogram and polar plot
├── requirements.txt           # Python dependencies
│
├── model_ser_mfcc.h5          # Model for 6-class emotion prediction
├── model_ser_7cat.h5          # Model for 7-class emotion prediction
├── model_gender.h5            # Gender prediction model
├── test_audio.wav             # Sample audio for testing
├── df_audio_sources.csv       # Data used in Project Description tab
│
├── images/                    # UI images
│   ├── emotion_image.jpg
│   ├── female_avatar.png
│   ├── male_avatar.png
│   ├── li_logo.png
│   └── sticky_note.png
│
└── audio/                     # Temporary folder created automatically to store uploads
```
