import os
from datetime import datetime

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import cv2
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from PIL import Image

import melspec_utils as melspec  # renamed helper module

# ——— load primary model ———
model = load_model("model_ser_mfcc.h5")

# ——— constants ———
CAT6 = ['fear', 'angry', 'neutral', 'happy', 'sad', 'surprise']
CAT7 = ['fear', 'disgust', 'neutral', 'happy', 'sad', 'surprise', 'angry']
CAT3 = ["positive", "neutral", "negative"]
COLOR_DICT = {
    "neutral": "grey", "positive": "green", "happy": "green",
    "surprise": "orange", "fear": "purple",
    "negative": "red", "angry": "red",
    "sad": "lightblue", "disgust": "brown"
}

# page setup
st.set_page_config(page_title="SER web-app",
                   page_icon=":speech_balloon:", layout="wide")
st.set_option('deprecation.showfileUploaderEncoding', False)


def log_file(txt):
    with open("log.txt", "a") as f:
        now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        f.write(f"{txt} — {now}\n")


def save_audio(file):
    """Save uploaded audio, clearing old files first."""
    if file.size > 4_000_000:
        return 1
    folder = "audio"
    os.makedirs(folder, exist_ok=True)
    for fn in os.listdir(folder):
        fp = os.path.join(folder, fn)
        if os.path.isfile(fp):
            os.unlink(fp)
    with open(os.path.join(folder, file.name), "wb") as out:
        out.write(file.getbuffer())
    return 0


def get_melspec(audio_path):
    y, sr = librosa.load(audio_path, sr=44100)
    S = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(S))
    img = np.stack((S_db,) * 3, -1).astype(np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (224, 224))
    rgb = np.repeat(gray[..., np.newaxis], 3, -1)
    return rgb, S_db


def get_mfccs(audio_path, limit):
    y, sr = librosa.load(audio_path, sr=None)
    mfcc = librosa.feature.mfcc(y, sr=sr, n_mfcc=40)
    if mfcc.shape[1] >= limit:
        return mfcc[:, :limit]
    pad = np.zeros((40, limit))
    pad[:, :mfcc.shape[1]] = mfcc
    return pad


@st.cache
def get_title(preds, categories=CAT6):
    idx = preds.argmax()
    return f"Detected emotion: {categories[idx]} — {preds[idx]*100:.2f}%"


@st.cache
def color_dict():
    return COLOR_DICT


def main():
    # — sidebar image & menu —
    sidebar_img = Image.open("images/emotion_image.jpg")
    with st.sidebar:
        st.image(sidebar_img, width=300)
        choice = st.selectbox("Menu",
                              ["Emotion Recognition",
                               "Project description",
                               "Our team",
                               "Leave feedback",
                               "Relax"])

    # — Emotion Recognition —
    if choice == "Emotion Recognition":
        st.header("Upload an audio clip")
        model_type = st.sidebar.selectbox("Predict with", ["mfccs"])
        st.sidebar.subheader("Options")
        em3 = st.sidebar.checkbox("3 emotions", value=True)
        em6 = st.sidebar.checkbox("6 emotions", value=True)
        em7 = st.sidebar.checkbox("7 emotions")
        gender = st.sidebar.checkbox("Gender")

        # uploader
        audio_file = st.file_uploader("Audio file", type=["wav", "mp3", "ogg"])
        if audio_file:
            err = save_audio(audio_file)
            if err:
                st.error("File too large (>4 MB).")
                return
            path = os.path.join("audio", audio_file.name)
            st.audio(audio_file, format="audio/wav")
            # extract
            _, S_db = get_melspec(path)
            mfcc = get_mfccs(path, model.input_shape[-1])

            # show waveplot
            fig = plt.figure(figsize=(10, 2))
            librosa.display.waveplot(librosa.load(path, sr=44100)[0],
                                     sr=44100)
            plt.gca().set_axis_off()
            st.pyplot(fig)

            # feature plots
            col1, col2 = st.columns(2)
            with col1:
                fig1 = plt.figure(figsize=(10,2))
                librosa.display.specshow(mfcc, sr=44100, x_axis="time")
                plt.axis("off")
                st.pyplot(fig1)
            with col2:
                fig2 = plt.figure(figsize=(10,2))
                librosa.display.specshow(S_db, sr=44100,
                                         x_axis="time", y_axis="hz")
                plt.axis("off")
                st.pyplot(fig2)

            # predictions
            st.subheader("Predictions")
            # reshape & predict
            mfcc_input = mfcc.reshape((1, *mfcc.shape))
            pred6 = model.predict(mfcc_input)[0]

            cols = st.columns(4)
            if em3:
                data3 = np.array([
                    pred6[3] + 0.5*pred6[5],
                    pred6[2] + 0.5*pred6[5] + 0.5*pred6[4],
                    pred6[0] + pred6[1] + 0.5*pred6[4]
                ])
                txt = "MFCCs — " + get_title(data3, CAT3)
                fig3 = plt.figure(figsize=(5,5))
                melspec.plot_colored_polar(fig3, predictions=data3,
                                           categories=CAT3,
                                           title=txt,
                                           colors=color_dict())
                cols[0].pyplot(fig3)

            if em6:
                txt6 = "MFCCs — " + get_title(pred6, CAT6)
                fig6 = plt.figure(figsize=(5,5))
                melspec.plot_colored_polar(fig6, predictions=pred6,
                                           categories=CAT6,
                                           title=txt6,
                                           colors=color_dict())
                cols[1].pyplot(fig6)

            if em7:
                model7 = load_model("model_ser_7cat.h5")
                mf7 = get_mfccs(path, model7.input_shape[-2]).T
                pred7 = model7.predict(mf7.reshape((1, *mf7.shape)))[0]
                txt7 = "MFCCs — " + get_title(pred7, CAT7)
                fig7 = plt.figure(figsize=(5,5))
                melspec.plot_colored_polar(fig7, predictions=pred7,
                                           categories=CAT7,
                                           title=txt7,
                                           colors=color_dict())
                cols[2].pyplot(fig7)

            if gender:
                gmodel = load_model("model_gender.h5")
                gmf = get_mfccs(path, gmodel.input_shape[-1]).reshape((1,40,-1))
                gpred = gmodel.predict(gmf)[0]
                labels = ["female", "male"]
                idx = gpred.argmax()
                txtg = f"Predicted gender: {labels[idx]}"
                avatar = Image.open(
                    f"images/{'female_avatar.png' if idx==0 else 'male_avatar.png'}"
                )
                figg = plt.figure(figsize=(3,3))
                plt.title(txtg)
                plt.imshow(avatar)
                plt.axis("off")
                cols[3].pyplot(figg)

    # — Project description —
    elif choice == "Project description":
        st.title("Project description")
        st.write("""
            This app performs speech-emotion recognition
            using public datasets (Crema-D, Ravdess, Savee, Tess).
        """)
        import pandas as pd, plotly.express as px
        df = pd.read_csv("df_audio_sources.csv")
        fig = px.violin(df, y="source", x="emotion4",
                        color="actors", box=True, points="all")
        st.plotly_chart(fig, use_container_width=True)

    # — Our team —
    elif choice == "Our team":
        st.subheader("Our team")
        st.balloons()
        c1, c2 = st.columns([3,2])
        with c1:
            st.info("@gmail.com")
            st.info("@gmail.com")
            st.info("@gmail.com")
        with c2:
            st.image("images/li_logo.png", use_column_width=True)

    # — Leave feedback —
    elif choice == "Leave feedback":
        st.subheader("Leave feedback")
        fb = st.text_area("Your feedback")
        who = st.selectbox("Your name",
                           ["checker1","checker2","checker3","checker4"])
        if st.button("Submit"):
            st.success(f"Thanks, {who}!")
            log_file(f"{who}: {fb}")
            st.image("images/sticky_note.png")

    # — Relax —
    else:
        st.subheader("Relax")
        if st.button("Get random mood"):
            quotes = {
                "Good job and almost done":"checker1",
                "Great start!!":"checker2",
                "Please make corrections":"checker3",
                "Well done!":"checker1"
            }
            q = np.random.choice(list(quotes.keys()))
            st.markdown(f"## *{q}*")
            st.markdown(f"### ***{quotes[q]}***")


if __name__ == "__main__":
    main()
