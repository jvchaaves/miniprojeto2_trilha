import streamlit as st
import numpy as np
import librosa
import joblib
import tensorflow as tf
import os
import soundfile as sf
import tempfile
import matplotlib.pyplot as plt

# --------------------------------- PARTE 1: EXTRAIR FEATURES --------------------------------- #

# Carregar o modelo e o scaler
MODEL_PATH = r"C:\Users\joaov\OneDrive\Área de Trabalho\programação\trilha\miniprojeto2\miniprojeto2_trilha\models\audio_emotion_model.keras" 
SCALER_PATH = r"C:\Users\joaov\OneDrive\Área de Trabalho\programação\trilha\miniprojeto2\miniprojeto2_trilha\models\scaler.joblib"              

model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Lista de emoções
EMOTIONS = ["angry", "calm", "disgust", "fear",
            "happy", "neutral", "sad", "surprise"]


# Função para extrair features
def extract_features(audio_path):
    data, sr = librosa.load(audio_path, sr=16000, mono=True)
    features = []


    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y=data)
    features = np.hstack((features, np.mean(zcr, axis=1)))

    # Chroma_stft
    chroma_stft = librosa.feature.chroma_stft(y=data, sr=sr)
    features = np.hstack((features, np.mean(chroma_stft, axis=1)))

    # MFCC
    mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=20)
    features = np.hstack((features, np.mean(mfcc, axis=1)))

    # RMS
    rms = librosa.feature.rms(y=data)
    features = np.hstack((features, np.mean(rms, axis=1)))

    # MelSpectrogram
    mel = librosa.feature.melspectrogram(y=data, sr=sr)
    features = np.hstack((features, np.mean(mel, axis=1)))

    target_length = 162
    if len(features) < target_length:
        features.extend([0] * (target_length - len(features)))
    elif len(features) > target_length:
        features = features[:target_length]

    return np.array(features).reshape(1, -1)


# --------------------------------- PARTE 2: STREAMLIT --------------------------------- #

# Configuração do app Streamlit (Título e descrição)
st.title('detectando emoções por áudio 🎵')
st.write('carregue seu áudio e veja qual emoção ele transmite!! ')

# Upload de arquivo de áudio (wav, mp3, ogg)
uploaded_file = st.file_uploader(
    "Escolha um arquivo de áudio...", type=["wav", "mp3", "ogg"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    
    features = extract_features(tmp_path)  
    features = scaler.transform(features)
    features = np.expand_dims(features, axis=2)

    pred = model.predict(features)
    pred_class = np.argmax(pred, axis=1).reshape(-1, 1)
    pred_label = EMOTIONS[pred_class[0][0]]


    st.subheader("Emoção detectada 🥳:")
    
    colors = ['#FF6F61', '#6B5B95', '#88B04B', '#F7CAC9', '#92A8D1']
    classes = EMOTIONS
    fig, ax = plt.subplots()
    ax.set_ylabel("Probabilidade")
    ax.bar(classes, pred[0],color = colors)
    st.pyplot(fig)

    os.remove(tmp_path)
