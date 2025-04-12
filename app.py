import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

# === Charger le modèle entraîné ===
model = load_model("best_model_fold2.keras")

# === Fonction pour encoder une séquence en One-Hot ===
def preprocess_sequence(seq):
    seq = seq.upper()
    mapping = {'A': [1,0,0,0], 'C': [0,1,0,0], 'G': [0,0,1,0], 'T': [0,0,0,1]}
    
    encoded = [mapping.get(nuc, [0,0,0,0]) for nuc in seq]
    
    # Padding si la séquence est trop courte
    while len(encoded) < 60:
        encoded.append([0,0,0,0])  # padding neutre
    
    encoded = encoded[:60]  # Tronque à 60 caractères si trop long

    return np.array(encoded)

# === Fonction de prédiction ===
def predict(seq):
    input_seq = preprocess_sequence(seq)
    input_seq = np.expand_dims(input_seq, axis=0)  # ajoute la dimension batch
    prediction = model.predict(input_seq)[0][0]
    return prediction

# === Interface utilisateur Streamlit ===
st.title("🧬 Is It a Promoter?")
st.subheader("Entrez une séquence ADN (A, C, G, T) de 60 nucléotides :")

user_seq = st.text_input("Séquence ADN")

if user_seq:
    if set(user_seq.upper()).issubset({'A', 'C', 'G', 'T'}):
        prob = predict(user_seq)
        if prob > 0.5:
            st.success(f"✅ Promoteur détecté avec une probabilité de {prob:.2f}")
        else:
            st.error(f"❌ Ce n'est PAS un promoteur (probabilité = {prob:.2f})")
    else:
        st.warning("❗ La séquence ne doit contenir que des lettres A, C, G, ou T.")
