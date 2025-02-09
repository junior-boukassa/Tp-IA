 # 📌 Étape 5 : Déploiement du modèle Random Forest avec Streamlit

import streamlit as st  # Bibliothèque pour l'interface
import pandas as pd
import numpy as np
import joblib  # Pour charger le modèle
from sklearn.preprocessing import StandardScaler

# 📂 Charger le modèle entraîné
model = joblib.load("random_forest_model.pkl")  # Assurez-vous que ce fichier existe

# 📂 Charger les données pour la mise à l'échelle
df = pd.read_csv("diabetes_cleaned.csv")  # Utilisé pour normaliser les entrées
scaler = StandardScaler()
scaler.fit(df.drop(columns=["Outcome"]))  # Adapter la normalisation aux données d'entraînement

# 🎨 Configuration de l'application Streamlit
st.title("🔬 Prédiction du Diabète - IA avec Random Forest")
st.write("Remplissez les champs ci-dessous pour prédire si une personne est diabétique.")

# 📊 Création du formulaire d'entrée utilisateur
pregnancies = st.number_input("Nombre de grossesses", min_value=0, max_value=20, value=1)
glucose = st.number_input("Taux de glucose", min_value=50, max_value=300, value=100)
blood_pressure = st.number_input("Pression artérielle", min_value=40, max_value=200, value=70)
skin_thickness = st.number_input("Épaisseur de la peau (mm)", min_value=0, max_value=100, value=20)
insulin = st.number_input("Taux d'insuline", min_value=0, max_value=900, value=80)
bmi = st.number_input("Indice de Masse Corporelle (BMI)", min_value=10.0, max_value=60.0, value=25.0, format="%.1f")
diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5, format="%.3f")
age = st.number_input("Âge", min_value=10, max_value=100, value=30)

# 📥 Récupérer les entrées utilisateur et les structurer
input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])

# 📏 Normaliser les données
input_scaled = scaler.transform(input_data)

# 📌 Prédiction lorsque l'utilisateur appuie sur le bouton
if st.button("Prédire 🚀"):
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]  # Probabilité d'être diabétique

    # 📌 Affichage du résultat
    if prediction == 1:
        st.error(f"⚠️ **Le modèle prédit que cette personne est diabétique.** (Probabilité : {prob:.2f})")
    else:
        st.success(f"✅ **Le modèle prédit que cette personne n'est pas diabétique.** (Probabilité : {prob:.2f})")

    # 📊 Affichage des valeurs saisies sous forme de tableau
    df_input = pd.DataFrame(input_data, columns=["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigree", "Age"])
    st.write("🔎 **Résumé des valeurs saisies** :")
    st.dataframe(df_input)

# 🎨 Personnalisation du design
st.markdown("""
<style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        padding: 10px 24px;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)
