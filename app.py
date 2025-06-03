import streamlit as st
import requests

API_URL = "http://localhost:7071/api/recommand"
API_URL = "https://contentrecomapp.azurewebsites.net/api/recommand"
# Définition des labels et valeurs pour le selectbox
labels = ['Utilisateur 1', 'Utilisateur 2', 'Utilisateur 3', 'Utilisateur 4', 
          'Utilisateur 5', 'Utilisateur 6', 'Utilisateur 7', 'Utilisateur 8', 
          'Utilisateur 9']
values = [2520, 2546, 10188, 14073, 16280, 26751, 33937, 65739, 188046]

def predict(user_input, hybrid, similarity, alpha, k):
    response = requests.post(API_URL, json={"user_id": user_input, "hybrid": hybrid, "similarity": similarity, "alpha": alpha, "k": k})
    print(response)
    st.write(response.json())

def main():
    st.title("Content Recom")
    st.write("Bienvenue sur le POC de My Content pour la recommandation de contenu !")
    
    # Exemple de widget avec labels et values séparés
    user_input = st.selectbox("Choisissez un utilisateur", 
                            options=values,
                            format_func=lambda x: labels[values.index(x)])
    hybrid = st.checkbox("Hybrid", value=True)
    similarity = st.checkbox("Similarity", value=True)
    alpha = st.slider("Alpha", min_value=0.0, max_value=1.0, value=0.7)
    k = st.slider("K", min_value=1, max_value=10, value=5)
    if st.button("Predict"):
        predict(user_input, hybrid, similarity, alpha, k)

if __name__ == "__main__":
    main() 