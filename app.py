import os
import re
import joblib
import pandas as pd
import numpy as np
import streamlit as st

# --- 1. CONFIGURATION DE LA PAGE STREAMLIT ---
st.set_page_config(
    page_title="Détecteur de Phishing Expliquable",
    page_icon="🛡️",
    layout="centered"
)

# --- 2. CHARGEMENT DU MODÈLE ET DU VECTORISEUR ---
@st.cache_resource
def load_resources():
    model_path = os.path.join('models', 'phishing_detection_model.pkl')
    vectorizer_path = os.path.join('models', 'tfidf_vectorizer.pkl')
    
    # Chemins alternatifs si l'application est lancée depuis un sous-dossier
    if not os.path.exists(model_path):
        model_path = os.path.join('..', 'models', 'phishing_detection_model.pkl')
        vectorizer_path = os.path.join('..', 'models', 'tfidf_vectorizer.pkl')
        
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

# --- 3. FONCTIONS DE PRÉTRAITEMENT ET D'ANALYSE ---
def preprocess(text):
    if isinstance(text, str):
        text_clean = text.lower()
        text_clean = re.sub(r'<.*?>', '', text_clean)
        text_clean = re.sub(r'http\S+|www\S+|https\S+', '', text_clean, flags=re.MULTILINE)
        text_clean = re.sub(r'[^a-zA-Z\s]', '', text_clean)
        text_clean = re.sub(r'\s+', ' ', text_clean).strip()
        return text_clean
    return ''

def analyze_phishing_triggers(original_text):
    reasons = []
    text_lower = original_text.lower()

    # Détection de liens/URLs
    if re.search(r'http[s]?://|www\.', original_text):
        reasons.append("🔗 **Liens/URLs détectés** : Présence de liens externes (potentielle redirection vers un faux site).")

    # Détection d'un sentiment d'urgence
    urgency_words = ['urgent', 'immediately', 'act now', 'account suspended', 'limited time', 'verify immediately', 'expire', '24 hours', 'immédiat', 'suspendu']
    found_urgency = [word for word in urgency_words if word in text_lower]
    if found_urgency:
        reasons.append(f"⏰ **Urgence artificielle** : Mots repérés → *{', '.join(found_urgency)}*.")

    # Détection de demandes d'informations sensibles
    financial_words = ['bank', 'password', 'credit card', 'ssn', 'verify your account', 'payment', 'paypal', 'update billing', 'compte', 'mot de passe', 'banque']
    found_financial = [word for word in financial_words if word in text_lower]
    if found_financial:
        reasons.append(f"💳 **Demande d'informations sensibles** : Mots repérés → *{', '.join(found_financial)}*.")

    # Salutations génériques
    generic_greetings = ['dear customer', 'dear user', 'dear member', 'cher client', 'cher utilisateur']
    found_greetings = [word for word in generic_greetings if word in text_lower]
    if found_greetings:
        reasons.append(f"👤 **Salutation impersonnelle** : Termes repérés → *{', '.join(found_greetings)}*.")

    return reasons

def get_top_keywords(cleaned_text, vectorizer, top_n=5):
    words = cleaned_text.split()
    if not words:
        return []
    
    feature_names = np.array(vectorizer.get_feature_names_out())
    vectorized = vectorizer.transform([cleaned_text])
    
    nonzero_indices = vectorized.nonzero()[1]
    scores = vectorized.data
    
    word_scores = [(feature_names[i], score) for i, score in zip(nonzero_indices, scores)]
    word_scores = sorted(word_scores, key=lambda x: x[1], reverse=True)
    
    return [word for word, score in word_scores[:top_n]]

# --- 4. INITIALISATION ET INTERFACE ---
st.title("🛡️ Détecteur d'Emails de Phishing")
st.write("Analyse intelligente du texte avec **justification des risques**.")

# Chargement sécurisé avec spinner
try:
    with st.spinner("Chargement des modèles ML en cours..."):
        model, vectorizer = load_resources()
    st.sidebar.success("Modèle et Vectoriseur prêts !")
except Exception as e:
    st.error(f"Erreur lors du chargement des modèles `.pkl` : {e}")

email_input = st.text_area("Contenu de l'e-mail :", height=180, placeholder="Collez le texte de l'e-mail ici...")

if st.button("Analyser l'e-mail", type="primary"):
    if not email_input.strip():
        st.warning("Veuillez entrer du texte à analyser.")
    else:
        cleaned_input = preprocess(email_input)
        vectorized_input = vectorizer.transform([cleaned_input])
        
        prediction = model.predict(vectorized_input)[0]
        probabilities = model.predict_proba(vectorized_input)[0]
        
        st.markdown("---")
        
        if prediction == 1:
            st.error("⚠️ **ATTENTION : Cet e-mail est probablement du PHISHING !**")
            st.metric(label="Niveau de risque (Phishing)", value=f"{probabilities[1]*100:.1f}%")
            
            st.subheader("📋 Motifs de Suspicion Identifiés :")
            
            # 1. Analyse basée sur les règles heuristiques
            heuristic_reasons = analyze_phishing_triggers(email_input)
            if heuristic_reasons:
                for reason in heuristic_reasons:
                    st.write(f"- {reason}")
            
            # 2. Mots-clés influents (TF-IDF)
            top_words = get_top_keywords(cleaned_input, vectorizer)
            if top_words:
                st.write(f"- 🔤 **Mots-clés suspects influents** : `{', '.join(top_words)}`")
            
            if not heuristic_reasons and not top_words:
                st.write("- Le modèle a identifié des motifs textuels statistiquement similaires aux attaques de phishing enregistrées.")

        else:
            st.success("✅ **E-mail Légitime (Sain)**")
            st.metric(label="Confiance (Légitime)", value=f"{probabilities[0]*100:.1f}%")
            st.write("Aucun motif majeur de phishing n'a été détecté dans ce message.")