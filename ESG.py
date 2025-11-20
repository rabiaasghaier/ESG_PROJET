import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import RFE
from scipy.stats import chi2_contingency, f_oneway
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import warnings
import requests
import json
from datetime import datetime
warnings.filterwarnings('ignore')
import pandas as pd
import os
import warnings
import requests
import json
from datetime import datetime


# Configuration de la page
st.set_page_config(
    page_title="Analyse ESG Avancée",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS moderne et élégant
st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Background avec gradient clair */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        background-attachment: fixed;
    }
    
    /* Container principal */
    .main .block-container {
        background: rgba(255, 255, 255, 0.98);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 20px 60px rgba(0,0,0,0.15);
        backdrop-filter: blur(10px);
    }
    
    /* Headers */
    .main-header {
        font-size: 48px;
        font-weight: 700;
        background: linear-gradient(135deg, #0066cc 0%, #0052a3 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 30px;
        animation: fadeInDown 0.8s ease-out;
    }
    
    .sub-header {
        font-size: 28px;
        font-weight: 600;
        color: #004d99;
        margin-top: 30px;
        margin-bottom: 20px;
        padding-left: 15px;
        border-left: 5px solid #0066cc;
        animation: fadeInLeft 0.6s ease-out;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f0f4f8 100%);
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
        border: 2px solid #0066cc;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(0, 102, 204, 0.25);
        border: 2px solid #0052a3;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #0066cc 0%, #0052a3 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 12px 30px;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 102, 204, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 102, 204, 0.5);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: linear-gradient(135deg, #f0f4f8 0%, #e8ecf1 100%);
        padding: 10px;
        border-radius: 15px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: 600;
        transition: all 0.3s ease;
        color: #333333;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #0066cc 0%, #0052a3 100%);
        color: white;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0066cc 0%, #0052a3 100%);
        padding: 2rem 1rem;
    }
    
    [data-testid="stSidebar"] .stRadio label {
        color: white;
        font-weight: 500;
        padding: 10px;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    
    [data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(255, 255, 255, 0.2);
    }
    
    /* Chatbot moderne */
    .chatbot-fab {
        position: fixed;
        bottom: 30px;
        right: 30px;
        width: 65px;
        height: 65px;
        border-radius: 50%;
        background: linear-gradient(135deg, #0066cc 0%, #0052a3 100%);
        box-shadow: 0 8px 24px rgba(0, 102, 204, 0.4);
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        transition: all 0.3s ease;
        animation: pulse 2s infinite;
        z-index: 9999;
    }
    
    .chatbot-fab:hover {
        transform: scale(1.1) rotate(5deg);
        box-shadow: 0 12px 32px rgba(0, 102, 204, 0.6);
    }
    
    @keyframes pulse {
        0%, 100% { box-shadow: 0 8px 24px rgba(0, 102, 204, 0.4); }
        50% { box-shadow: 0 8px 32px rgba(0, 102, 204, 0.7); }
    }
    
    .chat-container {
        position: fixed;
        bottom: 110px;
        right: 30px;
        width: 420px;
        height: 600px;
        background: white;
        border-radius: 20px;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        display: flex;
        flex-direction: column;
        z-index: 9998;
        animation: slideUp 0.4s cubic-bezier(0.68, -0.55, 0.265, 1.55);
    }
    
    @keyframes slideUp {
        from {
            transform: translateY(30px) scale(0.9);
            opacity: 0;
        }
        to {
            transform: translateY(0) scale(1);
            opacity: 1;
        }
    }
    
    .chat-header {
        background: linear-gradient(135deg, #0066cc 0%, #0052a3 100%);
        color: white;
        padding: 20px;
        border-radius: 20px 20px 0 0;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    
    .chat-avatar {
        width: 45px;
        height: 45px;
        border-radius: 50%;
        background: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 24px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    
    .chat-messages {
        flex: 1;
        overflow-y: auto;
        padding: 20px;
        background: #f8f9fa;
    }
    
    .chat-messages::-webkit-scrollbar {
        width: 6px;
    }
    
    .chat-messages::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #0066cc 0%, #0052a3 100%);
        border-radius: 3px;
    }
    
    .message {
        margin-bottom: 20px;
        animation: messageSlide 0.3s ease-out;
    }
    
    @keyframes messageSlide {
        from {
            opacity: 0;
            transform: translateY(15px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .message.user .message-bubble {
        background: linear-gradient(135deg, #0066cc 0%, #0052a3 100%);
        color: white;
        margin-left: auto;
        border-radius: 20px 20px 5px 20px;
    }
    
    .message.assistant .message-bubble {
        background: white;
        color: #1a3a52;
        margin-right: auto;
        border-radius: 20px 20px 20px 5px;
        border: 2px solid #0066cc;
    }
    
    .message-bubble {
        padding: 15px 20px;
        max-width: 75%;
        word-wrap: break-word;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .typing-indicator {
        display: flex;
        gap: 6px;
        padding: 15px 20px;
        background: white;
        border-radius: 20px;
        width: fit-content;
        border: 2px solid #e9ecef;
    }
    
    .typing-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: #0066cc;
        animation: typing 1.4s infinite;
    }
    
    .typing-dot:nth-child(2) { animation-delay: 0.2s; }
    .typing-dot:nth-child(3) { animation-delay: 0.4s; }
    
    @keyframes typing {
        0%, 60%, 100% {
            transform: translateY(0);
            opacity: 0.4;
        }
        30% {
            transform: translateY(-12px);
            opacity: 1;
        }
    }
    
    .quick-action {
        display: inline-block;
        margin: 5px;
        padding: 8px 16px;
        background: white;
        border: 2px solid #0066cc;
        border-radius: 20px;
        color: #0066cc;
        font-size: 13px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .quick-action:hover {
        background: linear-gradient(135deg, #0066cc 0%, #0052a3 100%);
        color: white;
        transform: translateY(-2px);
    }
    
    .notification-badge {
        position: absolute;
        top: -5px;
        right: -5px;
        background: #ff4757;
        color: white;
        border-radius: 50%;
        width: 24px;
        height: 24px;
        font-size: 12px;
        font-weight: bold;
        display: flex;
        align-items: center;
        justify-content: center;
        border: 3px solid white;
        animation: bounce 0.5s infinite alternate;
    }
    
    @keyframes bounce {
        from { transform: translateY(0); }
        to { transform: translateY(-5px); }
    }
    
    /* Animations */
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeInLeft {
        from {
            opacity: 0;
            transform: translateX(-30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    /* Dataframes */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    /* Info boxes */
    .stInfo, .stWarning, .stSuccess, .stError {
        border-radius: 12px;
        border-left: 5px solid;
        animation: fadeInLeft 0.5s ease-out;
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .chat-container {
            width: calc(100vw - 40px);
            height: calc(100vh - 140px);
            right: 20px;
            bottom: 100px;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Titre principal avec animation
st.markdown('<p class="main-header">🌍 Analyse ESG Avancée avec IA</p>', unsafe_allow_html=True)

# Initialisation de session_state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'df_esg' not in st.session_state:
    st.session_state.df_esg = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_page' not in st.session_state:
    st.session_state.current_page = "📁 Chargement des Données"
if 'chat_open' not in st.session_state:
    st.session_state.chat_open = False
if 'chat_notifications' not in st.session_state:
    st.session_state.chat_notifications = 0
if 'is_typing' not in st.session_state:
    st.session_state.is_typing = False
if 'current_visualization' not in st.session_state:
    st.session_state.current_visualization = None

# Configuration API Gemini
GEMINI_API_KEY = "api"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"

# Liste des codes pays souverains (195 pays)
SOVEREIGN_COUNTRIES_CODES = ['AFG', 'AGO', 'ALB', 'AND', 'ARE', 'ARG', 'ARM', 'ATG', 'AUS', 'AUT', 'AZE', 'BDI', 'BEL', 'BEN', 'BFA', 'BGD', 'BGR', 'BHR', 'BHS', 'BIH', 'BLR', 'BLZ', 'BOL', 'BRA', 'BRB', 'BRN', 'BTN', 'BWA', 'CAF', 'CAN', 'CHE', 'CHL', 'CHN', 'CIV', 'CMR', 'COD', 'COG', 'COL', 'COM', 'CPV', 'CRI', 'CUB', 'CYP', 'CZE', 'DEU', 'DJI', 'DMA', 'DNK', 'DOM', 'DZA', 'ECU', 'EGY', 'ERI', 'ESP', 'EST', 'ETH', 'FIN', 'FJI', 'FRA', 'FSM', 'GAB', 'GBR', 'GEO', 'GHA', 'GIN', 'GMB', 'GNB', 'GNQ', 'GRC', 'GRD', 'GTM', 'GUY', 'HND', 'HRV', 'HTI', 'HUN', 'IDN', 'IND', 'IRL', 'IRN', 'IRQ', 'ISL', 'ISR', 'ITA', 'JAM', 'JOR', 'JPN', 'KAZ', 'KEN', 'KGZ', 'KHM', 'KIR', 'KNA', 'KOR', 'KWT', 'LAO', 'LBN', 'LBR', 'LBY', 'LCA', 'LIE', 'LKA', 'LSO', 'LTU', 'LUX', 'LVA', 'MAR', 'MCO', 'MDA', 'MDG', 'MDV', 'MEX', 'MHL', 'MKD', 'MLI', 'MLT', 'MMR', 'MNE', 'MNG', 'MOZ', 'MRT', 'MUS', 'MWI', 'MYS', 'NAM', 'NER', 'NGA', 'NIC', 'NLD', 'NOR', 'NPL', 'NRU', 'NZL', 'OMN', 'PAK', 'PAN', 'PER', 'PHL', 'PLW', 'PNG', 'POL', 'PRK', 'PRT', 'PRY', 'PSE', 'QAT', 'ROU', 'RUS', 'RWA', 'SAU', 'SDN', 'SEN', 'SGP', 'SLB', 'SLE', 'SLV', 'SMR', 'SOM', 'SRB', 'SSD', 'STP', 'SUR', 'SVK', 'SVN', 'SWE', 'SWZ', 'SYC', 'SYR', 'TCD', 'TGO', 'THA', 'TJK', 'TKM', 'TLS', 'TON', 'TTO', 'TUN', 'TUR', 'TUV', 'TZA', 'UGA', 'UKR', 'URY', 'USA', 'UZB', 'VAT', 'VCT', 'VEN', 'VNM', 'VUT', 'WSM', 'YEM', 'ZAF', 'ZMB', 'ZWE']

# Actions rapides contextuelles améliorées
QUICK_ACTIONS = {
    "📁 Chargement des Données": [
        "Comment charger mes données ?",
        "Quels sont les 195 pays analysés ?",
        "Quelle est la structure des données ?"
    ],
    "📈 Scores ESG & Top Pays": [
        "Qui sont les leaders ESG ?",
        "Comment interpréter le score ESG ?",
        "Explique-moi ce classement"
    ],
    "🔍 Analyse Exploratoire": [
        "Que signifie cette corrélation ?",
        "Comment lire la heatmap ?",
        "Analyse les tendances principales"
    ],
    "📊 Volatilité & Comparaisons": [
        "Qu'est-ce que la volatilité ESG ?",
        "Compare ces pays pour moi",
        "Explique le graphique radar"
    ],
    "🎯 Feature Importance": [
        "Quels indicateurs sont importants ?",
        "Explique XGBoost simplement",
        "Comment utiliser ces insights ?"
    ],
    "🤖 Machine Learning": [
        "Comment fonctionne Random Forest ?",
        "Interprète la matrice de confusion",
        "C'est quoi l'accuracy ?"
    ],
    "🧠 Deep Learning": [
        "Explique le réseau de neurones",
        "Pourquoi utiliser le dropout ?",
        "Comment lire ces courbes ?"
    ],
    "🎨 Clustering": [
        "C'est quoi K-Means ?",
        "Comment choisir K ?",
        "Interprète ces clusters"
    ],
    "🌏 Analyses Régionales": [
        "Quelle région est la meilleure ?",
        "Compare Europe vs Asie",
        "Explique ces différences"
    ]
}

# Fonction améliorée pour appeler Gemini API avec contexte enrichi
def call_gemini_api(prompt, context="", data_summary=None):
    """Appelle l'API Gemini avec contexte enrichi des données"""
    try:
        headers = {'Content-Type': 'application/json'}
        
        # Construire un prompt enrichi avec les données
        full_prompt = f"""Tu es un assistant expert en analyse ESG (Environnement, Social, Gouvernance) et data science.
Tu aides les utilisateurs à comprendre leurs données ESG, interpréter les visualisations et tirer des insights actionnables.

📊 **Contexte de la page actuelle:** {context}

"""
        
        if data_summary:
            full_prompt += f"""📈 **Données actuelles:**
{data_summary}

"""
        
        full_prompt += f"""❓ **Question de l'utilisateur:** {prompt}

💡 **Instructions:**
- Réponds en français de manière claire, structurée et professionnelle
- Si la question concerne un graphique, fournis une interprétation détaillée avec des insights concrets
- Utilise des émojis pour rendre la réponse plus engageante
- Donne des exemples chiffrés quand c'est possible
- Propose des actions ou recommandations si pertinent
- Sois pédagogue et accessible, même pour des concepts complexes"""

        data = {
            "contents": [{
                "parts": [{"text": full_prompt}]
            }],
            "generationConfig": {
                "temperature": 0.7,
                "topK": 40,
                "topP": 0.95,
                "maxOutputTokens": 1024,
            }
        }
        
        response = requests.post(
            f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            if 'candidates' in result and len(result['candidates']) > 0:
                return result['candidates'][0]['content']['parts'][0]['text']
            else:
                return "Désolé, je n'ai pas pu générer une réponse. Pouvez-vous reformuler votre question ?"
        else:
            return f"⚠️ Erreur API ({response.status_code}). Veuillez réessayer dans quelques instants."
            
    except Exception as e:
        return f"❌ Erreur de connexion: {str(e)}. Vérifiez votre connexion internet et réessayez."

# Fonction pour obtenir un résumé des données enrichi
def get_data_summary(df_esg, page_name):
    """Génère un résumé détaillé des données selon la page"""
    if df_esg is None or df_esg.empty:
        return None
    
    summary = f"""
- Nombre total d'observations: {len(df_esg):,}
- Nombre de pays: {df_esg['Country Code'].nunique()}
- Période couverte: {df_esg['Year'].min()} - {df_esg['Year'].max()}
- Score ESG moyen global: {df_esg['Score_ESG_Total'].mean():.3f}
- Meilleur pays (score moyen): {df_esg.groupby('Country Name')['Score_ESG_Total'].mean().idxmax()}
- Score E moyen: {df_esg['Score_E'].mean():.3f}
- Score S moyen: {df_esg['Score_S'].mean():.3f}
- Score G moyen: {df_esg['Score_G'].mean():.3f}
"""
    
    if page_name == "📈 Scores ESG & Top Pays":
        top_3 = df_esg.groupby('Country Name')['Score_ESG_Total'].mean().nlargest(3)
        summary += f"\n🏆 Top 3 pays:\n"
        for i, (country, score) in enumerate(top_3.items(), 1):
            summary += f"  {i}. {country}: {score:.3f}\n"
    
    elif page_name == "📊 Volatilité & Comparaisons":
        volatility_stats = df_esg.groupby('Country Name')['Volatilite_Score_ESG'].first()
        summary += f"\n📉 Volatilité moyenne: {volatility_stats.mean():.4f}"
        summary += f"\n   Pays le plus stable: {volatility_stats.idxmin()}"
        summary += f"\n   Pays le plus volatile: {volatility_stats.idxmax()}"
    
    return summary

# Fonction pour obtenir le contexte de la page enrichi
def get_page_context(page_name, df_esg=None):
    """Génère un contexte détaillé basé sur la page actuelle"""
    contexts = {
        "📁 Chargement des Données": """Page de chargement des données ESG. 
Cette page permet d'importer les fichiers Excel contenant les indicateurs Environnement (E), Social (S) et Gouvernance (G) pour 195 pays souverains reconnus par l'ONU. 
Les données sont ensuite nettoyées, normalisées et fusionnées pour créer des scores ESG composites.""",
        
        "📈 Scores ESG & Top Pays": """Page d'analyse des scores et classements ESG. 
Affiche le Top 10 des pays avec les meilleurs scores ESG, la distribution des scores (histogrammes), 
et les tendances temporelles. Les graphiques interactifs permettent d'explorer l'évolution des performances ESG.""",
        
        "🔍 Analyse Exploratoire": """Page d'exploration statistique approfondie. 
Présente les statistiques descriptives (moyenne, médiane, écart-type), 
des heatmaps de corrélation entre piliers E, S, G, et l'évolution détaillée par pays au fil du temps.""",
        
        "📊 Volatilité & Comparaisons": """Page d'analyse de la volatilité (stabilité) des scores ESG. 
La volatilité mesure l'écart-type du score dans le temps : plus elle est élevée, plus le score est instable. 
Permet de comparer plusieurs pays avec des graphiques radar et barres groupées.""",
        
        "🎯 Feature Importance": """Page d'analyse de l'importance des caractéristiques (features). 
Utilise plusieurs méthodes (XGBoost, Permutation, RFE, ANOVA) pour identifier quels indicateurs ESG 
ont le plus d'impact sur le score global. Aide à prioriser les actions.""",
        
        "🤖 Machine Learning": """Page de modélisation par Machine Learning avec Random Forest. 
Entraîne un modèle pour prédire les catégories ESG (Faible/Moyen/Élevé). 
Affiche l'accuracy, la matrice de confusion et l'importance des variables.""",
        
        "🧠 Deep Learning": """Page de Deep Learning avec réseau de neurones TensorFlow. 
Utilise un modèle séquentiel avec couches denses et dropout pour la classification ESG. 
Les courbes d'apprentissage montrent l'évolution de la loss et de l'accuracy.""",
        
        "🎨 Clustering": """Page de clustering K-Means non supervisé. 
Regroupe les pays en clusters similaires basés sur leurs scores ESG. 
La méthode du coude aide à choisir le nombre optimal de clusters. Visualisation PCA en 2D.""",
        
        "🌏 Analyses Régionales": """Page de comparaison régionale (Asie, Europe, Amérique, Afrique, Moyen-Orient). 
Compare les performances ESG moyennes par région avec des graphiques en barres, 
heatmaps régionales et analyse des disparités."""
    }
    
    context = contexts.get(page_name, "Navigation dans l'application d'analyse ESG avancée.")
    
    if df_esg is not None and not df_esg.empty:
        context += f"\n\n📊 Données chargées: {len(df_esg):,} observations, {df_esg['Country Code'].nunique()} pays, période {df_esg['Year'].min()}-{df_esg['Year'].max()}."
    
    return context

# Fonctions utilitaires
@st.cache_data
def filter_sovereign_countries(df, codes):
    if df.empty:
        return df
    return df[df['Country Code'].isin(codes)]

@st.cache_data
def clean_and_melt(df, esg_type):
    """Transforme le DataFrame du format large au format long"""
    if df.empty:
        return pd.DataFrame()
    
    id_vars = ['Country Name', 'Country Code', 'Series Name', 'Code']
    value_vars = [col for col in df.columns if 'YR' in str(col)]
    
    df_long = pd.melt(df, id_vars=id_vars, value_vars=value_vars,
                      var_name='Year_Label', value_name='Value')
    
    df_long['Year'] = df_long['Year_Label'].str.extract(r'(\d{4})').astype(int)
    df_long.drop(columns=['Year_Label'], inplace=True)
    df_long['Value'] = df_long['Value'].replace('..', np.nan)
    df_long['Value'] = pd.to_numeric(df_long['Value'], errors='coerce')
    df_long['ESG_Type'] = esg_type
    df_long['Indicator'] = df_long['Code'] + '_' + esg_type
    
    return df_long

@st.cache_data
def process_data(data_E, data_S, data_G):
    """Traite et fusionne les données ESG avec filtrage des pays souverains"""
    data_E = filter_sovereign_countries(data_E, SOVEREIGN_COUNTRIES_CODES)
    data_S = filter_sovereign_countries(data_S, SOVEREIGN_COUNTRIES_CODES)
    data_G = filter_sovereign_countries(data_G, SOVEREIGN_COUNTRIES_CODES)
    
    df_E_long = clean_and_melt(data_E, 'E')
    df_S_long = clean_and_melt(data_S, 'S')
    df_G_long = clean_and_melt(data_G, 'G')
    
    df_combined = pd.concat([df_E_long, df_S_long, df_G_long], ignore_index=True)
    df_combined = df_combined.drop_duplicates(subset=['Country Name', 'Country Code', 'Year', 'Indicator'])
    df_combined = df_combined[['Country Name', 'Country Code', 'Year', 'Indicator', 'Value']]
    
    pivot_index = ['Country Name', 'Country Code', 'Year']
    
    try:
        df_final = df_combined.set_index(pivot_index + ['Indicator'])['Value'].unstack(fill_value=np.nan)
        df_final.reset_index(inplace=True)
        df_final.columns.name = None
    except MemoryError:
        st.error("❌ Erreur mémoire: Les données sont trop volumineuses.")
        return pd.DataFrame()
    
    indicator_cols = [col for col in df_final.columns if col not in ['Country Name', 'Country Code', 'Year']]
    
    if len(indicator_cols) > 500:
        st.warning(f"⚠️ {len(indicator_cols)} indicateurs détectés. Réduction pour optimiser la mémoire...")
        nan_counts = df_final[indicator_cols].isna().sum()
        best_indicators = nan_counts.nsmallest(500).index.tolist()
        indicator_cols = best_indicators
        df_final = df_final[['Country Name', 'Country Code', 'Year'] + indicator_cols]
    
    df_final[indicator_cols] = df_final[indicator_cols].apply(lambda x: x.fillna(x.mean()), axis=0)
    
    scaler = MinMaxScaler()
    df_final[indicator_cols] = scaler.fit_transform(df_final[indicator_cols])
    
    e_cols = [col for col in indicator_cols if col.endswith('_E')]
    s_cols = [col for col in indicator_cols if col.endswith('_S')]
    g_cols = [col for col in indicator_cols if col.endswith('_G')]
    
    if len(e_cols) > 0:
        df_final['Score_E'] = df_final[e_cols].mean(axis=1)
    else:
        df_final['Score_E'] = 0.5
        
    if len(s_cols) > 0:
        df_final['Score_S'] = df_final[s_cols].mean(axis=1)
    else:
        df_final['Score_S'] = 0.5
        
    if len(g_cols) > 0:
        df_final['Score_G'] = df_final[g_cols].mean(axis=1)
    else:
        df_final['Score_G'] = 0.5
    
    df_final['Score_ESG_Total'] = df_final[['Score_E', 'Score_S', 'Score_G']].mean(axis=1)
    
    df_final.sort_values(by=['Country Code', 'Year'], inplace=True)
    for score in ['Score_E', 'Score_S', 'Score_G', 'Score_ESG_Total']:
        df_final[f'{score}_Lag1'] = df_final.groupby('Country Code')[score].shift(1)
        df_final[f'{score}_Change'] = df_final[score] - df_final[f'{score}_Lag1']
    df_final.fillna(0, inplace=True)
    
    volatility = df_final.groupby('Country Name')['Score_ESG_Total'].std().reset_index()
    volatility.rename(columns={'Score_ESG_Total': 'Volatilite_Score_ESG'}, inplace=True)
    df_final = pd.merge(df_final, volatility, on='Country Name', how='left')
    
    quantiles = df_final['Score_ESG_Total'].quantile([0.33, 0.66])
    def categorize_esg(score):
        if score <= quantiles.iloc[0]:
            return 'Faible'
        elif score <= quantiles.iloc[1]:
            return 'Moyen'
        else:
            return 'Élevé'
    
    df_final['ESG_Category'] = df_final['Score_ESG_Total'].apply(categorize_esg)
    df_final['ESG_Target'] = df_final['ESG_Category'].astype('category').cat.codes
    
    return df_final

# Sidebar - Navigation moderne
st.sidebar.title("🎯 Navigation")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Choisissez une section:",
    ["📁 Chargement des Données", 
     "📈 Scores ESG & Top Pays",
     "🔍 Analyse Exploratoire",
     "📊 Volatilité & Comparaisons",
     "🎯 Feature Importance",
     "🤖 Machine Learning",
     "🧠 Deep Learning",
     "🎨 Clustering",
     "🌏 Analyses Régionales"],
    key="navigation_radio"
)

# Mettre à jour la page actuelle
if page != st.session_state.current_page:
    st.session_state.current_page = page
    if not st.session_state.chat_open and len(st.session_state.chat_history) > 0:
        st.session_state.chat_notifications += 1

# Fonction pour gérer les messages utilisateur
def handle_user_message(user_input):
    """Traite le message de l'utilisateur avec contexte enrichi"""
    current_time = datetime.now().strftime("%H:%M")
    
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_input,
        "time": current_time
    })
    
    st.session_state.is_typing = True
    
    # Obtenir le contexte enrichi
    context = get_page_context(st.session_state.current_page, st.session_state.df_esg)
    data_summary = get_data_summary(st.session_state.df_esg, st.session_state.current_page)
    
    # Appeler l'API Gemini avec contexte enrichi
    response = call_gemini_api(user_input, context, data_summary)
    
    st.session_state.is_typing = False
    
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": response,
        "time": datetime.now().strftime("%H:%M")
    })

# Interface du chatbot moderne
st.markdown("""
<div class="chatbot-fab" onclick="document.getElementById('chat_toggle').click()">
    <span style="font-size: 32px;">💬</span>
    """ + (f'<div class="notification-badge">{st.session_state.chat_notifications}</div>' if st.session_state.chat_notifications > 0 else '') + """
</div>
""", unsafe_allow_html=True)

# Toggle du chatbot (bouton invisible)
if st.button("toggle", key="chat_toggle", help="Toggle chat"):
    st.session_state.chat_open = not st.session_state.chat_open
    if st.session_state.chat_open:
        st.session_state.chat_notifications = 0
    st.rerun()

# Afficher le chatbot si ouvert
if st.session_state.chat_open:
    with st.container():
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        # Header du chat
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(f"""
            <div class="chat-header">
                <div style="display: flex; align-items: center; gap: 12px;">
                    <div class="chat-avatar">🤖</div>
                    <div>
                        <h3 style="margin: 0; font-size: 18px;">Assistant ESG</h3>
                        <p style="margin: 0; font-size: 13px; opacity: 0.9;">En ligne • Expert IA</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            if st.button("✖️", key="close_chat", help="Fermer le chat"):
                st.session_state.chat_open = False
                st.rerun()
        
        # Zone de messages
        st.markdown('<div class="chat-messages">', unsafe_allow_html=True)
        
        # Message de bienvenue
        if len(st.session_state.chat_history) == 0:
            st.markdown(f"""
            <div class="message assistant">
                <div class="message-bubble">
                    <strong>👋 Bonjour !</strong><br><br>
                    Je suis votre assistant expert en analyse ESG. Je peux vous aider à :<br><br>
                    📊 Interpréter les graphiques et visualisations<br>
                    💡 Expliquer les concepts ESG et data science<br>
                    🎯 Analyser vos données et identifier des insights<br>
                    🔍 Répondre à vos questions sur le projet<br><br>
                    📍 Vous êtes actuellement sur : <strong>{st.session_state.current_page}</strong>
                    <div style="font-size: 11px; color: #888; margin-top: 8px;">{datetime.now().strftime("%H:%M")}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Afficher l'historique
            for msg in st.session_state.chat_history:
                role_class = "user" if msg["role"] == "user" else "assistant"
                icon = "👤" if msg["role"] == "user" else "🤖"
                st.markdown(f"""
                <div class="message {role_class}">
                    <div class="message-bubble">
                        <strong>{icon} {"Vous" if msg["role"] == "user" else "Assistant"}</strong><br>
                        {msg["content"]}
                        <div style="font-size: 11px; color: #888; margin-top: 8px; {'text-align: right;' if msg['role'] == 'user' else ''}">{msg.get("time", "")}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Indicateur de frappe
        if st.session_state.is_typing:
            st.markdown("""
            <div class="message assistant">
                <div class="typing-indicator">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Actions rapides
        quick_actions = QUICK_ACTIONS.get(st.session_state.current_page, [])
        if quick_actions and len(st.session_state.chat_history) < 2:
            st.markdown("**💡 Questions suggérées :**")
            for idx, action in enumerate(quick_actions):
                if st.button(action, key=f"qa_{idx}", use_container_width=True):
                    handle_user_message(action)
                    st.rerun()
        
        # Zone de saisie
        with st.form(key="chat_form", clear_on_submit=True):
            user_input = st.text_input(
                "Message",
                placeholder="💬 Posez votre question ici...",
                label_visibility="collapsed",
                key="chat_input"
            )
            
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                submit = st.form_submit_button("📤 Envoyer", use_container_width=True, type="primary")
            with col2:
                if st.form_submit_button("🗑️ Effacer", use_container_width=True):
                    st.session_state.chat_history = []
                    st.rerun()
            with col3:
                if st.form_submit_button("🔄 Reset", use_container_width=True):
                    st.session_state.chat_history = []
                    st.session_state.chat_notifications = 0
                    st.rerun()
            
            if submit and user_input.strip():
                handle_user_message(user_input.strip())
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

# PAGE 1: Chargement des données
if page == "📁 Chargement des Données":
    st.markdown('<p class="sub-header">📂 Chargement et Préparation des Données</p>', unsafe_allow_html=True)
    
    st.info("📌 **Note:** Cette application analyse les données ESG de 195 pays souverains reconnus par l'ONU.")
    
    base_path = r"C:\Users\rabia\OneDrive\Bureau\StreamlitEsg"
    
    load_option = st.radio(
        "**Choisissez votre mode de chargement:**",
        ["📂 Automatique (depuis dossier)", "📤 Manuel (upload fichiers)"],
        horizontal=True
    )
    
    if load_option == "📂 Automatique (depuis dossier)":
        st.success(f"📁 Chemin configuré : `{base_path}`")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Charger et Traiter les Données", type="primary", use_container_width=True):
                try:
                    with st.spinner("🔄 Chargement des fichiers en cours..."):
                        data_E = pd.read_excel(f"{base_path}\\environment.xlsx")
                        data_S = pd.read_excel(f"{base_path}\\social.xlsx")
                        data_G = pd.read_excel(f"{base_path}\\governance.xlsx")
                        
                        st.success("✅ Fichiers chargés avec succès!")
                        
                    with st.spinner("⚙️ Traitement et normalisation..."):
                        df_esg = process_data(data_E, data_S, data_G)
                        st.session_state.df_esg = df_esg
                        st.session_state.data_loaded = True
                        
                        st.success("✅ Données traitées avec succès! (195 pays souverains)")
                        st.balloons()
                        
                    # Métriques
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("📊 Observations", f"{len(df_esg):,}")
                    with col2:
                        st.metric("🌍 Pays", df_esg['Country Code'].nunique())
                    with col3:
                        st.metric("📅 Période", f"{df_esg['Year'].min()}-{df_esg['Year'].max()}")
                    with col4:
                        indicators = len([c for c in df_esg.columns if c.endswith(('_E', '_S', '_G'))])
                        st.metric("📈 Indicateurs", indicators)
                    
                    st.markdown("---")
                    st.markdown("### 👀 Aperçu des données")
                    st.dataframe(df_esg.head(15), use_container_width=True, height=400)
                    
                except FileNotFoundError:
                    st.error(f"❌ Fichiers introuvables dans : `{base_path}`")
                    st.info("💡 Vérifiez que les fichiers `environment.xlsx`, `social.xlsx` et `governance.xlsx` existent dans ce dossier.")
                except Exception as e:
                    st.error(f"❌ Erreur lors du traitement : {str(e)}")
    
    else:
        st.info("📤 Téléchargez les trois fichiers Excel contenant vos données ESG")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            file_E = st.file_uploader("🌱 **Environnement (E)**", type=['xlsx'], key="file_E")
        with col2:
            file_S = st.file_uploader("👥 **Social (S)**", type=['xlsx'], key="file_S")
        with col3:
            file_G = st.file_uploader("⚖️ **Gouvernance (G)**", type=['xlsx'], key="file_G")
        
        if file_E and file_S and file_G:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🚀 Traiter les Données", type="primary", use_container_width=True):
                    with st.spinner("⚙️ Traitement en cours..."):
                        data_E = pd.read_excel(file_E)
                        data_S = pd.read_excel(file_S)
                        data_G = pd.read_excel(file_G)
                        
                        df_esg = process_data(data_E, data_S, data_G)
                        st.session_state.df_esg = df_esg
                        st.session_state.data_loaded = True
                        
                        st.success("✅ Données traitées avec succès!")
                        st.balloons()
                        
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("📊 Observations", f"{len(df_esg):,}")
                    with col2:
                        st.metric("🌍 Pays", df_esg['Country Code'].nunique())
                    with col3:
                        st.metric("📅 Période", f"{df_esg['Year'].min()}-{df_esg['Year'].max()}")
                    with col4:
                        indicators = len([c for c in df_esg.columns if c.endswith(('_E', '_S', '_G'))])
                        st.metric("📈 Indicateurs", indicators)
                    
                    st.markdown("---")
                    st.markdown("### 👀 Aperçu des données")
                    st.dataframe(df_esg.head(15), use_container_width=True, height=400)

# PAGE 2: Scores ESG & Top Pays
elif page == "📈 Scores ESG & Top Pays":
    if not st.session_state.data_loaded:
        st.warning("⚠️ Veuillez d'abord charger les données dans la section '📁 Chargement des Données'")
    else:
        df_esg = st.session_state.df_esg
        st.markdown('<p class="sub-header">🏆 Scores ESG et Classements Mondiaux</p>', unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["🏆 Top Pays", "📊 Distribution des Scores", "📈 Tendances Temporelles"])
        
        with tab1:
            st.markdown("### 🥇 Top 10 des Pays par Score ESG Total")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                year_option = st.selectbox(
                    "**Sélectionnez l'année d'analyse:**",
                    ['Dernière année disponible'] + sorted(df_esg['Year'].unique().tolist(), reverse=True)
                )
            
            if year_option == 'Dernière année disponible':
                df_latest = df_esg.sort_values('Year', ascending=False).drop_duplicates(subset=['Country Name'])
                year_display = df_latest['Year'].mode()[0]
            else:
                df_latest = df_esg[df_esg['Year'] == year_option]
                year_display = year_option
            
            df_top_10 = df_latest.nlargest(10, 'Score_ESG_Total').reset_index(drop=True)
            df_top_10.index = df_top_10.index + 1
            
            # Graphique moderne
            fig = px.bar(
                df_top_10, 
                x='Country Name', 
                y='Score_ESG_Total',
                color='Score_ESG_Total',
                title=f"🏆 Top 10 Pays - Score ESG Total ({year_display})",
                color_continuous_scale='Viridis',
                text='Score_ESG_Total'
            )
            fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            fig.update_layout(
                height=500,
                xaxis_title="Pays",
                yaxis_title="Score ESG Total",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Tableau détaillé
            st.markdown("### 📋 Détails des Scores")
            df_display = df_top_10[['Country Name', 'Year', 'Score_E', 'Score_S', 'Score_G', 'Score_ESG_Total']].copy()
            df_display.columns = ['Pays', 'Année', 'Environnement', 'Social', 'Gouvernance', 'ESG Total']
            st.dataframe(
                df_display.style.background_gradient(subset=['Environnement', 'Social', 'Gouvernance', 'ESG Total'], cmap='RdYlGn'),
                use_container_width=True
            )
            
            # Comparaison E, S, G - Top 3
            st.markdown("### 📊 Comparaison Détaillée E, S, G - Top 3")
            df_top_3 = df_top_10.head(3)
            
            fig = go.Figure()
            colors = ['#2ecc71', '#3498db', '#9b59b6']
            for idx, (_, row) in enumerate(df_top_3.iterrows()):
                fig.add_trace(go.Bar(
                    name=row['Country Name'],
                    x=['🌱 Environnement', '👥 Social', '⚖️ Gouvernance'],
                    y=[row['Score_E'], row['Score_S'], row['Score_G']],
                    marker_color=colors[idx],
                    text=[f"{row['Score_E']:.3f}", f"{row['Score_S']:.3f}", f"{row['Score_G']:.3f}"],
                    textposition='outside'
                ))
            
            fig.update_layout(
                barmode='group',
                height=450,
                title="Décomposition des Scores E, S, G des 3 Premiers Pays",
                yaxis_title="Score",
                xaxis_title="Piliers ESG"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.markdown("### 📊 Distribution et Statistiques des Scores")
            
            score_type = st.selectbox(
                "**Choisissez le score à analyser:**",
                ['Score_ESG_Total', 'Score_E', 'Score_S', 'Score_G'],
                format_func=lambda x: {
                    'Score_ESG_Total': '🎯 Score ESG Total',
                    'Score_E': '🌱 Score Environnement',
                    'Score_S': '👥 Score Social',
                    'Score_G': '⚖️ Score Gouvernance'
                }[x]
            )
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig = px.histogram(
                    df_esg,
                    x=score_type,
                    nbins=50,
                    title=f"Distribution du {score_type}",
                    marginal="box",
                    color_discrete_sequence=['#667eea']
                )
                fig.update_layout(height=450, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### 📈 Statistiques Clés")
                stats_data = {
                    "Métrique": ["Moyenne", "Médiane", "Écart-type", "Minimum", "Maximum", "Q1 (25%)", "Q3 (75%)"],
                    "Valeur": [
                        f"{df_esg[score_type].mean():.4f}",
                        f"{df_esg[score_type].median():.4f}",
                        f"{df_esg[score_type].std():.4f}",
                        f"{df_esg[score_type].min():.4f}",
                        f"{df_esg[score_type].max():.4f}",
                        f"{df_esg[score_type].quantile(0.25):.4f}",
                        f"{df_esg[score_type].quantile(0.75):.4f}"
                    ]
                }
                st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)
            
            st.markdown("### 🎯 Répartition des Catégories ESG")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.pie(
                    df_esg,
                    names='ESG_Category',
                    title="Distribution des Catégories",
                    color='ESG_Category',
                    color_discrete_map={'Faible':'#ff6b6b', 'Moyen':'#feca57', 'Élevé':'#48dbfb'},
                    hole=0.4
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                category_counts = df_esg['ESG_Category'].value_counts()
                fig = px.bar(
                    x=category_counts.index,
                    y=category_counts.values,
                    title="Nombre de Pays par Catégorie",
                    color=category_counts.index,
                    color_discrete_map={'Faible':'#ff6b6b', 'Moyen':'#feca57', 'Élevé':'#48dbfb'},
                    text=category_counts.values
                )
                fig.update_traces(textposition='outside')
                fig.update_layout(showlegend=False, xaxis_title="Catégorie", yaxis_title="Nombre de Pays")
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.markdown("### 📈 Évolution Temporelle des Scores ESG")
            
            sample_countries = st.multiselect(
                "**Sélectionnez jusqu'à 5 pays à comparer:**",
                sorted(df_esg['Country Name'].unique()),
                default=df_latest.nlargest(3, 'Score_ESG_Total')['Country Name'].tolist()[:3],
                max_selections=5
            )
            
            if sample_countries:
                df_trend = df_esg[df_esg['Country Name'].isin(sample_countries)]
                
                fig = go.Figure()
                for country in sample_countries:
                    df_country = df_trend[df_trend['Country Name'] == country].sort_values('Year')
                    fig.add_trace(go.Scatter(
                        x=df_country['Year'],
                        y=df_country['Score_ESG_Total'],
                        mode='lines+markers',
                        name=country,
                        line=dict(width=3),
                        marker=dict(size=8)
                    ))
                
                fig.update_layout(
                    title="Évolution du Score ESG Total au Fil du Temps",
                    xaxis_title="Année",
                    yaxis_title="Score ESG Total",
                    height=500,
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Analyse des tendances
                st.markdown("### 📊 Analyse des Tendances")
                trend_analysis = []
                for country in sample_countries:
                    df_country = df_trend[df_trend['Country Name'] == country].sort_values('Year')
                    if len(df_country) > 1:
                        first_score = df_country.iloc[0]['Score_ESG_Total']
                        last_score = df_country.iloc[-1]['Score_ESG_Total']
                        change = last_score - first_score
                        change_pct = (change / first_score) * 100 if first_score != 0 else 0
                        trend_analysis.append({
                            'Pays': country,
                            'Score Initial': f"{first_score:.3f}",
                            'Score Final': f"{last_score:.3f}",
                            'Évolution': f"{change:+.3f}",
                            'Évolution %': f"{change_pct:+.2f}%",
                            'Tendance': '📈' if change > 0 else '📉' if change < 0 else '➡️'
                        })
                
                if trend_analysis:
                    st.dataframe(pd.DataFrame(trend_analysis), use_container_width=True, hide_index=True)
            else:
                st.info("👆 Sélectionnez au moins un pays pour visualiser l'évolution temporelle")

# PAGE 3: Analyse Exploratoire
elif page == "🔍 Analyse Exploratoire":
    if not st.session_state.data_loaded:
        st.warning("⚠️ Veuillez d'abord charger les données")
    else:
        df_esg = st.session_state.df_esg
        st.markdown('<p class="sub-header">🔬 Analyse Exploratoire Approfondie</p>', unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["📊 Statistiques Descriptives", "🗺️ Heatmap Corrélation", "📈 Évolution Détaillée"])
        
        with tab1:
            st.markdown("### 📈 Statistiques Descriptives Complètes")
            
            stats = df_esg[['Score_E', 'Score_S', 'Score_G', 'Score_ESG_Total']].describe().T
            stats.columns = ['Nombre', 'Moyenne', 'Écart-type', 'Min', '25%', '50% (Médiane)', '75%', 'Max']
            stats.index = ['🌱 Environnement', '👥 Social', '⚖️ Gouvernance', '🎯 ESG Total']
            
            st.dataframe(
                stats.style.background_gradient(cmap='RdYlGn', axis=1),
                use_container_width=True
            )
            
            st.markdown("### 🔗 Matrice de Corrélation entre Piliers ESG")
            
            corr = df_esg[['Score_E', 'Score_S', 'Score_G', 'Score_ESG_Total']].corr()
            
            fig = px.imshow(
                corr,
                text_auto='.3f',
                aspect="auto",
                color_continuous_scale='RdBu_r',
                title="Corrélations entre les Scores E, S, G et ESG Total",
                labels=dict(x="Score", y="Score", color="Corrélation")
            )
            fig.update_xaxes(side="bottom")
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### 📊 Boxplots Comparatifs")
            
            df_melted = df_esg.melt(
                value_vars=['Score_E', 'Score_S', 'Score_G'],
                var_name='Pilier',
                value_name='Score'
            )
            df_melted['Pilier'] = df_melted['Pilier'].replace({
                'Score_E': '🌱 Environnement',
                'Score_S': '👥 Social',
                'Score_G': '⚖️ Gouvernance'
            })
            
            fig = px.box(
                df_melted,
                x='Pilier',
                y='Score',
                color='Pilier',
                title="Distribution Comparative des Scores par Pilier",
                color_discrete_map={
                    '🌱 Environnement': '#2ecc71',
                    '👥 Social': '#3498db',
                    '⚖️ Gouvernance': '#9b59b6'
                }
            )
            fig.update_layout(height=450, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.markdown("### 🗺️ Heatmap des Scores ESG Temporels")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                n_countries = st.slider(
                    "**Nombre de pays à afficher:**",
                    min_value=10,
                    max_value=50,
                    value=20,
                    step=5
                )
            
            with col2:
                sort_by = st.selectbox(
                    "**Trier par:**",
                    ['Score ESG', 'Nom alphabétique'],
                    key='heatmap_sort'
                )
            
            if sort_by == 'Score ESG':
                top_countries = df_esg.groupby('Country Name')['Score_ESG_Total'].mean().nlargest(n_countries).index
            else:
                top_countries = sorted(df_esg['Country Name'].unique())[:n_countries]
            
            df_heat = df_esg[df_esg['Country Name'].isin(top_countries)]
            pivot_data = df_heat.pivot_table(
                values='Score_ESG_Total',
                index='Country Name',
                columns='Year'
            )
            
            fig = px.imshow(
                pivot_data,
                labels=dict(x="Année", y="Pays", color="Score ESG"),
                color_continuous_scale='RdYlGn',
                aspect="auto",
                title=f"Évolution des Scores ESG - Top {n_countries} Pays"
            )
            fig.update_layout(height=800)
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("💡 **Interprétation:** Les couleurs vertes indiquent des scores élevés, les rouges des scores faibles. Cette heatmap permet d'identifier rapidement les pays performants et leur évolution dans le temps.")
        
        with tab3:
            st.markdown("### 📈 Analyse Détaillée par Pays")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                country = st.selectbox(
                    "**Sélectionnez un pays:**",
                    sorted(df_esg['Country Name'].unique())
                )
            
            with col2:
                show_all = st.checkbox("Afficher tous les piliers", value=True)
            
            df_country = df_esg[df_esg['Country Name'] == country].sort_values('Year')
            
            if not df_country.empty:
                fig = go.Figure()
                
                if show_all:
                    fig.add_trace(go.Scatter(
                        x=df_country['Year'],
                        y=df_country['Score_E'],
                        mode='lines+markers',
                        name='🌱 Environnement',
                        line=dict(color='#2ecc71', width=2),
                        marker=dict(size=8)
                    ))
                    fig.add_trace(go.Scatter(
                        x=df_country['Year'],
                        y=df_country['Score_S'],
                        mode='lines+markers',
                        name='👥 Social',
                        line=dict(color='#3498db', width=2),
                        marker=dict(size=8)
                    ))
                    fig.add_trace(go.Scatter(
                        x=df_country['Year'],
                        y=df_country['Score_G'],
                        mode='lines+markers',
                        name='⚖️ Gouvernance',
                        line=dict(color='#9b59b6', width=2),
                        marker=dict(size=8)
                    ))
                
                fig.add_trace(go.Scatter(
                    x=df_country['Year'],
                    y=df_country['Score_ESG_Total'],
                    mode='lines+markers',
                    name='🎯 ESG Total',
                    line=dict(color='#e74c3c', width=3),
                    marker=dict(size=10)
                ))
                
                fig.update_layout(
                    title=f"Évolution des Scores ESG - {country}",
                    xaxis_title="Année",
                    yaxis_title="Score",
                    height=500,
                    hovermode='x unified',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Statistiques du pays
                st.markdown(f"### 📊 Statistiques - {country}")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Score ESG Moyen",
                        f"{df_country['Score_ESG_Total'].mean():.3f}",
                        f"{df_country['Score_ESG_Total'].iloc[-1] - df_country['Score_ESG_Total'].iloc[0]:+.3f}"
                    )
                
                with col2:
                    st.metric(
                        "Environnement Moyen",
                        f"{df_country['Score_E'].mean():.3f}"
                    )
                
                with col3:
                    st.metric(
                        "Social Moyen",
                        f"{df_country['Score_S'].mean():.3f}"
                    )
                
                with col4:
                    st.metric(
                        "Gouvernance Moyen",
                        f"{df_country['Score_G'].mean():.3f}"
                    )
                
                # Volatilité
                volatility = df_country['Volatilite_Score_ESG'].iloc[0]
                st.metric(
                    "📉 Volatilité (Stabilité)",
                    f"{volatility:.4f}",
                    "Plus faible = plus stable"
                )
            else:
                st.warning(f"Aucune donnée disponible pour {country}")

# PAGE 4: Volatilité & Comparaisons
elif page == "📊 Volatilité & Comparaisons":
    if not st.session_state.data_loaded:
        st.warning("⚠️ Veuillez d'abord charger les données")
    else:
        df_esg = st.session_state.df_esg
        st.markdown('<p class="sub-header">📉 Analyse de Volatilité et Comparaisons Internationales</p>', unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["📉 Volatilité des Scores", "🔍 Comparaison Multi-Pays"])
        
        with tab1:
            st.markdown("### 📉 Analyse de la Volatilité ESG")
            
            st.info("💡 **La volatilité mesure l'écart-type du score ESG au fil du temps.** Plus elle est élevée, plus le score est instable. Une faible volatilité indique une performance constante.")
            
            df_volatility = df_esg.groupby('Country Name').agg({
                'Score_ESG_Total': ['mean', 'std'],
                'Volatilite_Score_ESG': 'first'
            }).reset_index()
            df_volatility.columns = ['Country Name', 'Score_Moyen', 'Score_Std', 'Volatilite']
            df_volatility = df_volatility.sort_values('Volatilite', ascending=False)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                n_display = st.slider("Nombre de pays à afficher:", 10, 30, 15)
                
                fig = px.bar(
                    df_volatility.head(n_display),
                    x='Volatilite',
                    y='Country Name',
                    orientation='h',
                    title=f"Top {n_display} Pays avec la Plus Forte Volatilité",
                    color='Volatilite',
                    color_continuous_scale='Reds',
                    text='Volatilite'
                )
                fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
                fig.update_layout(height=600, yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### 🔝 Top 10 Plus Volatiles")
                st.dataframe(
                    df_volatility[['Country Name', 'Volatilite']].head(10).reset_index(drop=True),
                    use_container_width=True,
                    hide_index=False
                )
                
                st.markdown("#### ✅ Top 10 Plus Stables")
                st.dataframe(
                    df_volatility[['Country Name', 'Volatilite']].tail(10).reset_index(drop=True),
                    use_container_width=True,
                    hide_index=False
                )
            
            # Scatter plot: Score moyen vs Volatilité
            st.markdown("### 📊 Score Moyen vs Volatilité")
            
            fig = px.scatter(
                df_volatility,
                x='Score_Moyen',
                y='Volatilite',
                hover_data=['Country Name'],
                title="Relation entre Score ESG Moyen et Volatilité",
                labels={'Score_Moyen': 'Score ESG Moyen', 'Volatilite': 'Volatilité'},
                color='Score_Moyen',
                color_continuous_scale='Viridis',
                size='Volatilite',
                size_max=15
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("💡 **Interprétation:** Les pays en haut à gauche ont un faible score mais une forte volatilité (instables et faibles). Les pays en bas à droite ont un score élevé et une faible volatilité (stables et performants) - ce sont les meilleurs profils.")
        
        with tab2:
            st.markdown("### 🔍 Comparaison Personnalisée Multi-Pays")
            
            available_countries = sorted(df_esg['Country Name'].unique())
            
            # Sélection des pays par défaut
            default_countries = []
            suggested = ['China', 'India', 'Japan', 'Korea, Rep.', 'United States', 'Germany', 'France', 'United Kingdom']
            for c in suggested:
                if c in available_countries:
                    default_countries.append(c)
                    if len(default_countries) >= 4:
                        break
            
            if len(default_countries) == 0:
                default_countries = available_countries[:min(4, len(available_countries))]
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                countries_to_compare = st.multiselect(
                    "**Sélectionnez 2 à 6 pays à comparer:**",
                    available_countries,
                    default=default_countries,
                    max_selections=6
                )
            
            with col2:
                year_compare = st.selectbox(
                    "**Année:**",
                    ['Moyenne de toutes les années'] + sorted(df_esg['Year'].unique().tolist(), reverse=True)
                )
            
            if len(countries_to_compare) >= 2:
                if year_compare == 'Moyenne de toutes les années':
                    df_compare = df_esg[df_esg['Country Name'].isin(countries_to_compare)].groupby('Country Name')[
                        ['Score_E', 'Score_S', 'Score_G', 'Score_ESG_Total']].mean().reset_index()
                else:
                    df_compare = df_esg[
                        (df_esg['Country Name'].isin(countries_to_compare)) &
                        (df_esg['Year'] == year_compare)
                    ]
                
                if not df_compare.empty:
                    # Graphique en barres groupées
                    st.markdown("### 📊 Comparaison des Scores E, S, G")
                    
                    df_compare_melt = df_compare.melt(
                        id_vars='Country Name',
                        value_vars=['Score_E', 'Score_S', 'Score_G'],
                        var_name='Pilier',
                        value_name='Score'
                    )
                    df_compare_melt['Pilier'] = df_compare_melt['Pilier'].replace({
                        'Score_E': '🌱 Environnement',
                        'Score_S': '👥 Social',
                        'Score_G': '⚖️ Gouvernance'
                    })
                    
                    fig = px.bar(
                        df_compare_melt,
                        x='Country Name',
                        y='Score',
                        color='Pilier',
                        barmode='group',
                        title=f"Comparaison des Scores ESG - {year_compare}",
                        color_discrete_map={
                            '🌱 Environnement': '#2ecc71',
                            '👥 Social': '#3498db',
                            '⚖️ Gouvernance': '#9b59b6'
                        },
                        text='Score'
                    )
                    fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
                    fig.update_layout(height=500, xaxis_title="Pays", yaxis_title="Score")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Graphique Radar
                    st.markdown("### 📊 Graphique Radar Multi-Dimensionnel")
                    
                    fig = go.Figure()
                    
                    colors = ['#667eea', '#764ba2', '#f093fb', '#4facfe', '#43e97b', '#fa709a']
                    
                    for idx, country in enumerate(countries_to_compare):
                        country_data = df_compare[df_compare['Country Name'] == country]
                        if not country_data.empty:
                            row = country_data.iloc[0]
                            fig.add_trace(go.Scatterpolar(
                                r=[row['Score_E'], row['Score_S'], row['Score_G']],
                                theta=['🌱 Environnement', '👥 Social', '⚖️ Gouvernance'],
                                fill='toself',
                                name=country,
                                line=dict(color=colors[idx % len(colors)], width=2)
                            ))
                    
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 1]
                            )
                        ),
                        height=500,
                        title="Comparaison Radar des Piliers ESG"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Tableau comparatif
                    st.markdown("### 📋 Tableau Comparatif Détaillé")
                    
                    df_display = df_compare[['Country Name', 'Score_E', 'Score_S', 'Score_G', 'Score_ESG_Total']].copy()
                    df_display.columns = ['Pays', '🌱 Environnement', '👥 Social', '⚖️ Gouvernance', '🎯 ESG Total']
                    df_display = df_display.sort_values('🎯 ESG Total', ascending=False).reset_index(drop=True)
                    df_display.index = df_display.index + 1
                    
                    st.dataframe(
                        df_display.style.background_gradient(
                            subset=['🌱 Environnement', '👥 Social', '⚖️ Gouvernance', '🎯 ESG Total'],
                            cmap='RdYlGn'
                        ),
                        use_container_width=True
                    )
                    
                    # Analyse comparative
                    st.markdown("### 🔍 Analyse Comparative")
                    
                    best_country = df_compare.loc[df_compare['Score_ESG_Total'].idxmax()]
                    worst_country = df_compare.loc[df_compare['Score_ESG_Total'].idxmin()]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.success(f"""
                        **🏆 Meilleur Pays: {best_country['Country Name']}**
                        - Score ESG: {best_country['Score_ESG_Total']:.3f}
                        - Environnement: {best_country['Score_E']:.3f}
                        - Social: {best_country['Score_S']:.3f}
                        - Gouvernance: {best_country['Score_G']:.3f}
                        """)
                    
                    with col2:
                        st.warning(f"""
                        **📊 Pays à Améliorer: {worst_country['Country Name']}**
                        - Score ESG: {worst_country['Score_ESG_Total']:.3f}
                        - Environnement: {worst_country['Score_E']:.3f}
                        - Social: {worst_country['Score_S']:.3f}
                        - Gouvernance: {worst_country['Score_G']:.3f}
                        """)
                else:
                    st.warning("Aucune donnée disponible pour cette sélection")
            elif len(countries_to_compare) == 1:
                st.info("👆 Sélectionnez au moins 2 pays pour effectuer une comparaison")
            else:
                st.info("👆 Sélectionnez des pays à comparer dans la liste ci-dessus")

# PAGE 5: Feature Importance
elif page == "🎯 Feature Importance":
    if not st.session_state.data_loaded:
        st.warning("⚠️ Veuillez d'abord charger les données")
    else:
        df_esg = st.session_state.df_esg
        st.markdown('<p class="sub-header">🎯 Analyse de l\'Importance des Caractéristiques</p>', unsafe_allow_html=True)
        
        st.info("💡 Cette analyse identifie quels indicateurs ESG ont le plus d'impact sur le score global, en utilisant plusieurs méthodes de data science.")
        
        # Préparation des données
        feature_cols = [col for col in df_esg.columns if col.endswith(('_E', '_S', '_G'))]
        
        if len(feature_cols) > 100:
            st.warning(f"⚠️ {len(feature_cols)} indicateurs détectés. Sélection des 100 plus significatifs...")
            X_temp = df_esg[feature_cols].fillna(0)
            variance = X_temp.var()
            top_features = variance.nlargest(100).index.tolist()
            feature_cols = top_features
        
        X = df_esg[feature_cols].fillna(0)
        y = df_esg['ESG_Target']
        
        if len(X) > 1000:
            X_sample = X.sample(n=1000, random_state=42)
            y_sample = y.loc[X_sample.index]
        else:
            X_sample = X
            y_sample = y
        
        tab1, tab2, tab3 = st.tabs(["🌳 XGBoost", "🔄 Permutation & RFE", "📊 ANOVA F-test"])
        
        with tab1:
            st.markdown("### 🌳 Feature Importance - XGBoost")
            
            with st.spinner("🔄 Entraînement du modèle XGBoost..."):
                model_xgb = xgb.XGBClassifier(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
                model_xgb.fit(X_sample, y_sample)
                
                importance_xgb = pd.DataFrame({
                    'Feature': feature_cols,
                    'Importance': model_xgb.feature_importances_
                }).sort_values('Importance', ascending=False).head(20)
            
            fig = px.bar(
                importance_xgb,
                x='Importance',
                y='Feature',
                orientation='h',
                title="Top 20 Features - XGBoost Importance",
                color='Importance',
                color_continuous_scale='Viridis',
                text='Importance'
            )
            fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
            fig.update_layout(height=600, yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(importance_xgb.reset_index(drop=True), use_container_width=True)
            
            st.success(f"✅ Modèle entraîné sur {len(X_sample)} observations avec {len(feature_cols)} features")
        
        with tab2:
            st.markdown("### 🔄 Permutation Importance & RFE")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Permutation Importance")
                
                with st.spinner("🔄 Calcul de Permutation Importance..."):
                    model_rf = RandomForestClassifier(n_estimators=30, max_depth=10, random_state=42, n_jobs=-1)
                    model_rf.fit(X_sample, y_sample)
                    
                    perm_importance = permutation_importance(
                        model_rf, X_sample, y_sample,
                        n_repeats=5, random_state=42, n_jobs=-1
                    )
                    
                    importance_perm = pd.DataFrame({
                        'Feature': feature_cols,
                        'Importance': perm_importance.importances_mean
                    }).sort_values('Importance', ascending=False).head(15)
                
                fig = px.bar(
                    importance_perm,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title="Top 15 - Permutation Importance",
                    color='Importance',
                    color_continuous_scale='Plasma'
                )
                fig.update_layout(height=500, yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### RFE (Recursive Feature Elimination)")
                
                with st.spinner("🔄 Calcul de RFE..."):
                    n_features_to_select = min(15, len(feature_cols))
                    rfe = RFE(estimator=model_rf, n_features_to_select=n_features_to_select)
                    rfe.fit(X_sample, y_sample)
                    
                    rfe_features = pd.DataFrame({
                        'Feature': feature_cols,
                        'Selected': rfe.support_,
                        'Ranking': rfe.ranking_
                    }).sort_values('Ranking').head(15)
                
                fig = px.bar(
                    rfe_features,
                    x='Ranking',
                    y='Feature',
                    orientation='h',
                    title="Top 15 - RFE Ranking (1 = meilleur)",
                    color='Ranking',
                    color_continuous_scale='Viridis_r'
                )
                fig.update_layout(height=500, yaxis={'categoryorder':'total descending'})
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.markdown("### 📊 ANOVA F-test pour Classification")
            
            with st.spinner("🔄 Calcul des F-scores ANOVA..."):
                f_scores = []
                p_values = []
                
                for col in feature_cols:
                    groups = [X_sample[y_sample == i][col].dropna() for i in y_sample.unique()]
                    groups = [g for g in groups if len(g) > 0]
                    
                    if len(groups) >= 2:
                        f_stat, p_val = f_oneway(*groups)
                        f_scores.append(f_stat)
                        p_values.append(p_val)
                    else:
                        f_scores.append(0)
                        p_values.append(1)
                
                anova_results = pd.DataFrame({
                    'Feature': feature_cols,
                    'F-Score': f_scores,
                    'P-Value': p_values
                }).sort_values('F-Score', ascending=False).head(20)
            
            fig = px.bar(
                anova_results,
                x='F-Score',
                y='Feature',
                orientation='h',
                title="Top 20 Features - ANOVA F-test",
                color='F-Score',
                color_continuous_scale='Sunset',
                text='F-Score',
                hover_data=['P-Value']
            )
            fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            fig.update_layout(height=600, yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(anova_results.reset_index(drop=True), use_container_width=True)
            
            st.info("💡 **Interprétation:** Un F-Score élevé et une P-Value faible (<0.05) indiquent que la feature est significativement différente entre les catégories ESG.")

# PAGE 6: Machine Learning
if page == "🤖 Machine Learning":
    if not st.session_state.data_loaded:
        st.warning("⚠️ Veuillez d'abord charger les données")
    else:
        df_esg = st.session_state.df_esg
        st.markdown('<p class="sub-header">🤖 Modélisation par Machine Learning</p>', unsafe_allow_html=True)
        
        st.info("💡 Cette section utilise Random Forest pour prédire les catégories ESG (Faible/Moyen/Élevé) basées sur les indicateurs.")
        
        # Préparation des données
        feature_cols = [col for col in df_esg.columns if col.endswith(('_E', '_S', '_G'))]
        
        if len(feature_cols) > 100:
            st.warning(f"⚠️ Réduction de {len(feature_cols)} features à 100 pour optimiser le temps de calcul")
            X_temp = df_esg[feature_cols].fillna(0)
            variance = X_temp.var()
            top_features = variance.nlargest(100).index.tolist()
            feature_cols = top_features
        
        X = df_esg[feature_cols].fillna(0)
        y = df_esg['ESG_Target']
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            test_size = st.slider("**Taille du jeu de test (%):**", 10, 40, 20) / 100
        
        with col2:
            n_estimators = st.selectbox("**Nombre d'arbres:**", [50, 100, 150, 200], index=1)
        
        if st.button("🚀 Entraîner le Modèle Random Forest", type="primary", use_container_width=True):
            with st.spinner("🔄 Entraînement en cours..."):
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=y
                )
                
                model_rf = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=15,
                    random_state=42,
                    n_jobs=-1
                )
                model_rf.fit(X_train, y_train)
                
                y_pred = model_rf.predict(X_test)
                
                st.success("✅ Modèle entraîné avec succès!")
                
                # Métriques
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("🎯 Accuracy", f"{accuracy:.3f}")
                with col2:
                    st.metric("🎯 Precision", f"{precision:.3f}")
                with col3:
                    st.metric("🎯 Recall", f"{recall:.3f}")
                with col4:
                    st.metric("🎯 F1-Score", f"{f1:.3f}")
                
                # Matrice de confusion
                st.markdown("### 📊 Matrice de Confusion")
                
                cm = confusion_matrix(y_test, y_pred)
                labels = ['Faible', 'Moyen', 'Élevé']
                
                fig = px.imshow(
                    cm,
                    labels=dict(x="Prédiction", y="Réalité", color="Nombre"),
                    x=labels,
                    y=labels,
                    text_auto=True,
                    color_continuous_scale='Blues',
                    title="Matrice de Confusion"
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Feature Importance
                st.markdown("### 🎯 Importance des Features")
                
                importance_df = pd.DataFrame({
                    'Feature': feature_cols,
                    'Importance': model_rf.feature_importances_
                }).sort_values('Importance', ascending=False).head(20)
                
                fig = px.bar(
                    importance_df,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title="Top 20 Features les Plus Importantes",
                    color='Importance',
                    color_continuous_scale='Viridis',
                    text='Importance'
                )
                fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
                fig.update_layout(height=600, yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
                
                # Rapport de classification
                st.markdown("### 📋 Rapport de Classification Détaillé")
                
                report = classification_report(y_test, y_pred, target_names=labels, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                
                st.dataframe(
                    report_df.style.background_gradient(cmap='RdYlGn', axis=0),
                    use_container_width=True
                )

# PAGE 7: Deep Learning
elif page == "🧠 Deep Learning":
    if not st.session_state.data_loaded:
        st.warning("⚠️ Veuillez d'abord charger les données")
    else:
        df_esg = st.session_state.df_esg
        st.markdown('<p class="sub-header">🧠 Modélisation par Deep Learning</p>', unsafe_allow_html=True)
        
        st.info("💡 Cette section utilise un réseau de neurones profond (TensorFlow) pour la classification ESG.")
        
        # Préparation des données
        feature_cols = [col for col in df_esg.columns if col.endswith(('_E', '_S', '_G'))]
        
        if len(feature_cols) > 100:
            st.warning(f"⚠️ Réduction à 100 features")
            X_temp = df_esg[feature_cols].fillna(0)
            variance = X_temp.var()
            top_features = variance.nlargest(100).index.tolist()
            feature_cols = top_features
        
        X = df_esg[feature_cols].fillna(0)
        y = df_esg['ESG_Target']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            epochs = st.slider("**Nombre d'époques:**", 10, 100, 30, 10)
        with col2:
            batch_size = st.selectbox("**Batch size:**", [16, 32, 64, 128], index=1)
        with col3:
            dropout_rate = st.slider("**Dropout rate:**", 0.1, 0.5, 0.3, 0.1)
        
        if st.button("🚀 Entraîner le Réseau de Neurones", type="primary", use_container_width=True):
            with st.spinner("🔄 Construction et entraînement du modèle..."):
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Construction du modèle
                model_nn = Sequential([
                    Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
                    Dropout(dropout_rate),
                    Dense(64, activation='relu'),
                    Dropout(dropout_rate),
                    Dense(32, activation='relu'),
                    Dropout(dropout_rate),
                    Dense(3, activation='softmax')
                ])
                
                model_nn.compile(
                    optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy']
                )
                
                # Entraînement
                history = model_nn.fit(
                    X_train_scaled, y_train,
                    validation_split=0.2,
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=0
                )
                
                st.success("✅ Modèle entraîné avec succès!")
                
                # Évaluation
                test_loss, test_accuracy = model_nn.evaluate(X_test_scaled, y_test, verbose=0)
                y_pred_proba = model_nn.predict(X_test_scaled, verbose=0)
                y_pred = np.argmax(y_pred_proba, axis=1)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("🎯 Test Accuracy", f"{test_accuracy:.3f}")
                with col2:
                    st.metric("📉 Test Loss", f"{test_loss:.3f}")
                
                # Courbes d'apprentissage
                st.markdown("### 📈 Courbes d'Apprentissage")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        y=history.history['accuracy'],
                        mode='lines+markers',
                        name='Train Accuracy',
                        line=dict(color='#2ecc71', width=2)
                    ))
                    fig.add_trace(go.Scatter(
                        y=history.history['val_accuracy'],
                        mode='lines+markers',
                        name='Validation Accuracy',
                        line=dict(color='#e74c3c', width=2)
                    ))
                    fig.update_layout(
                        title="Évolution de l'Accuracy",
                        xaxis_title="Époque",
                        yaxis_title="Accuracy",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        y=history.history['loss'],
                        mode='lines+markers',
                        name='Train Loss',
                        line=dict(color='#3498db', width=2)
                    ))
                    fig.add_trace(go.Scatter(
                        y=history.history['val_loss'],
                        mode='lines+markers',
                        name='Validation Loss',
                        line=dict(color='#f39c12', width=2)
                    ))
                    fig.update_layout(
                        title="Évolution de la Loss",
                        xaxis_title="Époque",
                        yaxis_title="Loss",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Matrice de confusion
                st.markdown("### 📊 Matrice de Confusion")
                
                cm = confusion_matrix(y_test, y_pred)
                labels = ['Faible', 'Moyen', 'Élevé']
                
                fig = px.imshow(
                    cm,
                    labels=dict(x="Prédiction", y="Réalité", color="Nombre"),
                    x=labels,
                    y=labels,
                    text_auto=True,
                    color_continuous_scale='Purples',
                    title="Matrice de Confusion - Deep Learning"
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Architecture du modèle
                st.markdown("### 🏗️ Architecture du Modèle")
                
                model_summary = []
                model_nn.summary(print_fn=lambda x: model_summary.append(x))
                st.code('\n'.join(model_summary), language='text')

# PAGE 8: Clustering
elif page == "🎨 Clustering":
    if not st.session_state.data_loaded:
        st.warning("⚠️ Veuillez d'abord charger les données")
    else:
        df_esg = st.session_state.df_esg
        st.markdown('<p class="sub-header">🎨 Clustering K-Means des Pays</p>', unsafe_allow_html=True)
        
        st.info("💡 Le clustering K-Means regroupe les pays similaires en clusters basés sur leurs scores ESG.")
        
        # Préparation des données
        df_cluster = df_esg.groupby('Country Name')[['Score_E', 'Score_S', 'Score_G', 'Score_ESG_Total']].mean().reset_index()
        
        X_cluster = df_cluster[['Score_E', 'Score_S', 'Score_G']].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_cluster)
        
        tab1, tab2 = st.tabs(["📊 Méthode du Coude", "🎨 Visualisation des Clusters"])
        
        with tab1:
            st.markdown("### 📊 Méthode du Coude (Elbow Method)")
            
            st.info("💡 La méthode du coude aide à déterminer le nombre optimal de clusters en identifiant le point où l'inertie cesse de diminuer significativement.")
            
            with st.spinner("🔄 Calcul de l'inertie pour différents nombres de clusters..."):
                inertias = []
                K_range = range(2, 11)
                
                for k in K_range:
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    kmeans.fit(X_scaled)
                    inertias.append(kmeans.inertia_)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(K_range),
                    y=inertias,
                    mode='lines+markers',
                    line=dict(color='#667eea', width=3),
                    marker=dict(size=10, color='#764ba2')
                ))
                fig.update_layout(
                    title="Méthode du Coude - Choix du Nombre de Clusters",
                    xaxis_title="Nombre de Clusters (K)",
                    yaxis_title="Inertie",
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.success("✅ Le coude optimal se situe généralement entre 3 et 5 clusters")
        
        with tab2:
            st.markdown("### 🎨 Clustering et Visualisation")
            
            n_clusters = st.slider("**Choisissez le nombre de clusters:**", 2, 10, 4)
            
            if st.button("🚀 Appliquer le Clustering", type="primary"):
                with st.spinner("🔄 Application du K-Means..."):
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    df_cluster['Cluster'] = kmeans.fit_predict(X_scaled)
                    
                    # PCA pour visualisation 2D
                    pca = PCA(n_components=2)
                    X_pca = pca.fit_transform(X_scaled)
                    
                    df_cluster['PCA1'] = X_pca[:, 0]
                    df_cluster['PCA2'] = X_pca[:, 1]
                    
                    st.success(f"✅ {n_clusters} clusters identifiés!")
                    
                    # Visualisation PCA
                    st.markdown("### 📊 Visualisation PCA 2D des Clusters")
                    
                    fig = px.scatter(
                        df_cluster,
                        x='PCA1',
                        y='PCA2',
                        color='Cluster',
                        hover_data=['Country Name', 'Score_ESG_Total'],
                        title=f"Clustering K-Means avec {n_clusters} Clusters (PCA 2D)",
                        color_continuous_scale='Viridis',
                        size='Score_ESG_Total',
                        size_max=15
                    )
                    fig.update_layout(height=600)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.info(f"💡 **Variance expliquée par PCA:** PC1 = {pca.explained_variance_ratio_[0]:.2%}, PC2 = {pca.explained_variance_ratio_[1]:.2%}")
                    
                    # Graphique 3D
                    st.markdown("### 🌐 Visualisation 3D des Clusters")
                    
                    fig = px.scatter_3d(
                        df_cluster,
                        x='Score_E',
                        y='Score_S',
                        z='Score_G',
                        color='Cluster',
                        hover_data=['Country Name', 'Score_ESG_Total'],
                        title=f"Clusters en 3D (E, S, G)",
                        color_continuous_scale='Viridis',
                        size='Score_ESG_Total',
                        size_max=10
                    )
                    fig.update_layout(height=600)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Analyse par cluster
                    st.markdown("### 📋 Analyse par Cluster")
                    
                    for cluster_id in range(n_clusters):
                        with st.expander(f"🔍 Cluster {cluster_id} ({len(df_cluster[df_cluster['Cluster'] == cluster_id])} pays)"):
                            cluster_data = df_cluster[df_cluster['Cluster'] == cluster_id]
                            
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                st.markdown("**Pays dans ce cluster:**")
                                countries = cluster_data['Country Name'].tolist()
                                st.write(", ".join(countries[:20]) + ("..." if len(countries) > 20 else ""))
                            
                            with col2:
                                st.markdown("**Scores moyens:**")
                                st.metric("ESG Total", f"{cluster_data['Score_ESG_Total'].mean():.3f}")
                                st.metric("Environnement", f"{cluster_data['Score_E'].mean():.3f}")
                                st.metric("Social", f"{cluster_data['Score_S'].mean():.3f}")
                                st.metric("Gouvernance", f"{cluster_data['Score_G'].mean():.3f}")

# PAGE 9: Analyses Régionales
elif page == "🌏 Analyses Régionales":
    if not st.session_state.data_loaded:
        st.warning("⚠️ Veuillez d'abord charger les données")
    else:
        df_esg = st.session_state.df_esg
        st.markdown('<p class="sub-header">🌏 Analyses Comparatives par Région</p>', unsafe_allow_html=True)
        
        # Définition des régions
        regions = {
            'Asie': ['China', 'India', 'Japan', 'Korea, Rep.', 'Indonesia', 'Thailand', 'Vietnam', 'Malaysia', 'Philippines', 'Singapore', 'Bangladesh', 'Pakistan', 'Myanmar', 'Cambodia', 'Lao PDR', 'Mongolia'],
            'Europe': ['Germany', 'France', 'United Kingdom', 'Italy', 'Spain', 'Poland', 'Netherlands', 'Belgium', 'Sweden', 'Austria', 'Switzerland', 'Norway', 'Denmark', 'Finland', 'Ireland', 'Portugal'],
            'Amérique du Nord': ['United States', 'Canada', 'Mexico'],
            'Amérique du Sud': ['Brazil', 'Argentina', 'Chile', 'Colombia', 'Peru', 'Venezuela', 'Ecuador', 'Bolivia', 'Paraguay', 'Uruguay'],
            'Afrique': ['South Africa', 'Nigeria', 'Egypt, Arab Rep.', 'Kenya', 'Ghana', 'Ethiopia', 'Morocco', 'Algeria', 'Tunisia', 'Tanzania'],
            'Moyen-Orient': ['Saudi Arabia', 'United Arab Emirates', 'Qatar', 'Kuwait', 'Oman', 'Bahrain', 'Jordan', 'Lebanon', 'Iran, Islamic Rep.', 'Iraq'],
            'Océanie': ['Australia', 'New Zealand']
        }
        
        # Mapping pays -> région
        country_to_region = {}
        for region, countries in regions.items():
            for country in countries:
                country_to_region[country] = region
        
        df_esg['Region'] = df_esg['Country Name'].map(country_to_region).fillna('Autres')
        
        st.markdown("### 🌍 Comparaison des Scores ESG par Région")
        
        df_regional = df_esg.groupby('Region')[['Score_E', 'Score_S', 'Score_G', 'Score_ESG_Total']].mean().reset_index()
        df_regional = df_regional[df_regional['Region'] != 'Autres'].sort_values('Score_ESG_Total', ascending=False)
        
        # Graphique en barres
        fig = px.bar(
            df_regional,
            x='Region',
            y='Score_ESG_Total',
            color='Score_ESG_Total',
            title="Score ESG Moyen par Région",
            color_continuous_scale='Viridis',
            text='Score_ESG_Total'
        )
        fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        fig.update_layout(height=500, xaxis_title="Région", yaxis_title="Score ESG Total")
        st.plotly_chart(fig, use_container_width=True)
        
        # Comparaison E, S, G par région
        st.markdown("### 📊 Décomposition E, S, G par Région")
        
        df_regional_melt = df_regional.melt(
            id_vars='Region',
            value_vars=['Score_E', 'Score_S', 'Score_G'],
            var_name='Pilier',
            value_name='Score'
        )
        df_regional_melt['Pilier'] = df_regional_melt['Pilier'].replace({
            'Score_E': '🌱 Environnement',
            'Score_S': '👥 Social',
            'Score_G': '⚖️ Gouvernance'
        })
        
        fig = px.bar(
            df_regional_melt,
            x='Region',
            y='Score',
            color='Pilier',
            barmode='group',
            title="Scores E, S, G par Région",
            color_discrete_map={
                '🌱 Environnement': '#2ecc71',
                '👥 Social': '#3498db',
                '⚖️ Gouvernance': '#9b59b6'
            }
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Heatmap régionale
        st.markdown("### 🗺️ Heatmap Régionale")
        
        fig = px.imshow(
            df_regional.set_index('Region')[['Score_E', 'Score_S', 'Score_G', 'Score_ESG_Total']].T,
            labels=dict(x="Région", y="Score", color="Valeur"),
            color_continuous_scale='RdYlGn',
            aspect="auto",
            title="Heatmap des Scores par Région"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Tableau récapitulatif
        st.markdown("### 📋 Tableau Récapitulatif par Région")
        
        df_display = df_regional.copy()
        df_display.columns = ['Région', '🌱 Environnement', '👥 Social', '⚖️ Gouvernance', '🎯 ESG Total']
        df_display = df_display.sort_values('🎯 ESG Total', ascending=False).reset_index(drop=True)
        df_display.index = df_display.index + 1
        
        st.dataframe(
            df_display.style.background_gradient(
                subset=['🌱 Environnement', '👥 Social', '⚖️ Gouvernance', '🎯 ESG Total'],
                cmap='RdYlGn'
            ),
            use_container_width=True
        )
        
        # Insights
        best_region = df_regional.loc[df_regional['Score_ESG_Total'].idxmax()]
        worst_region = df_regional.loc[df_regional['Score_ESG_Total'].idxmin()]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success(f"""
            **🏆 Meilleure Région: {best_region['Region']}**
            - Score ESG: {best_region['Score_ESG_Total']:.3f}
            - Environnement: {best_region['Score_E']:.3f}
            - Social: {best_region['Score_S']:.3f}
            - Gouvernance: {best_region['Score_G']:.3f}
            """)
        
        with col2:
            st.warning(f"""
            **📊 Région à Améliorer: {worst_region['Region']}**
            - Score ESG: {worst_region['Score_ESG_Total']:.3f}
            - Environnement: {worst_region['Score_E']:.3f}
            - Social: {worst_region['Score_S']:.3f}
            - Gouvernance: {worst_region['Score_G']:.3f}
            """)