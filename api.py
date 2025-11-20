"""
====================================================================
APPLICATION STREAMLIT AMÉLIORÉE: Analyse Mobilité Post-JO Paris 2024
====================================================================
Dashboard interactif optimisé pour visualiser l'impact des JO 2024 
sur la mobilité cycliste à Paris

Installation requise:
pip install streamlit pandas numpy plotly scikit-learn requests openpyxl seaborn scipy

Exécution:
streamlit run app_streamlit_ameliore.py --server.port 8501

====================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import pickle
import json
import os
import requests
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION GLOBALE
# ==========================================

# Configuration de la page
st.set_page_config(
    page_title="🚴 Mobilité Paris JO 2024",
    page_icon="🚴",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/votre-repo',
        'Report a bug': "https://github.com/votre-repo/issues",
        'About': "# Analyse de la Mobilité Post-JO Paris 2024\nVersion 2.0"
    }
)

# CSS personnalisé amélioré
st.markdown("""
<style>
    /* Style général */
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
        padding: 1rem;
    }
    
    .sub-header {
        font-size: 1.3rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    
    /* Cartes métriques */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 12px rgba(0, 0, 0, 0.15);
    }
    
    /* Info box */
    .info-box {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1E88E5;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    .success-box {
        background-color: #d4edda;
        border-left-color: #28a745;
    }
    
    .warning-box {
        background-color: #fff3cd;
        border-left-color: #ffc107;
    }
    
    .danger-box {
        background-color: #f8d7da;
        border-left-color: #dc3545;
    }
    
    /* Tabs personnalisés */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 0.5rem 1.5rem;
        border-radius: 5px;
        font-weight: 600;
    }
    
    /* Boutons améliorés */
    .stButton>button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    /* Dataframes */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #666;
        border-top: 2px solid #e9ecef;
        margin-top: 3rem;
    }
    
    /* Loading animation */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .loading {
        animation: pulse 1.5s ease-in-out infinite;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# FONCTIONS UTILITAIRES AMÉLIORÉES
# ==========================================

@st.cache_data(ttl=3600)
def load_data(filepath):
    """Charge les données depuis un fichier CSV avec gestion d'erreurs robuste"""
    try:
        if not os.path.exists(filepath):
            st.warning(f"⚠️ Fichier non trouvé: {filepath}")
            return pd.DataFrame()
        
        # Détection automatique du séparateur
        with open(filepath, 'r') as f:
            first_line = f.readline()
            separator = ',' if ',' in first_line else ';'
        
        df = pd.read_csv(filepath, sep=separator, low_memory=False)
        
        # Parsing des dates si présentes
        date_columns = [col for col in df.columns if 'date' in col.lower()]
        for col in date_columns:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except:
                pass
        
        return df
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement: {e}")
        return pd.DataFrame()

@st.cache_resource
def load_model(model_name):
    """Charge un modèle ML sauvegardé avec vérification"""
    try:
        filepath = f'models/{model_name}'
        if not os.path.exists(filepath):
            st.warning(f"⚠️ Modèle non trouvé: {model_name}")
            return None
        
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"❌ Erreur chargement modèle: {e}")
        return None

@st.cache_data(ttl=300)  # Cache de 5 minutes pour les données temps réel
def fetch_live_data():
    """Récupère les données Vélib' en temps réel avec retry"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            url = "https://opendata.paris.fr/api/records/1.0/search/"
            params = {
                'dataset': 'velib-disponibilite-en-temps-reel',
                'rows': 100,
                'timezone': 'Europe/Paris'
            }
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                records = data.get('records', [])
                if records:
                    return pd.DataFrame([r['fields'] for r in records])
        except Exception as e:
            if attempt == max_retries - 1:
                st.error(f"❌ Erreur API après {max_retries} tentatives: {e}")
    
    return pd.DataFrame()

def create_date_features(df, date_col='date_comptage'):
    """Crée des features temporelles enrichies"""
    if date_col not in df.columns:
        return df
    
    df[date_col] = pd.to_datetime(df[date_col])
    
    df['annee'] = df[date_col].dt.year
    df['mois'] = df[date_col].dt.month
    df['jour'] = df[date_col].dt.day
    df['heure'] = df[date_col].dt.hour
    df['jour_semaine'] = df[date_col].dt.dayofweek
    df['nom_jour'] = df[date_col].dt.day_name()
    df['semaine'] = df[date_col].dt.isocalendar().week
    df['est_weekend'] = df['jour_semaine'].isin([5, 6]).astype(int)
    
    # Saisons
    def get_saison(mois):
        if mois in [12, 1, 2]:
            return 'Hiver'
        elif mois in [3, 4, 5]:
            return 'Printemps'
        elif mois in [6, 7, 8]:
            return 'Été'
        else:
            return 'Automne'
    
    df['saison'] = df['mois'].apply(get_saison)
    
    # Période JO
    jo_start = pd.Timestamp('2024-07-26')
    jo_end = pd.Timestamp('2024-08-11')
    
    df['periode_jo'] = df[date_col].apply(
        lambda x: 'Pendant JO' if jo_start <= x <= jo_end 
        else ('Avant JO' if x < jo_start else 'Après JO')
    )
    
    return df

def format_number(num):
    """Formate les grands nombres avec espaces"""
    return f"{num:,.0f}".replace(',', ' ')

def calculate_statistics(df, column='comptage'):
    """Calcule des statistiques descriptives complètes"""
    if column not in df.columns or df.empty:
        return {}
    
    return {
        'Moyenne': df[column].mean(),
        'Médiane': df[column].median(),
        'Écart-type': df[column].std(),
        'Minimum': df[column].min(),
        'Maximum': df[column].max(),
        'Total': df[column].sum(),
        'Q1': df[column].quantile(0.25),
        'Q3': df[column].quantile(0.75),
        'Observations': len(df)
    }

# ==========================================
# COMPOSANTS RÉUTILISABLES
# ==========================================

def display_metric_card(title, value, delta=None, icon="📊"):
    """Affiche une carte métrique stylisée"""
    delta_html = ""
    if delta is not None:
        color = "#28a745" if delta >= 0 else "#dc3545"
        arrow = "↗" if delta >= 0 else "↘"
        delta_html = f"<p style='color: {color}; font-size: 1.2rem; margin: 0;'>{arrow} {delta:+.1f}%</p>"
    
    st.markdown(f"""
    <div class="metric-card">
        <div style="font-size: 2rem;">{icon}</div>
        <h4 style="margin: 0.5rem 0; font-size: 0.9rem; opacity: 0.9;">{title}</h4>
        <h2 style="margin: 0; font-size: 2rem;">{value}</h2>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

def create_period_comparison_chart(df, periode_col='periode_jo', value_col='comptage'):
    """Crée un graphique de comparaison entre périodes"""
    if periode_col not in df.columns:
        return None
    
    comparison = df.groupby(periode_col)[value_col].agg(['mean', 'median', 'sum']).reset_index()
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Moyenne des passages', 'Total des passages'),
        specs=[[{'type': 'bar'}, {'type': 'pie'}]]
    )
    
    colors = {'Avant JO': '#667eea', 'Pendant JO': '#f5576c', 'Après JO': '#00f2fe'}
    
    # Barres
    for periode in comparison[periode_col]:
        data = comparison[comparison[periode_col] == periode]
        fig.add_trace(
            go.Bar(
                x=[periode],
                y=data['mean'],
                name=periode,
                marker_color=colors.get(periode, 'gray'),
                text=data['mean'].round(0),
                textposition='outside',
                showlegend=False
            ),
            row=1, col=1
        )
    
    # Pie chart
    fig.add_trace(
        go.Pie(
            labels=comparison[periode_col],
            values=comparison['sum'],
            marker=dict(colors=[colors.get(p, 'gray') for p in comparison[periode_col]]),
            hole=0.4,
            textinfo='percent+label'
        ),
        row=1, col=2
    )
    
    fig.update_layout(height=400, showlegend=False)
    return fig

def export_to_excel(df, filename='export.xlsx'):
    """Exporte un DataFrame vers Excel"""
    try:
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Données')
        
        return output.getvalue()
    except Exception as e:
        st.error(f"Erreur lors de l'export: {e}")
        return None

# ==========================================
# HEADER DE L'APPLICATION
# ==========================================

st.markdown('<div class="main-header">🚴 Tableau de Bord - Mobilité Paris JO 2024</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Analyse approfondie de l\'impact des Jeux Olympiques sur le trafic cycliste parisien</div>', unsafe_allow_html=True)

# Barre de progression d'état
with st.spinner('🔄 Chargement des données...'):
    df = load_data('data/processed/comptage_velo_prepared.csv')
    
    # Si les données n'ont pas les features temporelles, on les crée
    if not df.empty and 'date_comptage' in df.columns:
        required_cols = ['annee', 'mois', 'heure', 'periode_jo']
        if not all(col in df.columns for col in required_cols):
            df = create_date_features(df)

# ==========================================
# SIDEBAR - NAVIGATION ET FILTRES AMÉLIORÉS
# ==========================================

with st.sidebar:
    # Logo
    st.image("https://upload.wikimedia.org/wikipedia/fr/thumb/8/8e/Paris_2024_Logo.svg/200px-Paris_2024_Logo.svg.png", width=150)
    
    st.title("⚙️ Navigation")
    
    # Navigation avec icônes et descriptions
    pages = {
        "🏠 Vue d'ensemble": "Indicateurs clés et métriques principales",
        "📊 Analyses Exploratoires": "Visualisations et patterns",
        "🤖 Machine Learning": "Modèles prédictifs et performances",
        "📈 Séries Temporelles": "Tendances et autocorrélation",
        "🎯 Impact des JO": "Analyse comparative avant/après",
        "🔮 Prédictions": "Simulations et scénarios",
        "📡 Données Temps Réel": "État actuel du réseau Vélib'",
        "📥 Export & Rapports": "Téléchargement et documentation"
    }
    
    page = st.radio(
        "Sélectionner une section:",
        list(pages.keys()),
        format_func=lambda x: x
    )
    
    # Affichage de la description
    st.caption(pages[page])
    
    st.markdown("---")
    
    # Filtres avancés
    st.subheader("🎛️ Filtres")
    
    if not df.empty and 'date_comptage' in df.columns:
        df['date_comptage'] = pd.to_datetime(df['date_comptage'])
        
        # Filtre de dates avec présets
        col1, col2 = st.columns(2)
        with col1:
            preset = st.selectbox(
                "Période prédéfinie",
                ["Personnalisé", "Dernier mois", "Derniers 3 mois", "Derniers 6 mois", "Année complète"]
            )
        
        date_min = df['date_comptage'].min().date()
        date_max = df['date_comptage'].max().date()
        
        if preset == "Dernier mois":
            start_date = date_max - timedelta(days=30)
        elif preset == "Derniers 3 mois":
            start_date = date_max - timedelta(days=90)
        elif preset == "Derniers 6 mois":
            start_date = date_max - timedelta(days=180)
        elif preset == "Année complète":
            start_date = date_min
        else:
            start_date = date_min
        
        date_range = st.date_input(
            "Période d'analyse:",
            value=(start_date, date_max),
            min_value=date_min,
            max_value=date_max
        )
        
        # Filtres multiples
        col1, col2 = st.columns(2)
        
        with col1:
            if 'periode_jo' in df.columns:
                periodes = st.multiselect(
                    "Période JO:",
                    options=sorted(df['periode_jo'].unique()),
                    default=sorted(df['periode_jo'].unique())
                )
        
        with col2:
            if 'saison' in df.columns:
                saisons = st.multiselect(
                    "Saisons:",
                    options=sorted(df['saison'].unique()),
                    default=sorted(df['saison'].unique())
                )
        
        # Filtre heures
        if 'heure' in df.columns:
            heure_range = st.slider(
                "Plage horaire:",
                min_value=0,
                max_value=23,
                value=(0, 23),
                format="%dh"
            )
        
        # Filtre jours de semaine
        if 'est_weekend' in df.columns:
            jour_type = st.radio(
                "Type de jour:",
                ["Tous", "Semaine", "Weekend"],
                horizontal=True
            )
        
        # Application des filtres
        df_filtered = df.copy()
        
        if len(date_range) == 2:
            df_filtered = df_filtered[
                (df_filtered['date_comptage'].dt.date >= date_range[0]) &
                (df_filtered['date_comptage'].dt.date <= date_range[1])
            ]
        
        if 'periodes' in locals() and periodes:
            df_filtered = df_filtered[df_filtered['periode_jo'].isin(periodes)]
        
        if 'saisons' in locals() and saisons:
            df_filtered = df_filtered[df_filtered['saison'].isin(saisons)]
        
        if 'heure_range' in locals():
            df_filtered = df_filtered[
                (df_filtered['heure'] >= heure_range[0]) &
                (df_filtered['heure'] <= heure_range[1])
            ]
        
        if 'jour_type' in locals():
            if jour_type == "Semaine":
                df_filtered = df_filtered[df_filtered['est_weekend'] == 0]
            elif jour_type == "Weekend":
                df_filtered = df_filtered[df_filtered['est_weekend'] == 1]
        
        # Affichage du nombre de données
        st.info(f"📊 **{len(df_filtered):,}** observations affichées sur **{len(df):,}** total")
    else:
        df_filtered = df
    
    st.markdown("---")
    
    # Informations système
    with st.expander("ℹ️ Informations"):
        st.markdown(f"""
        **Version:** 2.0  
        **Dernière MAJ:** {datetime.now().strftime('%d/%m/%Y %H:%M')}  
        **Données:** {len(df)} enregistrements  
        **Période:** {df['date_comptage'].min().date() if not df.empty else 'N/A'} → {df['date_comptage'].max().date() if not df.empty else 'N/A'}
        """)

# ==========================================
# PAGES AMÉLIORÉES
# ==========================================

if page == "🏠 Vue d'ensemble":
    st.header("📊 Vue d'ensemble du projet")
    
    if df_filtered.empty:
        st.warning("⚠️ Aucune donnée disponible avec les filtres sélectionnés")
    else:
        # Métriques clés améliorées
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            periode_jours = (df_filtered['date_comptage'].max() - df_filtered['date_comptage'].min()).days
            display_metric_card(
                "📅 Période analysée",
                f"{periode_jours} jours",
                icon="📅"
            )
        
        with col2:
            total_passages = df_filtered['comptage'].sum()
            display_metric_card(
                "🚴 Total passages",
                format_number(total_passages),
                icon="🚴"
            )
        
        with col3:
            compteurs = df_filtered['nom_compteur'].nunique() if 'nom_compteur' in df_filtered.columns else 'N/A'
            display_metric_card(
                "📍 Compteurs actifs",
                str(compteurs),
                icon="📍"
            )
        
        with col4:
            if 'periode_jo' in df_filtered.columns:
                avant = df_filtered[df_filtered['periode_jo'] == 'Avant JO']['comptage'].mean()
                apres = df_filtered[df_filtered['periode_jo'] == 'Après JO']['comptage'].mean()
                evolution = ((apres - avant) / avant * 100) if avant > 0 else 0
                display_metric_card(
                    "📈 Évolution post-JO",
                    f"{evolution:+.1f}%",
                    delta=evolution,
                    icon="📈"
                )
        
        st.markdown("---")
        
        # Graphique principal amélioré avec annotations
        st.subheader("📈 Évolution temporelle du trafic cycliste")
        
        df_daily = df_filtered.groupby('date_comptage')['comptage'].agg(['sum', 'mean']).reset_index()
        
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            subplot_titles=('Trafic journalier total', 'Moyenne mobile 7 jours'),
            vertical_spacing=0.1
        )
        
        # Graphique principal
        fig.add_trace(
            go.Scatter(
                x=df_daily['date_comptage'],
                y=df_daily['sum'],
                mode='lines',
                name='Trafic journalier',
                line=dict(color='#1E88E5', width=2),
                fill='tozeroy',
                fillcolor='rgba(30, 136, 229, 0.1)'
            ),
            row=1, col=1
        )
        
        # Moyenne mobile
        df_daily['ma7'] = df_daily['sum'].rolling(window=7, center=True).mean()
        fig.add_trace(
            go.Scatter(
                x=df_daily['date_comptage'],
                y=df_daily['ma7'],
                mode='lines',
                name='Moyenne mobile 7j',
                line=dict(color='#FFA726', width=3, dash='dash')
            ),
            row=1, col=1
        )
        
        # Période JO
        fig.add_vrect(
            x0='2024-07-26', x1='2024-08-11',
            fillcolor='gold', opacity=0.2,
            annotation_text='JO Paris 2024',
            annotation_position='top left',
            row=1, col=1
        )
        
        # Sous-graphique: taux de variation
        df_daily['variation'] = df_daily['sum'].pct_change() * 100
        fig.add_trace(
            go.Bar(
                x=df_daily['date_comptage'],
                y=df_daily['variation'],
                name='Variation %',
                marker=dict(
                    color=df_daily['variation'],
                    colorscale='RdYlGn',
                    cmin=-50,
                    cmax=50
                )
            ),
            row=2, col=1
        )
        
        fig.update_xaxes(title_text='Date', row=2, col=1)
        fig.update_yaxes(title_text='Passages', row=1, col=1)
        fig.update_yaxes(title_text='Variation %', row=2, col=1)
        
        fig.update_layout(
            height=700,
            hovermode='x unified',
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistiques détaillées
        st.subheader("📊 Statistiques descriptives")
        
        col1, col2 = st.columns(2)
        
        with col1:
            stats = calculate_statistics(df_filtered, 'comptage')
            stats_df = pd.DataFrame([stats]).T
            stats_df.columns = ['Valeur']
            stats_df['Valeur'] = stats_df['Valeur'].apply(lambda x: f"{x:,.2f}" if isinstance(x, float) else f"{x:,}")
            
            st.dataframe(
                stats_df.style.set_properties(**{'text-align': 'right'}),
                use_container_width=True
            )
        
        with col2:
            # Graphique de distribution
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=df_filtered['comptage'],
                nbinsx=50,
                name='Distribution',
                marker_color='#1E88E5',
                opacity=0.7
            ))
            
            fig.add_vline(
                x=df_filtered['comptage'].mean(),
                line_dash="dash",
                line_color="red",
                annotation_text=f"Moyenne: {df_filtered['comptage'].mean():.0f}"
            )
            
            fig.update_layout(
                title='Distribution du trafic',
                xaxis_title='Nombre de passages',
                yaxis_title='Fréquence',
                height=350
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Description du projet améliorée
        st.markdown("---")
        st.subheader("🎯 À propos du projet")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="info-box">
            <h4>🎯 Objectifs</h4>
            <ul>
                <li>✅ Analyser l'impact des JO 2024</li>
                <li>✅ Identifier les patterns de mobilité</li>
                <li>✅ Prédire les tendances futures</li>
                <li>✅ Mesurer l'héritage olympique</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="info-box">
            <h4>📊 Sources de données</h4>
            <ul>
                <li>🚴 Comptages vélos temps réel</li>
                <li>📍 Localisation des compteurs</li>
                <li>🅿️ Stations Vélib' (API)</li>
                <li>📈 Données historiques 2023-2024</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="info-box">
            <h4>🤖 Technologies</h4>
            <ul>
                <li>🐍 Python & Pandas</li>
                <li>📊 Plotly & Streamlit</li>
                <li>🧠 Scikit-learn (ML)</li>
                <li>🌐 API Paris Open Data</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

elif page == "📊 Analyses Exploratoires":
    st.header("📊 Analyses Exploratoires (EDA)")
    
    if df_filtered.empty:
        st.warning("⚠️ Aucune donnée disponible avec les filtres sélectionnés")
    else:
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "⏰ Analyse Temporelle",
            "📅 Patterns Hebdomadaires",
            "🌡️ Analyse Saisonnière",
            "🔥 Heatmaps",
            "📍 Analyse Géographique"
        ])
        
        with tab1:
            st.subheader("⏰ Profil horaire du trafic")
            
            if 'heure' in df_filtered.columns:
                hourly = df_filtered.groupby('heure')['comptage'].agg([
                    'mean', 'median', 'std', 'min', 'max', 'count'
                ]).reset_index()
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    fig = go.Figure()
                    
                    # Barres avec moyenne
                    fig.add_trace(go.Bar(
                        x=hourly['heure'],
                        y=hourly['mean'],
                        name='Moyenne',
                        marker_color='lightblue',
                        error_y=dict(
                            type='data',
                            array=hourly['std'],
                            visible=True,
                            color='rgba(0,0,0,0.3)'
                        ),
                        hovertemplate='<b>%{x}h</b><br>Moyenne: %{y:.0f}<extra></extra>'
                    ))
                    
                    # Ligne médiane
                    fig.add_trace(go.Scatter(
                        x=hourly['heure'],
                        y=hourly['median'],
                        name='Médiane',
                        line=dict(color='red', width=3, dash='dash'),
                        mode='lines+markers',
                        hovertemplate='<b>%{x}h</b><br>Médiane: %{y:.0f}<extra></extra>'
                    ))
                    
                    # Zones de pointe
                    fig.add_vrect(x0=7, x1=10, fillcolor="yellow", opacity=0.1, 
                                  annotation_text="Pointe matin", annotation_position="top left")
                    fig.add_vrect(x0=17, x1=20, fillcolor="orange", opacity=0.1,
                                  annotation_text="Pointe soir", annotation_position="top left")
                    
                    fig.update_layout(
                        title='Distribution horaire du trafic cycliste',
                        xaxis_title='Heure de la journée',
                        yaxis_title='Nombre de passages',
                        height=500,
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Statistiques horaires
                    peak_hour = hourly.loc[hourly['mean'].idxmax(), 'heure']
                    peak_value = hourly['mean'].max()
                    low_hour = hourly.loc[hourly['mean'].idxmin(), 'heure']
                    low_value = hourly['mean'].min()
                    
                    st.markdown("""
                    <div class="info-box success-box">
                    <h4>🔝 Heure de pointe</h4>
                    <h2>{:.0f}h</h2>
                    <p>{:.0f} passages en moyenne</p>
                    </div>
                    """.format(peak_hour, peak_value), unsafe_allow_html=True)
                    
                    st.markdown("""
                    <div class="info-box warning-box">
                    <h4>📉 Heure creuse</h4>
                    <h2>{:.0f}h</h2>
                    <p>{:.0f} passages en moyenne</p>
                    </div>
                    """.format(low_hour, low_value), unsafe_allow_html=True)
                    
                    # Ratio pointe/creuse
                    ratio = peak_value / low_value if low_value > 0 else 0
                    st.metric("📊 Ratio pointe/creuse", f"{ratio:.1f}x")
                
                # Tableau détaillé
                with st.expander("📋 Tableau détaillé par heure"):
                    st.dataframe(
                        hourly.style.format({
                            'mean': '{:.0f}',
                            'median': '{:.0f}',
                            'std': '{:.0f}',
                            'min': '{:.0f}',
                            'max': '{:.0f}'
                        }).background_gradient(subset=['mean'], cmap='Blues'),
                        use_container_width=True
                    )
        
        with tab2:
            st.subheader("📅 Analyse par jour de la semaine")
            
            if 'nom_jour' in df_filtered.columns:
                jour_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                jour_fr = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']
                
                weekly = df_filtered.groupby('nom_jour')['comptage'].agg([
                    'mean', 'median', 'sum', 'count', 'std'
                ]).reset_index()
                
                weekly['nom_jour'] = pd.Categorical(weekly['nom_jour'], categories=jour_order, ordered=True)
                weekly = weekly.sort_values('nom_jour')
                weekly['jour_fr'] = jour_fr
                weekly['cv'] = (weekly['std'] / weekly['mean'] * 100)  # Coefficient de variation
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = go.Figure()
                    
                    fig.add_trace(go.Bar(
                        x=weekly['jour_fr'],
                        y=weekly['mean'],
                        marker=dict(
                            color=weekly['mean'],
                            colorscale='Blues',
                            showscale=True,
                            colorbar=dict(title="Passages")
                        ),
                        text=weekly['mean'].round(0),
                        textposition='outside',
                        hovertemplate='<b>%{x}</b><br>Moyenne: %{y:.0f}<extra></extra>'
                    ))
                    
                    fig.update_layout(
                        title='Trafic moyen par jour de la semaine',
                        xaxis_title='Jour',
                        yaxis_title='Passages moyens',
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.pie(
                        weekly,
                        values='sum',
                        names='jour_fr',
                        title='Répartition du trafic hebdomadaire',
                        hole=0.4,
                        color_discrete_sequence=px.colors.sequential.Blues_r
                    )
                    
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    fig.update_layout(height=400)
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Comparaison semaine vs weekend
                st.subheader("📊 Comparaison Semaine vs Weekend")
                
                if 'est_weekend' in df_filtered.columns:
                    comparison = df_filtered.groupby('est_weekend')['comptage'].agg([
                        'mean', 'median', 'sum'
                    ]).reset_index()
                    comparison['type'] = comparison['est_weekend'].map({0: 'Semaine', 1: 'Weekend'})
                    
                    col1, col2, col3 = st.columns(3)
                    
                    semaine_mean = comparison[comparison['est_weekend'] == 0]['mean'].values[0]
                    weekend_mean = comparison[comparison['est_weekend'] == 1]['mean'].values[0]
                    diff_pct = ((weekend_mean - semaine_mean) / semaine_mean * 100)
                    
                    with col1:
                        st.metric("🏢 Moyenne Semaine", f"{semaine_mean:.0f}")
                    with col2:
                        st.metric("🏖️ Moyenne Weekend", f"{weekend_mean:.0f}")
                    with col3:
                        st.metric("📊 Différence", f"{diff_pct:+.1f}%", delta=f"{diff_pct:+.1f}%")
        
        with tab3:
            st.subheader("🌡️ Analyse saisonnière")
            
            if 'saison' in df_filtered.columns:
                seasonal = df_filtered.groupby('saison')['comptage'].agg([
                    'mean', 'median', 'sum', 'count', 'std'
                ]).reset_index()
                
                # Ordre des saisons
                saison_order = ['Hiver', 'Printemps', 'Été', 'Automne']
                seasonal['saison'] = pd.Categorical(seasonal['saison'], categories=saison_order, ordered=True)
                seasonal = seasonal.sort_values('saison')
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    fig = go.Figure()
                    
                    colors_saison = {
                        'Hiver': '#1E88E5',
                        'Printemps': '#43A047',
                        'Été': '#FDD835',
                        'Automne': '#E53935'
                    }
                    
                    for idx, row in seasonal.iterrows():
                        fig.add_trace(go.Bar(
                            x=[row['saison']],
                            y=[row['mean']],
                            name=row['saison'],
                            marker_color=colors_saison.get(row['saison'], 'gray'),
                            text=f"{row['mean']:.0f}",
                            textposition='outside',
                            error_y=dict(type='data', array=[row['std']], visible=True),
                            hovertemplate=f"<b>{row['saison']}</b><br>" +
                                        f"Moyenne: {row['mean']:.0f}<br>" +
                                        f"Médiane: {row['median']:.0f}<br>" +
                                        f"Total: {row['sum']:.0f}<extra></extra>"
                        ))
                    
                    fig.update_layout(
                        title='Trafic cycliste moyen par saison',
                        yaxis_title='Passages moyens',
                        showlegend=False,
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("### 📊 Statistiques saisonnières")
                    
                    for _, row in seasonal.iterrows():
                        color = colors_saison.get(row['saison'], 'gray')
                        st.markdown(f"""
                        <div style='background: {color}; color: white; padding: 1rem; 
                                    border-radius: 10px; margin: 0.5rem 0;'>
                            <h4 style='margin: 0;'>{row['saison']}</h4>
                            <p style='margin: 0.5rem 0;'>
                                <strong>Moyenne:</strong> {row['mean']:.0f}<br>
                                <strong>Total:</strong> {format_number(row['sum'])}<br>
                                <strong>Jours:</strong> {row['count']:.0f}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Évolution saisonnière sur l'année
                if 'mois' in df_filtered.columns:
                    st.subheader("📈 Évolution mensuelle")
                    
                    monthly = df_filtered.groupby('mois')['comptage'].mean().reset_index()
                    monthly['saison'] = monthly['mois'].apply(
                        lambda x: 'Hiver' if x in [12, 1, 2] else
                                 'Printemps' if x in [3, 4, 5] else
                                 'Été' if x in [6, 7, 8] else 'Automne'
                    )
                    
                    fig = px.line(
                        monthly,
                        x='mois',
                        y='comptage',
                        color='saison',
                        markers=True,
                        title='Évolution du trafic moyen par mois',
                        labels={'mois': 'Mois', 'comptage': 'Passages moyens'},
                        color_discrete_map=colors_saison
                    )
                    
                    fig.update_xaxes(dtick=1)
                    fig.update_layout(height=400)
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.subheader("🔥 Heatmaps et matrices de corrélation")
            
            if all(col in df_filtered.columns for col in ['jour_semaine', 'heure', 'comptage']):
                col1, col2 = st.columns(2)
                
                with col1:
                    # Heatmap Jour × Heure
                    pivot = df_filtered.pivot_table(
                        values='comptage',
                        index='jour_semaine',
                        columns='heure',
                        aggfunc='mean'
                    )
                    
                    pivot.index = ['Lun', 'Mar', 'Mer', 'Jeu', 'Ven', 'Sam', 'Dim']
                    
                    fig = px.imshow(
                        pivot,
                        labels=dict(x="Heure", y="Jour", color="Passages"),
                        title="Intensité du trafic (Jour × Heure)",
                        color_continuous_scale='YlOrRd',
                        aspect='auto'
                    )
                    
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Heatmap Mois × Heure
                    if 'mois' in df_filtered.columns:
                        pivot_mois = df_filtered.pivot_table(
                            values='comptage',
                            index='mois',
                            columns='heure',
                            aggfunc='mean'
                        )
                        
                        mois_labels = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Juin',
                                      'Jul', 'Aoû', 'Sep', 'Oct', 'Nov', 'Déc']
                        pivot_mois.index = [mois_labels[int(i)-1] for i in pivot_mois.index]
                        
                        fig = px.imshow(
                            pivot_mois,
                            labels=dict(x="Heure", y="Mois", color="Passages"),
                            title="Intensité du trafic (Mois × Heure)",
                            color_continuous_scale='Viridis',
                            aspect='auto'
                        )
                        
                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True)
                
                # Matrice de corrélation
                st.subheader("📊 Matrice de corrélation")
                
                numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns
                correlation_cols = [col for col in ['heure', 'jour_semaine', 'mois', 'comptage', 
                                                    'est_weekend', 'semaine'] if col in numeric_cols]
                
                if len(correlation_cols) > 1:
                    corr_matrix = df_filtered[correlation_cols].corr()
                    
                    fig = px.imshow(
                        corr_matrix,
                        labels=dict(color="Corrélation"),
                        title="Matrice de corrélation des variables",
                        color_continuous_scale='RdBu_r',
                        zmin=-1,
                        zmax=1,
                        text_auto='.2f'
                    )
                    
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab5:
            st.subheader("📍 Analyse géographique (par compteur)")
            
            if 'nom_compteur' in df_filtered.columns:
                # Top compteurs
                top_compteurs = df_filtered.groupby('nom_compteur')['comptage'].agg([
                    'sum', 'mean', 'count'
                ]).sort_values('sum', ascending=False).head(20).reset_index()
                
                col1, col2 = st.columns([3, 2])
                
                with col1:
                    fig = px.bar(
                        top_compteurs.head(15),
                        x='sum',
                        y='nom_compteur',
                        orientation='h',
                        title='Top 15 des compteurs les plus actifs',
                        labels={'sum': 'Total passages', 'nom_compteur': 'Compteur'},
                        color='sum',
                        color_continuous_scale='Blues'
                    )
                    
                    fig.update_layout(
                        height=600,
                        yaxis={'categoryorder': 'total ascending'}
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("### 🏆 Top 5 des compteurs")
                    
                    for idx, row in top_compteurs.head(5).iterrows():
                        medal = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][idx]
                        st.markdown(f"""
                        <div class="info-box">
                            <h4>{medal} {row['nom_compteur']}</h4>
                            <p>
                                <strong>Total:</strong> {format_number(row['sum'])}<br>
                                <strong>Moyenne:</strong> {row['mean']:.0f}<br>
                                <strong>Mesures:</strong> {row['count']:.0f}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Distribution par compteur
                st.subheader("📊 Distribution du trafic par compteur")
                
                compteur_stats = df_filtered.groupby('nom_compteur')['comptage'].agg([
                    'mean', 'std', 'min', 'max'
                ]).reset_index()
                
                fig = go.Figure()
                
                fig.add_trace(go.Box(
                    y=df_filtered['comptage'],
                    x=df_filtered['nom_compteur'],
                    name='Distribution',
                    marker_color='lightblue'
                ))
                
                fig.update_layout(
                    title='Boxplot du trafic par compteur',
                    xaxis_title='Compteur',
                    yaxis_title='Passages',
                    height=500,
                    showlegend=False
                )
                
                fig.update_xaxes(tickangle=45)
                
                st.plotly_chart(fig, use_container_width=True)

elif page == "🤖 Machine Learning":
    st.header("🤖 Modèles de Machine Learning")
    
    # Chargement des métadonnées
    metadata_path = 'models/metadata.json'
    
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.success(f"✅ Modèles entraînés le: **{metadata.get('date_entrainement', 'N/A')}**")
        with col2:
            st.info(f"🏆 Meilleur modèle: **{metadata.get('meilleur_modele', 'N/A')}**")
        with col3:
            if 'performances' in metadata and metadata['meilleur_modele'] in metadata['performances']:
                best_r2 = metadata['performances'][metadata['meilleur_modele']].get('R²', 0)
                st.metric("📊 R² Score", f"{best_r2:.4f}")
        
        st.markdown("---")
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Performances",
            "🎯 Features Importance",
            "🔮 Prédictions",
            "🧪 Tests & Validation"
        ])
        
        with tab1:
            st.subheader("📊 Comparaison des performances des modèles")
            
            if 'performances' in metadata:
                results_df = pd.DataFrame(metadata['performances']).T
                results_df = results_df.sort_values('R²', ascending=False)
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Graphique comparatif
                    fig = go.Figure()
                    
                    x_pos = np.arange(len(results_df))
                    
                    fig.add_trace(go.Bar(
                        name='RMSE',
                        x=results_df.index,
                        y=results_df['RMSE'],
                        text=results_df['RMSE'].round(2),
                        textposition='auto',
                        marker_color='#FF6B6B'
                    ))
                    
                    fig.add_trace(go.Bar(
                        name='MAE',
                        x=results_df.index,
                        y=results_df['MAE'],
                        text=results_df['MAE'].round(2),
                        textposition='auto',
                        marker_color='#4ECDC4'
                    ))
                    
                    fig.add_trace(go.Scatter(
                        name='R² (×100)',
                        x=results_df.index,
                        y=results_df['R²'] * 100,
                        mode='lines+markers',
                        line=dict(color='#95E1D3', width=3),
                        marker=dict(size=12),
                        yaxis='y2'
                    ))
                    
                    fig.update_layout(
                        title='Métriques de performance par modèle',
                        barmode='group',
                        height=500,
                        yaxis=dict(title='RMSE / MAE'),
                        yaxis2=dict(title='R² Score (×100)', overlaying='y', side='right'),
                        hovermode='x unified',
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("### 🏆 Classement des modèles")
                    
                    for idx, (model, scores) in enumerate(results_df.iterrows(), 1):
                        medal = "🥇" if idx == 1 else "🥈" if idx == 2 else "🥉" if idx == 3 else f"{idx}."
                        
                        color = "#FFD700" if idx == 1 else "#C0C0C0" if idx == 2 else "#CD7F32" if idx == 3 else "#E8E8E8"
                        
                        st.markdown(f"""
                        <div style='background: {color}; padding: 1rem; border-radius: 10px; 
                                    margin: 0.5rem 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                            <h4 style='margin: 0;'>{medal} {model}</h4>
                            <p style='margin: 0.5rem 0; font-size: 0.9rem;'>
                                <strong>R²:</strong> {scores['R²']:.4f}<br>
                                <strong>RMSE:</strong> {scores['RMSE']:.2f}<br>
                                <strong>MAE:</strong> {scores['MAE']:.2f}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Tableau détaillé
                st.subheader("📋 Tableau détaillé des performances")
                
                results_styled = results_df.style.format({
                    'RMSE': '{:.2f}',
                    'MAE': '{:.2f}',
                    'R²': '{:.4f}'
                }).background_gradient(subset=['R²'], cmap='Greens').background_gradient(
                    subset=['RMSE', 'MAE'], cmap='Reds_r'
                )
                
                st.dataframe(results_styled, use_container_width=True)
        
        with tab2:
            st.subheader("📈 Importance des variables (Features)")
            
            rf_model = load_model('random_forest_model.pkl')
            
            if rf_model and hasattr(rf_model, 'feature_importances_') and 'features' in metadata:
                importance_df = pd.DataFrame({
                    'Feature': metadata['features'],
                    'Importance': rf_model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    fig = px.bar(
                        importance_df,
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title='Importance des features (Random Forest)',
                        color='Importance',
                        color_continuous_scale='Viridis',
                        text='Importance'
                    )
                    
                    fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
                    fig.update_layout(
                        height=400,
                        yaxis={'categoryorder': 'total ascending'}
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("### 🎯 Top Features")
                    
                    for idx, row in importance_df.head(5).iterrows():
                        importance_pct = row['Importance'] * 100
                        st.markdown(f"""
                        <div class="info-box">
                            <h4>{row['Feature']}</h4>
                            <div style='background: linear-gradient(90deg, #667eea 0%, 
                                        #667eea {importance_pct}%, #f0f0f0 {importance_pct}%); 
                                        height: 25px; border-radius: 10px; 
                                        display: flex; align-items: center; padding: 0 10px;'>
                                <span style='font-weight: bold;'>{importance_pct:.1f}%</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Analyse de l'importance
                st.subheader("💡 Insights sur les features")
                
                top_feature = importance_df.iloc[0]
                
                st.info(f"""
                🔝 **Feature la plus importante**: {top_feature['Feature']} 
                ({top_feature['Importance']*100:.1f}% d'importance)
                
                Cette variable a le plus grand impact sur les prédictions du modèle.
                """)
            else:
                st.warning("⚠️ Modèle Random Forest non disponible ou sans feature importances")
        
        with tab3:
            st.subheader("🎯 Prédictions interactives")
            
            rf_model = load_model('random_forest_model.pkl')
            
            if rf_model:
                st.info("💡 Remplissez les paramètres ci-dessous pour obtenir une prédiction")
                
                with st.form("prediction_form"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        annee = st.number_input("Année", min_value=2023, max_value=2026, value=2024)
                        mois = st.slider("Mois", 1, 12, datetime.now().month)
                        jour = st.slider("Jour", 1, 31, datetime.now().day)
                    
                    with col2:
                        heure = st.slider("Heure", 0, 23, 18)
                        jour_semaine = st.selectbox(
                            "Jour de la semaine",
                            options=list(range(7)),
                            format_func=lambda x: ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 
                                                   'Vendredi', 'Samedi', 'Dimanche'][x]
                        )
                    
                    with col3:
                        est_weekend = st.checkbox("Weekend", value=(jour_semaine in [5, 6]))
                        pendant_jo = st.checkbox("Pendant les JO", value=False)
                    
                    col1, col2, col3 = st.columns([1, 1, 1])
                    
                    with col2:
                        submitted = st.form_submit_button("🔮 Prédire", type="primary", use_container_width=True)
                    
                    if submitted:
                        try:
                            # Préparation des features
                            semaine = pd.Timestamp(f'{annee}-{mois}-{jour}').isocalendar().week
                            
                            features = np.array([[
                                annee, mois, jour, heure, jour_semaine,
                                int(est_weekend), int(pendant_jo), semaine
                            ]])
                            
                            # Prédiction
                            prediction = rf_model.predict(features)[0]
                            
                            # Intervalle de confiance
                            if hasattr(rf_model, 'estimators_'):
                                predictions_trees = np.array([
                                    tree.predict(features)[0] for tree in rf_model.estimators_
                                ])
                                ci_lower = np.percentile(predictions_trees, 2.5)
                                ci_upper = np.percentile(predictions_trees, 97.5)
                                std_pred = predictions_trees.std()
                            else:
                                ci_lower = prediction * 0.85
                                ci_upper = prediction * 1.15
                                std_pred = prediction * 0.1
                            
                            st.markdown("---")
                            st.success("✅ Prédiction effectuée avec succès!")
                            
                            # Affichage des résultats
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                display_metric_card(
                                    "🎯 Prédiction",
                                    f"{prediction:.0f}",
                                    icon="🎯"
                                )
                            
                            with col2:
                                display_metric_card(
                                    "📉 Borne inférieure",
                                    f"{ci_lower:.0f}",
                                    icon="📉"
                                )
                            
                            with col3:
                                display_metric_card(
                                    "📈 Borne supérieure",
                                    f"{ci_upper:.0f}",
                                    icon="📈"
                                )
                            
                            with col4:
                                display_metric_card(
                                    "📊 Incertitude",
                                    f"±{std_pred:.0f}",
                                    icon="📊"
                                )
                            
                            # Visualisation
                            fig = go.Figure()
                            
                            fig.add_trace(go.Bar(
                                x=['Prédiction'],
                                y=[prediction],
                                error_y=dict(
                                    type='data',
                                    symmetric=False,
                                    array=[ci_upper - prediction],
                                    arrayminus=[prediction - ci_lower],
                                    color='rgba(255, 0, 0, 0.5)',
                                    thickness=2,
                                    width=20
                                ),
                                marker=dict(
                                    color='#1E88E5',
                                    line=dict(color='#0D47A1', width=2)
                                ),
                                text=f"{prediction:.0f}",
                                textposition='outside',
                                textfont=dict(size=20, color='#1E88E5')
                            ))
                            
                            fig.update_layout(
                                title=f"Prédiction pour le {jour:02d}/{mois:02d}/{annee} à {heure:02d}h",
                                yaxis_title='Nombre de passages',
                                height=400,
                                showlegend=False
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Interprétation
                            if prediction > 200:
                                st.success("📊 **Trafic élevé prévu** - Prévoyez un flux important de cyclistes")
                            elif prediction > 100:
                                st.info("📊 **Trafic modéré prévu** - Conditions normales de circulation")
                            else:
                                st.warning("📊 **Trafic faible prévu** - Peu de passages attendus")
                        
                        except Exception as e:
                            st.error(f"❌ Erreur lors de la prédiction: {e}")
            else:
                st.warning("⚠️ Modèle non disponible. Veuillez d'abord entraîner les modèles.")
        
        with tab4:
            st.subheader("🧪 Tests et validation du modèle")
            
            st.info("""
            💡 Cette section présente les méthodes de validation utilisées pour évaluer 
            la performance et la fiabilité des modèles de prédiction.
            """)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div class="info-box">
                <h4>✅ Méthodes de validation</h4>
                <ul>
                    <li><strong>Train/Test Split:</strong> 80/20</li>
                    <li><strong>Cross-validation:</strong> K-Fold (k=5)</li>
                    <li><strong>Métriques:</strong> RMSE, MAE, R²</li>
                    <li><strong>Validation temporelle:</strong> Oui</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="info-box">
                <h4>🎯 Critères de qualité</h4>
                <ul>
                    <li><strong>R² > 0.80:</strong> Excellent</li>
                    <li><strong>R² > 0.70:</strong> Bon</li>
                    <li><strong>R² > 0.60:</strong> Acceptable</li>
                    <li><strong>RMSE:</strong> Plus faible = meilleur</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
            
            # Résidus (si disponibles)
            st.subheader("📊 Analyse des résidus")
            st.info("Les résidus représentent la différence entre les valeurs prédites et les valeurs réelles.")
            
            # Simulation de résidus pour démonstration
            if rf_model and not df_filtered.empty:
                st.caption("⚠️ Données simulées pour démonstration")
                
                n_samples = 1000
                residuals = np.random.normal(0, 30, n_samples)
                
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=('Distribution des résidus', 'Q-Q Plot')
                )
                
                # Histogramme
                fig.add_trace(
                    go.Histogram(
                        x=residuals,
                        nbinsx=50,
                        name='Résidus',
                        marker_color='lightblue'
                    ),
                    row=1, col=1
                )
                
                # Q-Q Plot
                from scipy import stats
                theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(residuals)))
                sample_quantiles = np.sort(residuals)
                
                fig.add_trace(
                    go.Scatter(
                        x=theoretical_quantiles,
                        y=sample_quantiles,
                        mode='markers',
                        name='Q-Q',
                        marker=dict(color='coral')
                    ),
                    row=1, col=2
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=theoretical_quantiles,
                        y=theoretical_quantiles,
                        mode='lines',
                        name='Référence',
                        line=dict(color='red', dash='dash')
                    ),
                    row=1, col=2
                )
                
                fig.update_layout(height=400, showlegend=True)
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ Métadonnées des modèles non trouvées. Veuillez d'abord entraîner les modèles.")
        
        st.markdown("""
        <div class="info-box warning-box">
        <h4>📝 Pour entraîner les modèles:</h4>
        <ol>
            <li>Assurez-vous d'avoir les données dans <code>data/processed/</code></li>
            <li>Exécutez le script d'entraînement: <code>python train_models.py</code></li>
            <li>Les modèles seront sauvegardés dans <code>models/</code></li>
        </ol>
        </div>
        """, unsafe_allow_html=True)

elif page == "📈 Séries Temporelles":
    st.header("📈 Analyse de Séries Temporelles")
    
    if df_filtered.empty or 'date_comptage' not in df_filtered.columns:
        st.warning("⚠️ Données insuffisantes pour l'analyse de séries temporelles")
    else:
        # Agrégation journalière
        df_ts = df_filtered.groupby('date_comptage')['comptage'].sum().reset_index()
        df_ts = df_ts.sort_values('date_comptage')
        
        tab1, tab2, tab3 = st.tabs([
            "📉 Tendances",
            "🔄 Autocorrélation",
            "🔮 Décomposition"
        ])
        
        with tab1:
            st.subheader("📉 Tendances et moyennes mobiles")
            
            # Calcul des moyennes mobiles
            window_sizes = st.multiselect(
                "Sélectionner les fenêtres de moyennes mobiles:",
                options=[3, 7, 14, 30, 60],
                default=[7, 30]
            )
            
            fig = go.Figure()
            
            # Données originales
            fig.add_trace(go.Scatter(
                x=df_ts['date_comptage'],
                y=df_ts['comptage'],
                mode='lines',
                name='Données journalières',
                line=dict(color='lightgray', width=1),
                opacity=0.5
            ))
            
            colors = ['blue', 'red', 'green', 'orange', 'purple']
            
            for idx, window in enumerate(window_sizes):
                ma = df_ts['comptage'].rolling(window=window, center=True).mean()
                fig.add_trace(go.Scatter(
                    x=df_ts['date_comptage'],
                    y=ma,
                    mode='lines',
                    name=f'MA {window} jours',
                    line=dict(color=colors[idx % len(colors)], width=2+idx)
                ))
            
            # Période JO
            fig.add_vrect(
                x0='2024-07-26', x1='2024-08-11',
                fillcolor='gold', opacity=0.2,
                annotation_text='JO 2024',
                annotation_position='top left'
            )
            
            fig.update_layout(
                title='Évolution du trafic avec tendances',
                xaxis_title='Date',
                yaxis_title='Nombre de passages',
                height=600,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistiques
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📊 Moyenne", f"{df_ts['comptage'].mean():.0f}")
            with col2:
                st.metric("📈 Maximum", f"{df_ts['comptage'].max():.0f}")
            with col3:
                st.metric("📉 Minimum", f"{df_ts['comptage'].min():.0f}")
            with col4:
                trend = (df_ts['comptage'].iloc[-30:].mean() - df_ts['comptage'].iloc[:30].mean())
                trend_pct = (trend / df_ts['comptage'].iloc[:30].mean() * 100) if df_ts['comptage'].iloc[:30].mean() > 0 else 0
                st.metric("📊 Tendance", f"{trend_pct:+.1f}%", delta=f"{trend_pct:+.1f}%")
        
        with tab2:
            st.subheader("🔄 Analyse d'autocorrélation")
            
            with st.expander("ℹ️ Comprendre l'autocorrélation"):
                st.markdown("""
                L'**autocorrélation** mesure la corrélation entre une série temporelle et elle-même 
                décalée de plusieurs périodes (lags).
                
                - **Valeurs positives**: comportement similaire avec délai
                - **Valeurs négatives**: comportement opposé avec délai
                - **Valeurs proches de 0**: pas de corrélation
                
                Les bandes rouges représentent les **intervalles de confiance** (95%).
                """)
            
            lags = st.slider("Nombre de décalages (lags)", 1, 90, 30)
            
            acf_values = [df_ts['comptage'].autocorr(lag=i) for i in range(1, lags+1)]
            
            fig = go.Figure()
            
            # Barres d'autocorrélation
            colors = ['green' if abs(v) > 1.96/np.sqrt(len(df_ts)) else 'lightblue' for v in acf_values]
            
            fig.add_trace(go.Bar(
                x=list(range(1, lags+1)),
                y=acf_values,
                marker_color=colors,
                hovertemplate='<b>Lag %{x}</b><br>ACF: %{y:.3f}<extra></extra>'
            ))
            
            # Lignes de confiance
            confidence_interval = 1.96/np.sqrt(len(df_ts))
            
            fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1)
            fig.add_hline(y=confidence_interval, line_dash="dash", line_color="red", 
                         annotation_text="95% CI", annotation_position="right")
            fig.add_hline(y=-confidence_interval, line_dash="dash", line_color="red")
            
            fig.update_layout(
                title='Fonction d\'autocorrélation (ACF)',
                xaxis_title='Lag (jours)',
                yaxis_title='Autocorrélation',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Interprétation
            significant_lags = [i+1 for i, v in enumerate(acf_values) if abs(v) > confidence_interval]
            
            if significant_lags:
                st.info(f"""
                🔍 **Lags significatifs détectés**: {', '.join(map(str, significant_lags[:5]))}
                
                Ces décalages montrent des **patterns cycliques** dans les données.
                Par exemple, un lag de 7 jours suggère un cycle hebdomadaire.
                """)
        
        with tab3:
            st.subheader("🔮 Décomposition de la série temporelle")
            
            st.info("""
            💡 La décomposition sépare la série en trois composantes:
            - **Tendance**: évolution à long terme
            - **Saisonnalité**: patterns cycliques réguliers
            - **Résidus**: variations aléatoires
            """)
            
            # Décomposition simplifiée
            from scipy.signal import savgol_filter
            
            # Tendance (filtre Savitzky-Golay)
            window = min(51, len(df_ts) // 2)
            if window % 2 == 0:
                window += 1
            
            if len(df_ts) > window:
                trend = savgol_filter(df_ts['comptage'].values, window, 3)
                seasonal = df_ts['comptage'].values - trend
                residual = df_ts['comptage'].values - trend - seasonal
                
                fig = make_subplots(
                    rows=4, cols=1,
                    subplot_titles=('Série originale', 'Tendance', 'Saisonnalité', 'Résidus'),
                    vertical_spacing=0.05
                )
                
                # Série originale
                fig.add_trace(
                    go.Scatter(x=df_ts['date_comptage'], y=df_ts['comptage'],
                              mode='lines', name='Original', line=dict(color='black')),
                    row=1, col=1
                )
                
                # Tendance
                fig.add_trace(
                    go.Scatter(x=df_ts['date_comptage'], y=trend,
                              mode='lines', name='Tendance', line=dict(color='blue', width=2)),
                    row=2, col=1
                )
                
                # Saisonnalité
                fig.add_trace(
                    go.Scatter(x=df_ts['date_comptage'], y=seasonal,
                              mode='lines', name='Saisonnalité', line=dict(color='green')),
                    row=3, col=1
                )
                
                # Résidus
                fig.add_trace(
                    go.Scatter(x=df_ts['date_comptage'], y=residual,
                              mode='lines', name='Résidus', line=dict(color='red')),
                    row=4, col=1
                )
                
                fig.update_layout(height=1000, showlegend=False)
                fig.update_xaxes(title_text='Date', row=4, col=1)
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ Données insuffisantes pour la décomposition")

elif page == "🎯 Impact des JO":
    st.header("🎯 Analyse d'impact des JO Paris 2024")
    
    if df_filtered.empty or 'periode_jo' not in df_filtered.columns:
        st.warning("⚠️ Données insuffisantes pour l'analyse d'impact des JO")
    else:
        # KPIs principaux
        st.subheader("📊 Indicateurs clés de comparaison")
        
        avant = df_filtered[df_filtered['periode_jo'] == 'Avant JO']['comptage']
        pendant = df_filtered[df_filtered['periode_jo'] == 'Pendant JO']['comptage']
        apres = df_filtered[df_filtered['periode_jo'] == 'Après JO']['comptage']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 2rem; border-radius: 15px; color: white; text-align: center;
                        box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                <h3 style='margin: 0; font-size: 1rem;'>Avant JO</h3>
                <h1 style='margin: 0.5rem 0; font-size: 2.5rem;'>{avant.mean():.0f}</h1>
                <p style='margin: 0; opacity: 0.9;'>passages moyens/heure</p>
                <p style='margin: 0.5rem 0; font-size: 0.9rem;'>📅 {len(avant)} mesures</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                        padding: 2rem; border-radius: 15px; color: white; text-align: center;
                        box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                <h3 style='margin: 0; font-size: 1rem;'>Pendant JO</h3>
                <h1 style='margin: 0.5rem 0; font-size: 2.5rem;'>{pendant.mean() if len(pendant) > 0 else 0:.0f}</h1>
                <p style='margin: 0; opacity: 0.9;'>passages moyens/heure</p>
                <p style='margin: 0.5rem 0; font-size: 0.9rem;'>📅 {len(pendant)} mesures</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            apres_mean = apres.mean() if len(apres) > 0 else 0
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                        padding: 2rem; border-radius: 15px; color: white; text-align: center;
                        box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                <h3 style='margin: 0; font-size: 1rem;'>Après JO</h3>
                <h1 style='margin: 0.5rem 0; font-size: 2.5rem;'>{apres_mean:.0f}</h1>
                <p style='margin: 0; opacity: 0.9;'>passages moyens/heure</p>
                <p style='margin: 0.5rem 0; font-size: 0.9rem;'>📅 {len(apres)} mesures</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            evolution = ((apres_mean - avant.mean()) / avant.mean() * 100) if avant.mean() > 0 and len(apres) > 0 else 0
            color = "#43A047" if evolution > 0 else "#E53935"
            arrow = "↗" if evolution > 0 else "↘"
            
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                        padding: 2rem; border-radius: 15px; color: white; text-align: center;
                        box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                <h3 style='margin: 0; font-size: 1rem;'>Évolution</h3>
                <h1 style='margin: 0.5rem 0; font-size: 2.5rem; color: {color};'>{arrow} {abs(evolution):.1f}%</h1>
                <p style='margin: 0; opacity: 0.9;'>Avant → Après JO</p>
                <p style='margin: 0.5rem 0; font-size: 0.9rem;'>
                    {'📈 Augmentation' if evolution > 0 else '📉 Diminution'}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Onglets d'analyse
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Évolution Temporelle",
            "📊 Distributions",
            "⏰ Profils Horaires",
            "💡 Insights & Recommandations"
        ])
        
        with tab1:
            st.subheader("📈 Évolution temporelle du trafic")
            
            df_impact = df_filtered.groupby(['date_comptage', 'periode_jo'])['comptage'].sum().reset_index()
            
            fig = px.line(
                df_impact,
                x='date_comptage',
                y='comptage',
                color='periode_jo',
                title='Impact des JO sur le trafic cycliste',
                labels={'date_comptage': 'Date', 'comptage': 'Passages', 'periode_jo': 'Période'},
                color_discrete_map={
                    'Avant JO': '#667eea',
                    'Pendant JO': '#f5576c',
                    'Après JO': '#00f2fe'
                }
            )
            
            fig.add_vrect(
                x0='2024-07-26', x1='2024-08-11',
                fillcolor='gold', opacity=0.15,
                annotation_text='Période JO 2024',
                annotation_position='top left'
            )
            
            fig.update_layout(height=500, hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("📦 Distributions comparatives")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Boxplots
                fig = go.Figure()
                
                for periode, color in [('Avant JO', '#667eea'), ('Pendant JO', '#f5576c'), ('Après JO', '#00f2fe')]:
                    data = df_filtered[df_filtered['periode_jo'] == periode]['comptage']
                    if len(data) > 0:
                        fig.add_trace(go.Box(
                            y=data,
                            name=periode,
                            marker_color=color,
                            boxmean='sd'
                        ))
                
                fig.update_layout(
                    title='Distribution du trafic par période',
                    yaxis_title='Nombre de passages',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Violin plots
                fig = go.Figure()
                
                for periode, color in [('Avant JO', '#667eea'), ('Pendant JO', '#f5576c'), ('Après JO', '#00f2fe')]:
                    data = df_filtered[df_filtered['periode_jo'] == periode]['comptage']
                    if len(data) > 0:
                        fig.add_trace(go.Violin(
                            y=data,
                            name=periode,
                            fillcolor=color,
                            opacity=0.6,
                            box_visible=True,
                            meanline_visible=True
                        ))
                
                fig.update_layout(
                    title='Distribution détaillée (Violin plot)',
                    yaxis_title='Nombre de passages',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Statistiques détaillées
            st.subheader("📊 Statistiques détaillées par période")
            
            stats_df = df_filtered.groupby('periode_jo')['comptage'].agg([
                ('Moyenne', 'mean'),
                ('Médiane', 'median'),
                ('Écart-type', 'std'),
                ('Min', 'min'),
                ('Max', 'max'),
                ('Total', 'sum'),
                ('Q1', lambda x: x.quantile(0.25)),
                ('Q3', lambda x: x.quantile(0.75)),
                ('Mesures', 'count')
            ]).round(2)
            
            st.dataframe(
                stats_df.style.background_gradient(cmap='Blues', subset=['Moyenne', 'Médiane'])
                .format(precision=2),
                use_container_width=True
            )
        
        with tab3:
            st.subheader("⏰ Profils horaires par période")
            
            if 'heure' in df_filtered.columns:
                hourly_comparison = df_filtered.groupby(['periode_jo', 'heure'])['comptage'].mean().reset_index()
                
                fig = px.line(
                    hourly_comparison,
                    x='heure',
                    y='comptage',
                    color='periode_jo',
                    title='Profil horaire moyen selon la période',
                    labels={'heure': 'Heure', 'comptage': 'Passages moyens', 'periode_jo': 'Période'},
                    color_discrete_map={
                        'Avant JO': '#667eea',
                        'Pendant JO': '#f5576c',
                        'Après JO': '#00f2fe'
                    },
                    markers=True
                )
                
                fig.update_layout(height=500, hovermode='x unified')
                st.plotly_chart(fig, use_container_width=True)
                
                # Analyse des différences horaires
                st.subheader("🔍 Analyse des changements par heure")
                
                pivot = hourly_comparison.pivot(index='heure', columns='periode_jo', values='comptage')
                pivot['Diff_Apres_Avant'] = pivot['Après JO'] - pivot['Avant JO']
                pivot['Diff_Pct'] = (pivot['Diff_Apres_Avant'] / pivot['Avant JO'] * 100).fillna(0)
                
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=pivot.index,
                    y=pivot['Diff_Pct'],
                    marker=dict(
                        color=pivot['Diff_Pct'],
                        colorscale='RdYlGn',
                        cmin=-50,
                        cmax=50,
                        colorbar=dict(title="Variation %")
                    ),
                    text=pivot['Diff_Pct'].round(1),
                    textposition='outside',
                    hovertemplate='<b>%{x}h</b><br>Variation: %{y:.1f}%<extra></extra>'
                ))
                
                fig.update_layout(
                    title='Variation horaire: Après JO vs Avant JO',
                    xaxis_title='Heure',
                    yaxis_title='Variation (%)',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.subheader("💡 Insights clés et recommandations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div class="info-box success-box">
                <h4>🔍 Observations principales</h4>
                <ul>
                    <li>📊 <strong>Évolution globale:</strong> Analyse de la tendance post-JO</li>
                    <li>⏰ <strong>Changements horaires:</strong> Nouveaux patterns de déplacement</li>
                    <li>📅 <strong>Impact hebdomadaire:</strong> Différences semaine/weekend</li>
                    <li>🌡️ <strong>Variations saisonnières:</strong> Effets du contexte temporel</li>
                    <li>📍 <strong>Disparités géographiques:</strong> Zones les plus impactées</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="info-box warning-box">
                <h4>🎯 Recommandations stratégiques</h4>
                <ul>
                    <li>🚴 <strong>Infrastructures:</strong> Renforcer les axes à forte croissance</li>
                    <li>🅿️ <strong>Vélib':</strong> Optimiser la distribution aux heures de pointe</li>
                    <li>🚦 <strong>Signalisation:</strong> Adapter aux nouveaux flux</li>
                    <li>📈 <strong>Communication:</strong> Promouvoir le vélo post-JO</li>
                    <li>🔄 <strong>Suivi continu:</strong> Monitorer l'évolution long terme</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
            
            # Test statistique
            st.subheader("📊 Test statistique de significativité")
            
            from scipy import stats
            
            if len(avant) > 0 and len(apres) > 0:
                # Test t de Student
                t_stat, p_value = stats.ttest_ind(avant, apres)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("📊 Statistique t", f"{t_stat:.4f}")
                with col2:
                    st.metric("🎯 p-value", f"{p_value:.4f}")
                with col3:
                    significatif = p_value < 0.05
                    st.metric("✅ Significatif (α=0.05)", "Oui" if significatif else "Non")
                
                if significatif:
                    st.success("""
                    ✅ **Conclusion**: La différence entre les périodes "Avant JO" et "Après JO" 
                    est **statistiquement significative** (p < 0.05). L'impact des JO sur la mobilité 
                    cycliste est mesurable et probablement durable.
                    """)
                else:
                    st.info("""
                    ℹ️ **Conclusion**: La différence n'est pas statistiquement significative (p ≥ 0.05). 
                    L'impact observé pourrait être dû au hasard ou à d'autres facteurs.
                    """)

elif page == "🔮 Prédictions":
    st.header("🔮 Prédictions et Simulations Avancées")
    
    rf_model = load_model('random_forest_model.pkl')
    
    if rf_model is None:
        st.warning("⚠️ Modèle non disponible. Veuillez d'abord entraîner les modèles.")
    else:
        tab1, tab2, tab3 = st.tabs([
            "🔮 Prédiction Simple",
            "📅 Prédiction Période",
            "📊 Analyse de Scénarios"
        ])
        
        with tab1:
            st.subheader("🎯 Prédiction pour une date/heure spécifique")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                with st.form("prediction_simple_form"):
                    col_a, col_b, col_c = st.columns(3)
                    
                    with col_a:
                        date_pred = st.date_input(
                            "Date",
                            value=datetime.now().date(),
                            min_value=datetime(2024, 1, 1).date(),
                            max_value=datetime(2026, 12, 31).date()
                        )
                    
                    with col_b:
                        heure_pred = st.slider("Heure", 0, 23, 18)
                    
                    with col_c:
                        pendant_jo_pred = st.checkbox("Période JO", value=False)
                    
                    submitted = st.form_submit_button("🔮 Prédire", type="primary", use_container_width=True)
                    
                    if submitted:
                        try:
                            annee = date_pred.year
                            mois = date_pred.month
                            jour = date_pred.day
                            jour_semaine = date_pred.weekday()
                            est_weekend = jour_semaine in [5, 6]
                            semaine = pd.Timestamp(date_pred).isocalendar().week
                            
                            features = np.array([[
                                annee, mois, jour, heure_pred, jour_semaine,
                                int(est_weekend), int(pendant_jo_pred), semaine
                            ]])
                            
                            prediction = rf_model.predict(features)[0]
                            
                            # Intervalle de confiance
                            if hasattr(rf_model, 'estimators_'):
                                predictions_trees = np.array([
                                    tree.predict(features)[0] for tree in rf_model.estimators_
                                ])
                                ci_lower = np.percentile(predictions_trees, 2.5)
                                ci_upper = np.percentile(predictions_trees, 97.5)
                            else:
                                ci_lower = prediction * 0.85
                                ci_upper = prediction * 1.15
                            
                            st.session_state['last_prediction'] = {
                                'date': date_pred,
                                'heure': heure_pred,
                                'prediction': prediction,
                                'ci_lower': ci_lower,
                                'ci_upper': ci_upper
                            }
                        
                        except Exception as e:
                            st.error(f"❌ Erreur: {e}")
            
            with col2:
                if 'last_prediction' in st.session_state:
                    pred = st.session_state['last_prediction']
                    
                    st.markdown(f"""
                    <div class="info-box success-box">
                    <h3 style='margin: 0;'>🎯 Résultat</h3>
                    <h1 style='margin: 1rem 0; font-size: 3rem;'>{pred['prediction']:.0f}</h1>
                    <p style='margin: 0;'>passages prévus</p>
                    <hr style='border: 1px solid rgba(0,0,0,0.1);'>
                    <p style='margin: 0.5rem 0;'>
                        <strong>IC 95%:</strong> [{pred['ci_lower']:.0f}, {pred['ci_upper']:.0f}]
                    </p>
                    <p style='margin: 0;'>
                        <strong>Date:</strong> {pred['date'].strftime('%d/%m/%Y')}<br>
                        <strong>Heure:</strong> {pred['heure']:02d}h00
                    </p>
                    </div>
                    """, unsafe_allow_html=True)
        
        with tab2:
            st.subheader("📅 Prédictions sur une période")
            
            col1, col2 = st.columns(2)
            
            with col1:
                start_date = st.date_input(
                    "Date de début",
                    value=datetime.now().date(),
                    key="start_period_pred"
                )
            
            with col2:
                end_date = st.date_input(
                    "Date de fin",
                    value=(datetime.now() + timedelta(days=7)).date(),
                    key="end_period_pred"
                )
            
            col1, col2 = st.columns(2)
            
            with col1:
                heure_fixe = st.slider("Heure fixe", 0, 23, 18, key="heure_period_pred")
            
            with col2:
                export_csv = st.checkbox("Exporter en CSV", value=False)
            
            if st.button("📊 Générer les prédictions", type="primary"):
                if start_date >= end_date:
                    st.error("❌ La date de début doit être avant la date de fin")
                else:
                    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
                    predictions_list = []
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for idx, date in enumerate(date_range):
                        status_text.text(f"Calcul en cours... {idx+1}/{len(date_range)}")
                        
                        annee = date.year
                        mois = date.month
                        jour = date.day
                        jour_semaine = date.dayofweek
                        est_weekend = jour_semaine in [5, 6]
                        pendant_jo = datetime(2024, 7, 26) <= date <= datetime(2024, 8, 11)
                        semaine = date.isocalendar().week
                        
                        features = np.array([[
                            annee, mois, jour, heure_fixe, jour_semaine,
                            int(est_weekend), int(pendant_jo), semaine
                        ]])
                        
                        pred = rf_model.predict(features)[0]
                        
                        predictions_list.append({
                            'Date': date,
                            'Prédiction': pred,
                            'Jour': ['Lun', 'Mar', 'Mer', 'Jeu', 'Ven', 'Sam', 'Dim'][jour_semaine],
                            'Weekend': 'Oui' if est_weekend else 'Non'
                        })
                        
                        progress_bar.progress((idx + 1) / len(date_range))
                    
                    progress_bar.empty()
                    status_text.empty()
                    
                    df_pred = pd.DataFrame(predictions_list)
                    
                    # Visualisation
                    fig = px.line(
                        df_pred,
                        x='Date',
                        y='Prédiction',
                        color='Weekend',
                        title=f'Prédictions du trafic ({start_date} → {end_date}) à {heure_fixe}h',
                        labels={'Date': 'Date', 'Prédiction': 'Passages prédits'},
                        markers=True,
                        color_discrete_map={'Oui': '#FFA726', 'Non': '#1E88E5'}
                    )
                    
                    fig.update_layout(height=500, hovermode='x unified')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Statistiques
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("📊 Moyenne", f"{df_pred['Prédiction'].mean():.0f}")
                    with col2:
                        st.metric("📈 Maximum", f"{df_pred['Prédiction'].max():.0f}")
                    with col3:
                        st.metric("📉 Minimum", f"{df_pred['Prédiction'].min():.0f}")
                    with col4:
                        st.metric("📅 Jours", len(df_pred))
                    
                    # Export CSV
                    if export_csv:
                        csv = df_pred.to_csv(index=False)
                        st.download_button(
                            label="📥 Télécharger CSV",
                            data=csv,
                            file_name=f"predictions_{start_date}_{end_date}.csv",
                            mime="text/csv"
                        )
                    
                    # Tableau
                    with st.expander("📋 Tableau détaillé"):
                        st.dataframe(
                            df_pred.style.format({'Prédiction': '{:.0f}'}),
                            use_container_width=True
                        )
        
        with tab3:
            st.subheader("📊 Analyse comparative de scénarios")
            
            st.info("💡 Comparez l'impact de différents paramètres sur les prédictions")
            
            # Définition des scénarios
            scenarios_predefined = {
                '🌅 Heure de pointe matin (8h)': {'heure': 8, 'jour_semaine': 1},
                '🌆 Heure de pointe soir (18h)': {'heure': 18, 'jour_semaine': 3},
                '🌙 Heure creuse nuit (3h)': {'heure': 3, 'jour_semaine': 2},
                '📅 Lundi matin (9h)': {'jour_semaine': 0, 'heure': 9},
                '🎉 Samedi après-midi (15h)': {'jour_semaine': 5, 'heure': 15},
                '☀️ Dimanche matin (11h)': {'jour_semaine': 6, 'heure': 11},
                '🏅 Pendant JO (midi)': {'pendant_jo': 1, 'heure': 12},
                '📈 Après JO (18h)': {'pendant_jo': 0, 'heure': 18}
            }
            
            selected_scenarios = st.multiselect(
                "Sélectionner les scénarios:",
                options=list(scenarios_predefined.keys()),
                default=list(scenarios_predefined.keys())[:4]
            )
            
            if st.button("📊 Comparer", type="primary"):
                if not selected_scenarios:
                    st.warning("⚠️ Veuillez sélectionner au moins un scénario")
                else:
                    results = []
                    base_date = datetime(2024, 9, 15)
                    
                    for scenario_name in selected_scenarios:
                        params = {
                            'annee': base_date.year,
                            'mois': base_date.month,
                            'jour': base_date.day,
                            'heure': 12,
                            'jour_semaine': base_date.weekday(),
                            'est_weekend': 0,
                            'pendant_jo': 0,
                            'semaine': base_date.isocalendar().week
                        }
                        
                        params.update(scenarios_predefined[scenario_name])
                        
                        features = np.array([[
                            params['annee'], params['mois'], params['jour'],
                            params['heure'], params['jour_semaine'],
                            params['est_weekend'], params['pendant_jo'],
                            params['semaine']
                        ]])
                        
                        pred = rf_model.predict(features)[0]
                        
                        results.append({
                            'Scénario': scenario_name,
                            'Prédiction': pred
                        })
                    
                    df_scenarios = pd.DataFrame(results).sort_values('Prédiction', ascending=False)
                    
                    # Visualisation
                    fig = px.bar(
                        df_scenarios,
                        x='Scénario',
                        y='Prédiction',
                        title='Comparaison des scénarios',
                        color='Prédiction',
                        color_continuous_scale='Viridis',
                        text='Prédiction'
                    )
                    
                    fig.update_traces(texttemplate='%{text:.0f}', textposition='outside')
                    fig.update_layout(height=500, xaxis_tickangle=-45)
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Tableau
                    st.dataframe(
                        df_scenarios.style.format({'Prédiction': '{:.0f}'})
                        .background_gradient(cmap='RdYlGn', subset=['Prédiction']),
                        use_container_width=True
                    )

elif page == "📡 Données Temps Réel":
    st.header("📡 Données Vélib' en Temps Réel")
    
    st.info("🔄 Cette section affiche l'état actuel du réseau Vélib' via l'API Paris Open Data")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        refresh_btn = st.button("🔄 Actualiser les données", type="primary", use_container_width=True)
    
    with col2:
        auto_refresh = st.checkbox("Auto-refresh (30s)", value=False)
    
    if refresh_btn or auto_refresh:
        with st.spinner("⏳ Récupération des données..."):
            df_live = fetch_live_data()
            
            if not df_live.empty:
                st.success(f"✅ {len(df_live)} stations récupérées • {datetime.now().strftime('%H:%M:%S')}")
                
                # KPIs temps réel
                st.subheader("📊 État du réseau Vélib'")
                
                if all(col in df_live.columns for col in ['numbikesavailable', 'numdocksavailable', 'capacity']):
                    col1, col2, col3, col4 = st.columns(4)
                    
                    total_bikes = df_live['numbikesavailable'].sum()
                    total_docks = df_live['numdocksavailable'].sum()
                    total_capacity = df_live['capacity'].sum()
                    taux_utilisation = (total_bikes / total_capacity * 100) if total_capacity > 0 else 0
                    
                    with col1:
                        display_metric_card(
                            "🚲 Vélos disponibles",
                            format_number(total_bikes),
                            icon="🚲"
                        )
                    
                    with col2:
                        display_metric_card(
                            "🅿️ Places libres",
                            format_number(total_docks),
                            icon="🅿️"
                        )
                    
                    with col3:
                        display_metric_card(
                            "📍 Capacité totale",
                            format_number(total_capacity),
                            icon="📍"
                        )
                    
                    with col4:
                        display_metric_card(
                            "📊 Taux d'occupation",
                            f"{taux_utilisation:.1f}%",
                            icon="📊"
                        )
                    
                    st.markdown("---")
                    
                    # Carte interactive
                    st.subheader("🗺️ Carte des stations Vélib'")
                    
                    if 'coordonnees_geo' in df_live.columns:
                        try:
                            df_live['lat'] = df_live['coordonnees_geo'].apply(
                                lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None
                            )
                            df_live['lon'] = df_live['coordonnees_geo'].apply(
                                lambda x: x[1] if isinstance(x, list) and len(x) > 1 else None
                            )
                            
                            df_live = df_live.dropna(subset=['lat', 'lon'])
                            df_live['taux_remplissage'] = (
                                df_live['numbikesavailable'] / df_live['capacity'] * 100
                            ).fillna(0)
                            
                            fig = px.scatter_mapbox(
                                df_live,
                                lat='lat',
                                lon='lon',
                                color='taux_remplissage',
                                size='capacity',
                                hover_name='name' if 'name' in df_live.columns else None,
                                hover_data={
                                    'numbikesavailable': True,
                                    'numdocksavailable': True,
                                    'capacity': True,
                                    'lat': False,
                                    'lon': False,
                                    'taux_remplissage': ':.1f'
                                },
                                color_continuous_scale='RdYlGn',
                                size_max=15,
                                zoom=11,
                                height=600,
                                labels={
                                    'taux_remplissage': 'Taux (%)',
                                    'numbikesavailable': 'Vélos',
                                    'numdocksavailable': 'Places',
                                    'capacity': 'Capacité'
                                }
                            )
                            
                            fig.update_layout(
                                mapbox_style='open-street-map',
                                title='Disponibilité des stations Vélib\' (temps réel)'
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                        except Exception as e:
                            st.error(f"❌ Erreur carte: {e}")
                    
                    # Tableaux Top/Flop
                    st.subheader("🏆 Classement des stations")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### 🔴 Stations saturées (>90%)")
                        saturees = df_live[df_live['taux_remplissage'] > 90].nlargest(10, 'numbikesavailable')
                        
                        if not saturees.empty and 'name' in saturees.columns:
                            display_cols = ['name', 'numbikesavailable', 'capacity', 'taux_remplissage']
                            st.dataframe(
                                saturees[display_cols].rename(columns={
                                    'name': 'Station',
                                    'numbikesavailable': 'Vélos',
                                    'capacity': 'Capacité',
                                    'taux_remplissage': 'Taux %'
                                }).style.format({'Taux %': '{:.1f}'}),
                                use_container_width=True,
                                hide_index=True
                            )
                        else:
                            st.info("Aucune station saturée")
                    
                    with col2:
                        st.markdown("#### 🟢 Stations vides (<10%)")
                        vides = df_live[df_live['taux_remplissage'] < 10].nlargest(10, 'numdocksavailable')
                        
                        if not vides.empty and 'name' in vides.columns:
                            display_cols = ['name', 'numbikesavailable', 'numdocksavailable', 'taux_remplissage']
                            st.dataframe(
                                vides[display_cols].rename(columns={
                                    'name': 'Station',
                                    'numbikesavailable': 'Vélos',
                                    'numdocksavailable': 'Places',
                                    'taux_remplissage': 'Taux %'
                                }).style.format({'Taux %': '{:.1f}'}),
                                use_container_width=True,
                                hide_index=True
                            )
                        else:
                            st.info("Aucune station vide")
            else:
                st.error("❌ Impossible de récupérer les données")
    
    # Documentation API
    with st.expander("ℹ️ À propos des données temps réel"):
        st.markdown("""
        ### 📡 Source des données
        
        **API:** Paris Open Data - Vélib' Métropole  
        **Endpoint:** `https://opendata.paris.fr/api/records/1.0/search/`  
        **Dataset:** `velib-disponibilite-en-temps-reel`  
        **Fréquence de mise à jour:** Toutes les minutes
        
        ### 📊 Données disponibles
        
        | Champ | Description |
        |-------|-------------|
        | `numbikesavailable` | Nombre de vélos disponibles |
        | `numdocksavailable` | Nombre de places disponibles |
        | `capacity` | Capacité totale de la station |
        | `name` | Nom de la station |
        | `coordonnees_geo` | Coordonnées GPS [lat, lon] |
        | `stationcode` | Code unique de la station |
        
        ### 🔄 Actualisation
        
        - **Manuelle:** Cliquez sur "Actualiser les données"
        - **Automatique:** Activez "Auto-refresh" pour rafraîchir toutes les 30 secondes
        """)

elif page == "📥 Export & Rapports":
    st.header("📥 Export et Génération de Rapports")
    
    st.info("💡 Cette section permet d'exporter les données et de générer des rapports personnalisés")
    
    tab1, tab2, tab3 = st.tabs([
        "📊 Export Données",
        "📄 Rapport PDF",
        "📈 Graphiques"
    ])
    
    with tab1:
        st.subheader("📊 Export des données")
        
        if not df_filtered.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                format_export = st.selectbox(
                    "Format d'export:",
                    ["CSV", "Excel", "JSON"]
                )
            
            with col2:
                inclure_stats = st.checkbox("Inclure les statistiques", value=True)
            
            if st.button("📥 Générer l'export", type="primary"):
                try:
                    if format_export == "CSV":
                        csv = df_filtered.to_csv(index=False)
                        st.download_button(
                            label="💾 Télécharger CSV",
                            data=csv,
                            file_name=f"export_mobilite_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv"
                        )
                    
                    elif format_export == "Excel":
                        from io import BytesIO
                        output = BytesIO()
                        
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            df_filtered.to_excel(writer, sheet_name='Données', index=False)
                            
                            if inclure_stats:
                                stats = calculate_statistics(df_filtered, 'comptage')
                                stats_df = pd.DataFrame([stats]).T
                                stats_df.to_excel(writer, sheet_name='Statistiques')
                        
                        st.download_button(
                            label="💾 Télécharger Excel",
                            data=output.getvalue(),
                            file_name=f"export_mobilite_{datetime.now().strftime('%Y%m%d')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    
                    elif format_export == "JSON":
                        json_data = df_filtered.to_json(orient='records', date_format='iso')
                        st.download_button(
                            label="💾 Télécharger JSON",
                            data=json_data,
                            file_name=f"export_mobilite_{datetime.now().strftime('%Y%m%d')}.json",
                            mime="application/json"
                        )
                    
                    st.success("✅ Export généré avec succès!")
                
                except Exception as e:
                    st.error(f"❌ Erreur lors de l'export: {e}")
        else:
            st.warning("⚠️ Aucune donnée à exporter")
    
    with tab2:
        st.subheader("📄 Génération de rapport PDF")
        
        st.info("🚧 Fonctionnalité en cours de développement")
        
        st.markdown("""
        Le rapport PDF inclurait:
        - 📊 Résumé exécutif
        - 📈 Graphiques principaux
        - 📉 Analyses statistiques
        - 💡 Insights et recommandations
        """)
    
    with tab3:
        st.subheader("📈 Export des graphiques")
        
        st.info("💡 Tous les graphiques Plotly peuvent être exportés directement via leur menu contextuel")
        
        st.markdown("""
        ### Comment exporter un graphique:
        
        1. **Survolez** le graphique
        2. Cliquez sur l'**icône appareil photo** 📷
        3. Le graphique sera téléchargé en PNG
        
        ### Formats disponibles:
        - PNG (recommandé)
        - SVG (vectoriel)
        - JPEG
        """)

# ==========================================
# FOOTER
# ==========================================

st.markdown("---")
st.markdown(f"""
<div class="footer">
    <h4 style='margin: 0 0 1rem 0;'>🚴 Analyse de la Mobilité Post-JO Paris 2024</h4>
    <p style='margin: 0.5rem 0;'>
        <strong>Projet Data Science</strong> | Données: Paris Open Data | Développé avec ❤️ et Streamlit
    </p>
    <p style='margin: 0.5rem 0; color: #999;'>
        📊 Dernière mise à jour: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
    </p>
    <p style='margin: 1rem 0 0 0;'>
        <a href='https://github.com' target='_blank' style='margin: 0 1rem; color: #1E88E5;'>GitHub</a> |
        <a href='https://opendata.paris.fr' target='_blank' style='margin: 0 1rem; color: #1E88E5;'>Paris Open Data</a> |
        <a href='https://streamlit.io' target='_blank' style='margin: 0 1rem; color: #1E88E5;'>Streamlit</a>
    </p>
</div>
""", unsafe_allow_html=True)

# Auto-refresh pour les données temps réel
if page == "📡 Données Temps Réel" and 'auto_refresh' in locals() and auto_refresh:
    import time
    time.sleep(30)
    st.rerun()