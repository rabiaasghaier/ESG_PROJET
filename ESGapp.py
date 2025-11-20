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
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import io
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="Analyse ESG",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
st.markdown("""
    <style>
    .main-header {
        font-size: 42px;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 30px;
    }
    .sub-header {
        font-size: 24px;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Titre principal
st.markdown('<p class="main-header">🌍 Analyse ESG (Environnement, Social, Gouvernance)</p>', unsafe_allow_html=True)

# Initialisation de session_state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'df_esg' not in st.session_state:
    st.session_state.df_esg = None

# Fonctions utilitaires
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
    """Traite et fusionne les données ESG"""
    df_E_long = clean_and_melt(data_E, 'E')
    df_S_long = clean_and_melt(data_S, 'S')
    df_G_long = clean_and_melt(data_G, 'G')
    
    df_combined = pd.concat([df_E_long, df_S_long, df_G_long], ignore_index=True)
    
    pivot_index = ['Country Name', 'Country Code', 'Year']
    df_final = df_combined.pivot_table(index=pivot_index, columns='Indicator',
                                       values='Value', aggfunc='first')
    df_final.reset_index(inplace=True)
    df_final.columns.name = None
    
    indicator_cols = df_final.columns[3:]
    df_final[indicator_cols] = df_final[indicator_cols].apply(lambda x: x.fillna(x.mean()), axis=0)
    
    scaler = MinMaxScaler()
    df_final[indicator_cols] = scaler.fit_transform(df_final[indicator_cols])
    
    e_cols = [col for col in indicator_cols if col.endswith('_E')]
    s_cols = [col for col in indicator_cols if col.endswith('_S')]
    g_cols = [col for col in indicator_cols if col.endswith('_G')]
    
    df_final['Score_E'] = df_final[e_cols].mean(axis=1)
    df_final['Score_S'] = df_final[s_cols].mean(axis=1)
    df_final['Score_G'] = df_final[g_cols].mean(axis=1)
    df_final['Score_ESG_Total'] = df_final[['Score_E', 'Score_S', 'Score_G']].mean(axis=1)
    
    # Feature Engineering
    df_final.sort_values(by=['Country Code', 'Year'], inplace=True)
    for score in ['Score_E', 'Score_S', 'Score_G', 'Score_ESG_Total']:
        df_final[f'{score}_Lag1'] = df_final.groupby('Country Code')[score].shift(1)
        df_final[f'{score}_Change'] = df_final[score] - df_final[f'{score}_Lag1']
    df_final.fillna(0, inplace=True)
    
    # Catégorisation
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

# Sidebar - Navigation
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Choisissez une section:",
    ["📁 Chargement des Données", 
     "📈 Scores ESG", 
     "🔍 Analyse Exploratoire",
     "🎯 Feature Importance",
     "🤖 Machine Learning",
     "🧠 Deep Learning",
     "🎨 Clustering"]
)

# PAGE 1: Chargement des données
if page == "📁 Chargement des Données":
    st.markdown('<p class="sub-header">Chargement et Préparation des Données</p>', unsafe_allow_html=True)
    
    # Chemins des fichiers
    base_path = r"C:\Users\rabia\OneDrive\Bureau\StreamlitEsg"
    
    # Option pour choisir entre chargement automatique ou manuel
    load_option = st.radio("Mode de chargement :", ["📂 Automatique (depuis dossier)", "📤 Manuel (upload)"])
    
    if load_option == "📂 Automatique (depuis dossier)":
        st.info(f"📁 Chargement depuis : {base_path}")
        
        if st.button("🚀 Charger et Traiter les Données", type="primary"):
            try:
                with st.spinner("Chargement des fichiers en cours..."):
                    data_E = pd.read_excel(f"{base_path}\\environment.xlsx")
                    data_S = pd.read_excel(f"{base_path}\\social.xlsx")
                    data_G = pd.read_excel(f"{base_path}\\governance.xlsx")
                    
                    st.success("✅ Fichiers chargés depuis le dossier local!")
            except FileNotFoundError as e:
                st.error(f"❌ Erreur : Fichier non trouvé. Vérifiez que les fichiers existent dans le dossier : {base_path}")
                st.stop()
            except Exception as e:
                st.error(f"❌ Erreur lors du chargement : {str(e)}")
                st.stop()
    
    else:  # Mode Manuel
        st.info("📤 Veuillez charger les trois fichiers Excel : environment.xlsx, social.xlsx, et governance.xlsx")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            file_E = st.file_uploader("🌱 Environnement (E)", type=['xlsx'])
        with col2:
            file_S = st.file_uploader("👥 Social (S)", type=['xlsx'])
        with col3:
            file_G = st.file_uploader("⚖️ Gouvernance (G)", type=['xlsx'])
        
        if file_E and file_S and file_G:
            if st.button("🚀 Traiter les Données", type="primary"):
                with st.spinner("Traitement des données en cours..."):
                    data_E = pd.read_excel(file_E)
                    data_S = pd.read_excel(file_S)
                    data_G = pd.read_excel(file_G)
                    
                    # Traitement des données
                    df_esg = process_data(data_E, data_S, data_G)
                    st.session_state.df_esg = df_esg
                    st.session_state.data_loaded = True
                    
                    st.success("✅ Données chargées et traitées avec succès!")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Observations", f"{len(df_esg):,}")
                    with col2:
                        st.metric("Pays", df_esg['Country Code'].nunique())
                    with col3:
                        st.metric("Années", f"{df_esg['Year'].min()}-{df_esg['Year'].max()}")
                    with col4:
                        st.metric("Indicateurs", len([c for c in df_esg.columns if c.endswith(('_E', '_S', '_G'))]))
                    
                    st.dataframe(df_esg.head(10), use_container_width=True)
        else:
            st.stop()

    if load_option == "📂 Automatique (depuis dossier)" or (file_E and file_S and file_G):
        if 'data_E' in locals():
            with st.spinner("Traitement des données en cours..."):
                df_esg = process_data(data_E, data_S, data_G)
                st.session_state.df_esg = df_esg
                st.session_state.data_loaded = True
                st.success("✅ Données chargées et traitées avec succès!")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Observations", f"{len(df_esg):,}")
            with col2:
                st.metric("Pays", df_esg['Country Code'].nunique())
            with col3:
                st.metric("Années", f"{df_esg['Year'].min()}-{df_esg['Year'].max()}")
            with col4:
                st.metric("Indicateurs", len([c for c in df_esg.columns if c.endswith(('_E', '_S', '_G'))]))
            
            st.dataframe(df_esg.head(10), use_container_width=True)

# PAGE 2: Scores ESG
elif page == "📈 Scores ESG":
    if not st.session_state.data_loaded:
        st.warning("⚠️ Veuillez d'abord charger les données dans la section 'Chargement des Données'")
    else:
        df_esg = st.session_state.df_esg
        st.markdown('<p class="sub-header">Analyse des Scores ESG</p>', unsafe_allow_html=True)
        
        # Filtres
        col1, col2, col3 = st.columns(3)
        with col1:
            countries = ['Tous'] + sorted(df_esg['Country Name'].unique().tolist())
            selected_country = st.selectbox("🌍 Pays", countries)
        with col2:
            years = ['Toutes'] + sorted(df_esg['Year'].unique().tolist(), reverse=True)
            selected_year = st.selectbox("📅 Année", years)
        with col3:
            pillar = st.selectbox("🎯 Pilier ESG", ['Total', 'E', 'S', 'G'])
        
        # Filtrage
        df_filtered = df_esg.copy()
        if selected_country != 'Tous':
            df_filtered = df_filtered[df_filtered['Country Name'] == selected_country]
        if selected_year != 'Toutes':
            df_filtered = df_filtered[df_filtered['Year'] == int(selected_year)]
        
        # Affichage des scores
        score_col = f'Score_{pillar}' if pillar != 'Total' else 'Score_ESG_Total'
        
        st.markdown("### 📊 Distribution des Scores")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = px.histogram(df_filtered, x=score_col, nbins=30,
                             title=f"Distribution du Score {pillar}",
                             labels={score_col: f'Score {pillar}'})
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.metric("Score Moyen", f"{df_filtered[score_col].mean():.3f}")
            st.metric("Score Médian", f"{df_filtered[score_col].median():.3f}")
            st.metric("Score Min", f"{df_filtered[score_col].min():.3f}")
            st.metric("Score Max", f"{df_filtered[score_col].max():.3f}")
        
        # Top 10
        st.markdown("### 🏆 Top 10 des Pays")
        top10 = df_filtered.nlargest(10, score_col)[['Country Name', 'Year', score_col]]
        
        fig = px.bar(top10, x='Country Name', y=score_col, color=score_col,
                    title=f"Top 10 - Score {pillar}",
                    color_continuous_scale='Viridis')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(top10, use_container_width=True)

# PAGE 3: Analyse Exploratoire
elif page == "🔍 Analyse Exploratoire":
    if not st.session_state.data_loaded:
        st.warning("⚠️ Veuillez d'abord charger les données")
    else:
        df_esg = st.session_state.df_esg
        st.markdown('<p class="sub-header">Analyse Exploratoire des Données</p>', unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["📊 Statistiques", "📈 Évolution Temporelle", "🗺️ Heatmap"])
        
        with tab1:
            st.markdown("### Statistiques Descriptives des Scores ESG")
            stats = df_esg[['Score_E', 'Score_S', 'Score_G', 'Score_ESG_Total']].describe()
            st.dataframe(stats.T, use_container_width=True)
            
            # Distribution des catégories
            fig = px.pie(df_esg, names='ESG_Category', title="Distribution des Catégories ESG",
                        color='ESG_Category', color_discrete_map={'Faible':'red', 'Moyen':'orange', 'Élevé':'green'})
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            country = st.selectbox("Sélectionnez un pays", sorted(df_esg['Country Name'].unique()))
            df_country = df_esg[df_esg['Country Name'] == country].sort_values('Year')
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_country['Year'], y=df_country['Score_E'], 
                                    mode='lines+markers', name='Environnement'))
            fig.add_trace(go.Scatter(x=df_country['Year'], y=df_country['Score_S'], 
                                    mode='lines+markers', name='Social'))
            fig.add_trace(go.Scatter(x=df_country['Year'], y=df_country['Score_G'], 
                                    mode='lines+markers', name='Gouvernance'))
            fig.add_trace(go.Scatter(x=df_country['Year'], y=df_country['Score_ESG_Total'], 
                                    mode='lines+markers', name='Total', line=dict(width=3)))
            
            fig.update_layout(title=f"Évolution des Scores ESG - {country}",
                            xaxis_title="Année", yaxis_title="Score",
                            height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.markdown("### Heatmap des Scores ESG par Pays et Année")
            top_countries = df_esg.groupby('Country Name')['Score_ESG_Total'].mean().nlargest(20).index
            df_heat = df_esg[df_esg['Country Name'].isin(top_countries)]
            pivot_data = df_heat.pivot_table(values='Score_ESG_Total', 
                                            index='Country Name', 
                                            columns='Year')
            
            fig = px.imshow(pivot_data, 
                          labels=dict(x="Année", y="Pays", color="Score ESG"),
                          color_continuous_scale='RdYlGn',
                          aspect="auto")
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)

# PAGE 4: Feature Importance
elif page == "🎯 Feature Importance":
    if not st.session_state.data_loaded:
        st.warning("⚠️ Veuillez d'abord charger les données")
    else:
        df_esg = st.session_state.df_esg
        st.markdown('<p class="sub-header">Analyse d\'Importance des Caractéristiques</p>', unsafe_allow_html=True)
        
        X_cols = [col for col in df_esg.columns if col.endswith(('_E', '_S', '_G')) or col.endswith('_Lag1')]
        X = df_esg[X_cols]
        y = df_esg['ESG_Target']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        
        method = st.radio("Choisissez une méthode:", 
                         ["XGBoost Feature Importance", "Permutation Importance", "RFE"])
        
        if st.button("🔍 Analyser l'Importance", type="primary"):
            with st.spinner("Calcul en cours..."):
                if method == "XGBoost Feature Importance":
                    xgb_model = xgb.XGBClassifier(objective='multi:softmax', num_class=3, 
                                                 use_label_encoder=False, eval_metric='mlogloss', 
                                                 random_state=42)
                    xgb_model.fit(X_train, y_train)
                    importance = pd.Series(xgb_model.feature_importances_, index=X_cols).sort_values(ascending=False)
                    
                elif method == "Permutation Importance":
                    xgb_model = xgb.XGBClassifier(objective='multi:softmax', num_class=3, 
                                                 use_label_encoder=False, eval_metric='mlogloss', 
                                                 random_state=42)
                    xgb_model.fit(X_train, y_train)
                    perm_importance = permutation_importance(xgb_model, X_test, y_test, 
                                                           n_repeats=5, random_state=42, n_jobs=-1)
                    importance = pd.Series(perm_importance.importances_mean, index=X_cols).sort_values(ascending=False)
                
                else:  # RFE
                    rf_model = RandomForestClassifier(random_state=42)
                    rfe = RFE(estimator=rf_model, n_features_to_select=20, step=1)
                    rfe.fit(X_train, y_train)
                    importance = pd.Series(rfe.ranking_, index=X_cols).sort_values()
                
                # Affichage
                top_features = importance.head(15)
                fig = px.bar(x=top_features.values, y=top_features.index, 
                           orientation='h', title=f"Top 15 - {method}",
                           labels={'x': 'Importance', 'y': 'Caractéristique'})
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
                
                st.dataframe(top_features.reset_index().rename(columns={'index': 'Feature', 0: 'Importance'}),
                           use_container_width=True)

# PAGE 5: Machine Learning
elif page == "🤖 Machine Learning":
    if not st.session_state.data_loaded:
        st.warning("⚠️ Veuillez d'abord charger les données")
    else:
        df_esg = st.session_state.df_esg
        st.markdown('<p class="sub-header">Modèle Machine Learning - Random Forest</p>', unsafe_allow_html=True)
        
        X_cols = [col for col in df_esg.columns if col.endswith(('_E', '_S', '_G')) or col.endswith('_Lag1')]
        X = df_esg[X_cols]
        y = df_esg['ESG_Target']
        
        col1, col2 = st.columns(2)
        with col1:
            test_size = st.slider("Taille de l'ensemble de test", 0.1, 0.5, 0.3)
        with col2:
            n_estimators = st.slider("Nombre d'arbres", 50, 500, 100, 50)
        
        if st.button("🚀 Entraîner le Modèle", type="primary"):
            with st.spinner("Entraînement en cours..."):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, 
                                                                    random_state=42, stratify=y)
                
                rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=42, 
                                                 class_weight='balanced')
                rf_model.fit(X_train, y_train)
                y_pred = rf_model.predict(X_test)
                
                st.success("✅ Modèle entraîné avec succès!")
                
                # Métriques
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### 📊 Rapport de Classification")
                    report = classification_report(y_test, y_pred, 
                                                  target_names=['Faible', 'Moyen', 'Élevé'],
                                                  output_dict=True)
                    st.dataframe(pd.DataFrame(report).T, use_container_width=True)
                
                with col2:
                    st.markdown("### 📈 Matrice de Confusion")
                    cm = confusion_matrix(y_test, y_pred)
                    fig = px.imshow(cm, text_auto=True, 
                                  labels=dict(x="Prédit", y="Réel"),
                                  x=['Faible', 'Moyen', 'Élevé'],
                                  y=['Faible', 'Moyen', 'Élevé'],
                                  color_continuous_scale='Blues')
                    st.plotly_chart(fig, use_container_width=True)
                
                # Feature Importance
                st.markdown("### 🎯 Top 15 Features")
                importance = pd.Series(rf_model.feature_importances_, index=X_cols).sort_values(ascending=False)
                top_features = importance.head(15)
                fig = px.bar(x=top_features.values, y=top_features.index, orientation='h')
                st.plotly_chart(fig, use_container_width=True)

# PAGE 6: Deep Learning
elif page == "🧠 Deep Learning":
    if not st.session_state.data_loaded:
        st.warning("⚠️ Veuillez d'abord charger les données")
    else:
        df_esg = st.session_state.df_esg
        st.markdown('<p class="sub-header">Modèle Deep Learning - Réseau de Neurones</p>', unsafe_allow_html=True)
        
        X_cols = [col for col in df_esg.columns if col.endswith(('_E', '_S', '_G')) or col.endswith('_Lag1')]
        X = df_esg[X_cols]
        y = df_esg['ESG_Target']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            epochs = st.slider("Époques", 10, 100, 50, 10)
        with col2:
            batch_size = st.slider("Batch Size", 16, 128, 32, 16)
        with col3:
            dropout = st.slider("Dropout", 0.1, 0.5, 0.3, 0.1)
        
        if st.button("🚀 Entraîner le Modèle", type="primary"):
            with st.spinner("Entraînement en cours..."):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                                    random_state=42, stratify=y)
                
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                model = Sequential([
                    Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
                    Dropout(dropout),
                    Dense(64, activation='relu'),
                    Dropout(dropout),
                    Dense(3, activation='softmax')
                ])
                
                model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', 
                            metrics=['accuracy'])
                
                history = model.fit(X_train_scaled, y_train, 
                                  validation_data=(X_test_scaled, y_test),
                                  epochs=epochs, batch_size=batch_size, verbose=0)
                
                st.success("✅ Modèle entraîné avec succès!")
                
                loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("📉 Loss sur Test", f"{loss:.4f}")
                with col2:
                    st.metric("🎯 Accuracy sur Test", f"{accuracy:.4f}")
                
                # Learning Curves
                st.markdown("### 📈 Courbes d'Apprentissage")
                
                col1, col2 = st.columns(2)
                with col1:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(y=history.history['loss'], mode='lines', name='Train'))
                    fig.add_trace(go.Scatter(y=history.history['val_loss'], mode='lines', name='Validation'))
                    fig.update_layout(title="Loss", xaxis_title="Époque", yaxis_title="Loss")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(y=history.history['accuracy'], mode='lines', name='Train'))
                    fig.add_trace(go.Scatter(y=history.history['val_accuracy'], mode='lines', name='Validation'))
                    fig.update_layout(title="Accuracy", xaxis_title="Époque", yaxis_title="Accuracy")
                    st.plotly_chart(fig, use_container_width=True)

# PAGE 7: Clustering
elif page == "🎨 Clustering":
    if not st.session_state.data_loaded:
        st.warning("⚠️ Veuillez d'abord charger les données")
    else:
        df_esg = st.session_state.df_esg
        st.markdown('<p class="sub-header">Clustering K-Means des Scores ESG</p>', unsafe_allow_html=True)
        
        n_clusters = st.slider("Nombre de clusters", 2, 10, 3)
        
        if st.button("🎨 Effectuer le Clustering", type="primary"):
            with st.spinner("Clustering en cours..."):
                clustering_data = df_esg[['Score_E', 'Score_S', 'Score_G', 'Score_ESG_Total']].copy()
                scaler = StandardScaler()
                data_scaled = scaler.fit_transform(clustering_data)
                
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                df_esg['Cluster'] = kmeans.fit_predict(data_scaled)
                
                # PCA pour visualisation
                pca = PCA(n_components=2)
                data_pca = pca.fit_transform(data_scaled)
                df_esg['PCA1'] = data_pca[:, 0]
                df_esg['PCA2'] = data_pca[:, 1]
                
                st.success("✅ Clustering effectué!")
                
                # Centres des clusters
                st.markdown("### 🎯 Centres des Clusters")
                centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_),
                                     columns=clustering_data.columns)
                centers['Cluster'] = range(n_clusters)
                st.dataframe(centers, use_container_width=True)
                
                # Visualisation PCA
                st.markdown("### 📊 Visualisation des Clusters (PCA)")
                fig = px.scatter(df_esg, x='PCA1', y='PCA2', color='Cluster',
                               hover_data=['Country Name', 'Year', 'Score_ESG_Total'],
                               title=f"Clustering K-Means (K={n_clusters})",
                               labels={'PCA1': f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)',
                                     'PCA2': f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)'})
                fig.update_traces(marker=dict(size=8))
                st.plotly_chart(fig, use_container_width=True)
                
                # Distribution par cluster
                st.markdown("### 📈 Distribution des Scores par Cluster")
                fig = px.box(df_esg, x='Cluster', y='Score_ESG_Total', color='Cluster',
                           title="Distribution des Scores ESG par Cluster")
                st.plotly_chart(fig, use_container_width=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.info("""
    ### 📚 À propos
    Application d'analyse ESG développée avec Streamlit.
    
    **Fonctionnalités:**
    - Chargement et traitement des données
    - Calcul des scores ESG
    - Visualisations interactives
    - Machine Learning & Deep Learning
    - Clustering K-Means
""")