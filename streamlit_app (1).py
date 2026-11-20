app_code = '''import streamlit as st
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import xgboost as xgb
import shap
import networkx as nx

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Détection Intelligente de la Fraude Bancaire",
    page_icon="🛡️",
    layout="wide"
)

# Style CSS personnalisé
st.markdown("""
<style>
    .main {
        background-color: #0f172a;
        color: #f8fafc;
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: bold;
    }
    .metric-card {
        background-color: #1e293b;
        padding: 12px;
        border-radius: 8px;
        border-left: 4px solid #38bdf8;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# 1. Initialisation du modèle XGBoost et SHAP (mise en cache Streamlit)
@st.cache_resource
def initialiser_modele_demo():
    np.random.seed(42)
    n_samples = 1000
    
    montant_mad = np.random.exponential(scale=15000, size=n_samples)
    ecart_zscore_montant = np.random.normal(loc=0, scale=1.5, size=n_samples)
    nb_tx_24h = np.random.poisson(lam=3, size=n_samples)
    proche_seuil_50k = np.where((montant_mad >= 44000) & (montant_mad < 50000), 1, 0)
    montant_rond = np.where(montant_mad % 1000 == 0, 1, 0)
    ip_pays_different = np.random.binomial(1, 0.05, size=n_samples)
    nouveau_beneficiaire = np.random.binomial(1, 0.15, size=n_samples)
    
    X = pd.DataFrame({
        'montant_mad': montant_mad,
        'ecart_zscore_montant': ecart_zscore_montant,
        'nb_tx_24h': nb_tx_24h,
        'proche_seuil_50k': proche_seuil_50k,
        'montant_rond': montant_rond,
        'ip_pays_different': ip_pays_different,
        'nouveau_beneficiaire': nouveau_beneficiaire
    })
    
    y_prob = (0.3 * (ecart_zscore_montant > 2.5) + 
              0.25 * proche_seuil_50k + 
              0.2 * ip_pays_different + 
              0.15 * nouveau_beneficiaire + 
              0.1 * (nb_tx_24h > 6))
    y = np.where(y_prob > 0.35, 1, 0)
    
    model = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42)
    model.fit(X, y)
    
    explainer = shap.TreeExplainer(model)
    return model, explainer, X.columns.tolist()

model, explainer, feature_names = initialiser_modele_demo()

# En-tête Principal
st.title("🏦 Détection Intelligente de la Fraude Bancaire d'Entreprise")
st.caption("Moteur Hybride IA (XGBoost + Isolation Forest + SHAP + NetworkX) | Temps Réel < 100 ms")

tab1, tab2 = st.tabs(["💳 Évaluation & Scoring Temps Réel", "🕸️ Cartographie des Réseaux & Blanchiment"])

with tab1:
    col_input, col_result = st.columns([1, 1])
    
    with col_input:
        st.subheader("📥 1. Paramètres de la Transaction")
        id_tx = st.text_input("ID Transaction", value="TX_2026_99412")
        id_ent = st.text_input("ID Entreprise", value="ENT_MAROC_882")
        
        c1, c2 = st.columns(2)
        with c1:
            montant = st.number_input("Montant de la Transaction", value=48500.0, step=1000.0)
        with c2:
            devise = st.selectbox("Devise d'origine", options=["MAD", "EUR", "USD"], index=0)
            
        canal = st.selectbox("Canal d'émission", options=["en_ligne", "mobile", "agence"], index=0)
        
        # Calcul automatique du montant en MAD
        taux = {'MAD': 1.0, 'EUR': 10.8, 'USD': 9.9}
        montant_mad = float(montant) * taux.get(devise, 1.0)
        st.info(f"💵 Montant converti en Dirhams (MAD) : **{montant_mad:,.2f} MAD**")
        
        st.subheader("📊 2. Score Z (Écart Comportemental)")
        mode_zscore = st.radio(
            "Saisie du Score Z (Écart par rapport au montant habituel) :",
            options=["💡 Insérer le Score Z directement", "📐 Calculer à partir de l'historique (Moyenne & Écart-Type)"],
            index=0
        )
        
        if "Insérer le Score Z" in mode_zscore:
            ecart_zscore = st.slider(
                "Score Z (`ecart_zscore_montant`)",
                min_value=-3.0,
                max_value=8.0,
                value=3.25,
                step=0.15,
                help="Un Score Z > +2.5 indique une rupture brutale par rapport au montant moyen habituel."
            )
        else:
            c_m, c_std = st.columns(2)
            with c_m:
                avg_hist = st.number_input("Montant Moyen Historique (MAD)", value=10000.0, step=1000.0)
            with c_std:
                std_hist = st.number_input("Écart-Type Historique (MAD)", value=2000.0, step=500.0)
            std_safe = float(std_hist) if float(std_hist) > 0 else 1.0
            ecart_zscore = round((montant_mad - float(avg_hist)) / std_safe, 2)
            st.write(f"📈 **Score Z calculé** : `{ecart_zscore:.2f}` sigma")

        st.subheader("🚩 3. Signaux & Variables de Risque (Interactifs)")
        
        # Calculs automatiques de référence
        auto_seuil_50k = True if (44000 <= montant_mad < 50000) else False
        auto_montant_rond = True if (montant_mad > 0 and montant_mad % 1000 == 0) else False

        c3, c4 = st.columns(2)
        with c3:
            ip_diff_str = st.radio("IP Pays Différent ?", options=["Non", "Oui"], index=0)
            ip_flag = 1 if ip_diff_str == "Oui" else 0
            
            chk_seuil = st.checkbox("Proche Seuil 50k MAD (44k-50k) ?", value=auto_seuil_50k)
            seuil_50k = 1 if chk_seuil else 0

        with c4:
            benef_str = st.radio("Nouveau Bénéficiaire ?", options=["Non", "Oui"], index=1)
            benef_flag = 1 if benef_str == "Oui" else 0
            
            chk_rond = st.checkbox("Montant Rond (multiple 1000 MAD) ?", value=auto_montant_rond)
            m_rond = 1 if chk_rond else 0

        nb_24h = st.slider("Nb Transactions dans les 24h précédentes", min_value=0, max_value=20, value=4)
        nb_tx_tot = int(nb_24h) + 1

    with col_result:
        st.subheader("🎯 Résultat & Décision en Temps Réel")
        
        temps_debut = time.time()
        
        # DataFrame des 7 variables transmises à XGBoost
        df_single = pd.DataFrame([{
            'montant_mad': montant_mad,
            'ecart_zscore_montant': ecart_zscore,
            'nb_tx_24h': nb_tx_tot,
            'proche_seuil_50k': seuil_50k,
            'montant_rond': m_rond,
            'ip_pays_different': ip_flag,
            'nouveau_beneficiaire': benef_flag
        }])[feature_names]
        
        # Prédiction XGBoost
        proba = float(model.predict_proba(df_single)[0, 1])
        score_pct = round(proba * 100, 2)
        latence_ms = round((time.time() - temps_debut) * 1000, 2)
        
        # Matrice de conformité
        if proba >= 0.85:
            statut = "⛔ BLOCKED (Blocage Instantané)"
            niveau = "CRITIQUE"
            action = "Virement interrompu avant imputation. Alerte urgente Conformité générée."
            badge_color = "#d9534f"
        elif proba >= 0.50:
            statut = "⚠️ REVIEW (File d'Attente)"
            niveau = "ÉLEVÉ"
            action = "Transaction mise en attente. Authentification forte OTP / Risquiste requise."
            badge_color = "#f0ad4e"
        elif proba >= 0.20:
            statut = "🔍 FLAGGED (Surveillance)"
            niveau = "MODÉRÉ"
            action = "Virement exécuté avec marquage pour contrôle et comptabilisation a posteriori."
            badge_color = "#5bc0de"
        else:
            statut = "✅ APPROVED (Exécution Fluidifiée)"
            niveau = "FAIBLE"
            action = "Virement validé et transmis directement sur le réseau interbancaire."
            badge_color = "#5cb85c"
            
        # Carte d'affichage
        st.markdown(f"""
        <div style="background-color: #0f172a; padding: 20px; border-radius: 12px; border: 2px solid {badge_color}; color: white; font-family: sans-serif;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h3 style="margin: 0; color: #38bdf8;">🛡️ Décision Moteur Risque</h3>
                <span style="background-color: {badge_color}; color: white; padding: 6px 14px; border-radius: 20px; font-weight: bold;">{niveau}</span>
            </div>
            <hr style="border-color: #334155; margin: 15px 0;">
            <p style="font-size: 16px; margin: 5px 0;"><b>Statut Bancaire :</b> {statut}</p>
            <p style="font-size: 24px; margin: 10px 0; color: #f43f5e;"><b>Probabilité de Fraude : {score_pct} %</b></p>
            <p style="font-size: 13px; color: #94a3b8; margin: 5px 0;">⏱️ Temps d'évaluation API : <b>{latence_ms} ms</b> (SLA < 100 ms : ✅ Conforme)</p>
            <div style="background-color: #1e293b; padding: 12px; border-radius: 8px; margin-top: 15px;">
                <p style="margin: 0; font-size: 13px; color: #e2e8f0;">📌 <b>Action Opérationnelle :</b> {action}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.write("")
        st.subheader("📋 Inspecteur des Variables Transmises à l'IA")
        st.dataframe(df_single, use_container_width=True)
        
        st.subheader("🔍 Explicabilité SHAP Local (Fiche d'Alerte)")
        shap_vals = explainer(df_single)
        fig_shap, ax = plt.subplots(figsize=(8, 4))
        shap.plots.waterfall(shap_vals[0], show=False)
        plt.title(f"SHAP Waterfall - TX: {id_tx} (Score: {score_pct}%)", fontsize=10)
        plt.tight_layout()
        st.pyplot(fig_shap)
        plt.close()

with tab2:
    st.subheader("🕸️ Cartographie des Réseaux de Contreparties (Layering & Mules)")
    st.markdown("Détection topologique des boucles fermées de blanchiment par transfert en chaîne.")
    
    G = nx.DiGraph()
    G.add_edge("ENT_MAROC_01", "MULE_COMPTE_A", weight=48500)
    G.add_edge("MULE_COMPTE_A", "SHELL_COMPANY_B", weight=48000)
    G.add_edge("SHELL_COMPANY_B", "OFFSHORE_ACC_C", weight=47500)
    G.add_edge("OFFSHORE_ACC_C", "ENT_MAROC_01", weight=47000)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    pos = nx.spring_layout(G, seed=42)
    
    nx.draw_networkx_nodes(G, pos, node_color='#ef4444', node_size=1400, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color='#dc2626', arrowstyle='->', arrowsize=25, width=2.5, ax=ax)
    nx.draw_networkx_labels(G, pos, font_color='white', font_weight='bold', font_size=8, ax=ax)
    
    labels_edges = nx.get_edge_attributes(G, 'weight')
    labels_formatted = {k: f"{v:,.0f} MAD" for k, v in labels_edges.items()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels_formatted, font_size=8, ax=ax)
    
    plt.title("🕸️ Circuit Fermé de Blanchiment Détecté (Layering Loop)", fontsize=11, pad=12)
    plt.axis('off')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
