import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# ============================================================
#  APP : Classification de matériaux — Génie des Matériaux
# ============================================================

st.write("""
# 🧱 Classification de Matériaux
### Prédiction du type de matériau à partir de ses propriétés mécaniques
Cette application prédit si un matériau est un **Acier**, un **Aluminium** ou un **Polymère**
en fonction de ses propriétés mécaniques.
""")

# ============================================================
#  DATASET SYNTHÉTIQUE
#  (pas besoin de fichier CSV externe)
# ============================================================
np.random.seed(42)
n = 150

# Acier : dureté haute, densité haute, résistance haute, module élevé
acier = {
    'durete':        np.random.uniform(150, 300, n//3),
    'densite':       np.random.uniform(7.5, 8.0, n//3),
    'resistance':    np.random.uniform(400, 800, n//3),
    'module_young':  np.random.uniform(180, 220, n//3),
    'materiau':      ['Acier'] * (n//3)
}

# Aluminium : dureté moyenne, densité faible, résistance moyenne
aluminium = {
    'durete':        np.random.uniform(40, 120, n//3),
    'densite':       np.random.uniform(2.5, 2.9, n//3),
    'resistance':    np.random.uniform(100, 400, n//3),
    'module_young':  np.random.uniform(65, 75, n//3),
    'materiau':      ['Aluminium'] * (n//3)
}

# Polymère : dureté faible, densité très faible, résistance faible
polymere = {
    'durete':        np.random.uniform(5, 40,  n//3),
    'densite':       np.random.uniform(0.9, 1.4, n//3),
    'resistance':    np.random.uniform(10, 100, n//3),
    'module_young':  np.random.uniform(0.5, 5,  n//3),
    'materiau':      ['Polymère'] * (n//3)
}

df = pd.DataFrame({
    'durete':       np.concatenate([acier['durete'],       aluminium['durete'],       polymere['durete']]),
    'densite':      np.concatenate([acier['densite'],      aluminium['densite'],      polymere['densite']]),
    'resistance':   np.concatenate([acier['resistance'],   aluminium['resistance'],   polymere['resistance']]),
    'module_young': np.concatenate([acier['module_young'], aluminium['module_young'], polymere['module_young']]),
    'materiau':     acier['materiau'] + aluminium['materiau'] + polymere['materiau']
})

# ============================================================
#  ENCODAGE + MODÈLE
# ============================================================
le = LabelEncoder()
df['label'] = le.fit_transform(df['materiau'])

X = df[['durete', 'densite', 'resistance', 'module_young']]
y = df['label']

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)

# ============================================================
#  SIDEBAR — PARAMÈTRES UTILISATEUR
# ============================================================
st.sidebar.header("🔧 Propriétés mécaniques du matériau")

durete       = st.sidebar.slider("Dureté (HB)",              5.0,  300.0, 100.0)
densite      = st.sidebar.slider("Densité (g/cm³)",          0.9,  8.0,   3.0)
resistance   = st.sidebar.slider("Résistance à la traction (MPa)", 10.0, 800.0, 200.0)
module_young = st.sidebar.slider("Module de Young (GPa)",    0.5,  220.0, 70.0)

# ============================================================
#  TABLEAU DES VALEURS SAISIES
# ============================================================
input_data = pd.DataFrame({
    'Dureté (HB)':           [durete],
    'Densité (g/cm³)':       [densite],
    'Résistance (MPa)':      [resistance],
    'Module de Young (GPa)': [module_young]
})

st.subheader("📋 Propriétés saisies")
st.write(input_data)

# ============================================================
#  PRÉDICTION
# ============================================================
prediction      = clf.predict(input_data.values)
prediction_proba = clf.predict_proba(input_data.values)

materiau_predit = le.inverse_transform(prediction)[0]

# Icône selon le matériau
icones = {'Acier': '🔩', 'Aluminium': '🪨', 'Polymère': '🧴'}
icone = icones.get(materiau_predit, '🧱')

st.subheader("🎯 Résultat de la classification")
st.success(f"{icone}  Le matériau est probablement un **{materiau_predit}**")

# ============================================================
#  PROBABILITÉS PAR CLASSE
# ============================================================
st.subheader("📊 Probabilités par classe")
proba_df = pd.DataFrame(
    prediction_proba,
    columns=le.classes_
)
st.bar_chart(proba_df.T)
