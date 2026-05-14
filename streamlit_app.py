import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="EduPredict", page_icon="🎓", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FEATURE_COLS = [
    'Marital Status', 'Application mode', 'Application order', 'Course',
    'Daytime/evening attendance', 'Previous qualification',
    'Previous qualification (grade)', 'Nacionality',
    "Mother's qualification", "Father's qualification",
    "Mother's occupation", "Father's occupation",
    'Admission grade', 'Displaced', 'Educational special needs',
    'Debtor', 'Tuition fees up to date', 'Gender', 'Scholarship holder',
    'Age at enrollment', 'International',
    'Curricular units 1st sem (credited)',
    'Curricular units 1st sem (enrolled)',
    'Curricular units 1st sem (evaluations)',
    'Curricular units 1st sem (approved)',
    'Curricular units 1st sem (grade)',
    'Curricular units 1st sem (without evaluations)',
    'Curricular units 2nd sem (credited)',
    'Curricular units 2nd sem (enrolled)',
    'Curricular units 2nd sem (evaluations)',
    'Curricular units 2nd sem (approved)',
    'Curricular units 2nd sem (grade)',
    'Curricular units 2nd sem (without evaluations)',
    'Unemployment rate', 'Inflation rate', 'GDP'
]

CLASS_NAMES = ['Dropout', 'Enrolled', 'Graduate']

@st.cache_resource
def load_models():
    models = {}
    for name in ['logreg_model.pkl', 'ann_model.pkl', 'scaler.pkl', 'label_encoder.pkl', 'test_data.pkl']:
        ruta = os.path.join(BASE_DIR, name)
        if os.path.exists(ruta):
            with open(ruta, 'rb') as f:
                models[name] = pickle.load(f)
    return models

models = load_models()

if not models or 'logreg_model.pkl' not in models:
    st.error("No se encontraron los modelos entrenados. Por favor entrena los modelos ejecutando `app.py` primero.")
    st.stop()

logreg_model = models['logreg_model.pkl']
ann_model = models['ann_model.pkl']
scaler = models['scaler.pkl']
test_data = models['test_data.pkl']

# Sidebar
st.sidebar.title("🎓 EduPredict")
st.sidebar.write("Predicción de Abandono Escolar y Éxito Académico")
app_mode = st.sidebar.radio("Modo", ["Predicción por Lotes", "Métricas en Test"])

modelo_seleccionado = st.sidebar.selectbox("Seleccionar Modelo", ["Regresión Logística", "Red Neuronal (ANN)"])
model_to_use = logreg_model if modelo_seleccionado == "Regresión Logística" else ann_model

st.title(f"EduPredict - {app_mode}")

if app_mode == "Métricas en Test":
    st.header("📊 Rendimiento del Modelo en Conjunto de Prueba")
    st.write("Métricas calculadas sobre el 20% del dataset reservado para pruebas.")
    
    X_test = test_data['X_test']
    y_test = test_data['y_test']
    
    y_pred = model_to_use.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.metric("Exactitud (Accuracy)", f"{acc*100:.2f}%")
    
    st.subheader("Matriz de Confusión")
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    st.pyplot(fig)
    
    st.subheader("Reporte de Clasificación")
    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    df_report = pd.DataFrame(report).transpose()
    st.dataframe(df_report.style.format("{:.4f}"))

elif app_mode == "Predicción por Lotes":
    st.header("📂 Predicción por Lotes (CSV)")
    uploaded_file = st.file_uploader("Sube un archivo CSV con las características de los estudiantes", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        # Filtrar solo columnas relevantes
        missing_cols = [col for col in FEATURE_COLS if col not in df.columns]
        if missing_cols:
            st.error(f"El archivo no contiene todas las columnas requeridas. Faltan: {missing_cols}")
        else:
            st.success(f"Archivo cargado correctamente. Filas: {len(df)}")
            
            X_raw = df[FEATURE_COLS].values
            X_sc = scaler.transform(X_raw)
            
            preds = model_to_use.predict(X_sc)
            probs = model_to_use.predict_proba(X_sc)
            
            df_results = df.copy()
            df_results['Prediction_Class'] = [CLASS_NAMES[int(p)] for p in preds]
            for i, name in enumerate(CLASS_NAMES):
                df_results[f'Prob_{name}'] = probs[:, i].round(4)
                
            st.subheader("Resultados de Predicción")
            st.dataframe(df_results)
            
            # Descargar resultados
            csv = df_results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Descargar Resultados CSV",
                data=csv,
                file_name='resultados_prediccion.csv',
                mime='text/csv',
            )
            
            # Si el CSV tiene columna Target real, calcular metricas
            if 'Target' in df.columns:
                st.subheader("Métricas de Evaluación contra el Target Real")
                le = models['label_encoder.pkl']
                y_true = le.transform(df['Target'])
                acc = accuracy_score(y_true, preds)
                st.metric("Exactitud (Accuracy)", f"{acc*100:.2f}%")
                
                cm = confusion_matrix(y_true, preds, labels=[0, 1, 2])
                fig, ax = plt.subplots(figsize=(6,4))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
                plt.xlabel('Predicción')
                plt.ylabel('Real')
                st.pyplot(fig)
