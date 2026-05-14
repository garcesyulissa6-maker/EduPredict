# app.py - Prediccion de Abandono Escolar y Exito Academico
# Modelos: Regresion Logistica y Red Neuronal Artificial (ANN)
# Dataset: Predict Students' Dropout and Academic Success - UCI ML Repository (ID 697)
# Ejecutar: python app.py

import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# ----- Configuracion -----
app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Nombres de columnas del dataset
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

# Clases del target
CLASS_NAMES = ['Dropout', 'Enrolled', 'Graduate']


# ----- Funciones para modelos -----

def cargar_pkl(nombre):
    """Carga un archivo .pkl si existe."""
    ruta = os.path.join(BASE_DIR, nombre)
    if os.path.exists(ruta):
        with open(ruta, 'rb') as f:
            return pickle.load(f)
    return None


def entrenar_modelos():
    """Entrena ambos modelos desde students.csv y los guarda como .pkl."""
    print("  Entrenando modelos desde students.csv ...")

    # Leer datos
    df = pd.read_csv(os.path.join(BASE_DIR, 'students.csv')).dropna()

    # Separar features y target
    X = df[FEATURE_COLS].values
    y_raw = df['Target'].values

    # Codificar target: Dropout=0, Enrolled=1, Graduate=2
    le = LabelEncoder()
    le.fit(CLASS_NAMES)  # Orden fijo
    y = le.transform(y_raw)

    # Dividir en train y test (80/20 como recomienda el dataset)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Normalizar datos
    sc = StandardScaler()
    X_train_sc = sc.fit_transform(X_train)
    X_test_sc = sc.transform(X_test)

    # Modelo 1: Regresion Logistica (multi-clase)
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_sc, y_train)
    y_pred_lr = lr.predict(X_test_sc)
    acc_lr = accuracy_score(y_test, y_pred_lr)
    print(f"  Regresion Logistica  Accuracy: {acc_lr*100:.2f}%")

    # Modelo 2: Red Neuronal ANN (multi-clase)
    ann = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    ann.fit(X_train_sc, y_train)
    y_pred_ann = ann.predict(X_test_sc)
    acc_ann = accuracy_score(y_test, y_pred_ann)
    print(f"  Red Neuronal (ANN)   Accuracy: {acc_ann*100:.2f}%")

    # Guardar modelos, scaler, encoder, y datos de test para metricas
    for nombre, objeto in [
        ('logreg_model.pkl', lr),
        ('ann_model.pkl', ann),
        ('scaler.pkl', sc),
        ('label_encoder.pkl', le),
        ('test_data.pkl', {
            'X_test': X_test_sc,
            'y_test': y_test
        })
    ]:
        with open(os.path.join(BASE_DIR, nombre), 'wb') as f:
            pickle.dump(objeto, f)

    print("  Modelos guardados correctamente\n")
    return lr, ann, sc, le, X_test_sc, y_test


# ----- Cargar o entrenar modelos al iniciar -----

# Borrar pkl viejos para reentrenar con misma normalizacion
for f in ['logreg_model.pkl', 'ann_model.pkl', 'scaler.pkl', 'label_encoder.pkl', 'test_data.pkl']:
    ruta = os.path.join(BASE_DIR, f)
    if os.path.exists(ruta):
        os.remove(ruta)

logreg_model, ann_model, scaler, label_encoder, X_test_global, y_test_global = entrenar_modelos()


# ----- Rutas -----

@app.route('/')
def index():
    """Pagina principal."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Prediccion individual de un estudiante."""
    data = request.get_json()
    modelo = data.get('model', 'lr')
    features = data.get('features', [])

    X = np.array(features, dtype=float).reshape(1, -1)
    X_sc = scaler.transform(X)

    if modelo == 'lr':
        pred = int(logreg_model.predict(X_sc)[0])
        probs = logreg_model.predict_proba(X_sc)[0].tolist()
    else:
        pred = int(ann_model.predict(X_sc)[0])
        probs = ann_model.predict_proba(X_sc)[0].tolist()

    return jsonify({
        'prediction': pred,
        'label': CLASS_NAMES[pred],
        'probabilities': {CLASS_NAMES[i]: round(p, 4) for i, p in enumerate(probs)}
    })


@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """Prediccion por lotes de varios estudiantes con metricas."""
    data = request.get_json()
    modelo = data.get('model', 'lr')
    rows = data.get('rows', [])
    true_labels = data.get('true_labels', None)

    X = np.array(rows, dtype=float)
    X_sc = scaler.transform(X)

    if modelo == 'lr':
        preds = logreg_model.predict(X_sc).tolist()
        probs = logreg_model.predict_proba(X_sc).tolist()
    else:
        preds = ann_model.predict(X_sc).tolist()
        probs = ann_model.predict_proba(X_sc).tolist()

    labels = [CLASS_NAMES[p] for p in preds]

    result = {
        'predictions': preds,
        'labels': labels,
        'probabilities': probs
    }

    # Si se proporcionan etiquetas reales, calcular metricas
    if true_labels is not None:
        y_true = np.array(true_labels, dtype=int)
        y_pred = np.array(preds, dtype=int)

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()
        report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
        acc = accuracy_score(y_true, y_pred)

        result['confusion_matrix'] = cm
        result['accuracy'] = round(acc, 4)
        result['metrics'] = {
            name: {
                'precision': round(report[name]['precision'], 4),
                'recall': round(report[name]['recall'], 4),
                'f1-score': round(report[name]['f1-score'], 4),
                'support': int(report[name]['support'])
            }
            for name in CLASS_NAMES if name in report
        }

    return jsonify(result)


@app.route('/metrics', methods=['POST'])
def metrics():
    """Devuelve metricas del modelo sobre los datos de test."""
    data = request.get_json()
    modelo = data.get('model', 'lr')

    if modelo == 'lr':
        y_pred = logreg_model.predict(X_test_global)
        probs = logreg_model.predict_proba(X_test_global)
    else:
        y_pred = ann_model.predict(X_test_global)
        probs = ann_model.predict_proba(X_test_global)

    cm = confusion_matrix(y_test_global, y_pred, labels=[0, 1, 2]).tolist()
    report = classification_report(y_test_global, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    acc = accuracy_score(y_test_global, y_pred)

    return jsonify({
        'confusion_matrix': cm,
        'accuracy': round(acc, 4),
        'metrics': {
            name: {
                'precision': round(report[name]['precision'], 4),
                'recall': round(report[name]['recall'], 4),
                'f1-score': round(report[name]['f1-score'], 4),
                'support': int(report[name]['support'])
            }
            for name in CLASS_NAMES if name in report
        },
        'class_names': CLASS_NAMES
    })


# ----- Iniciar servidor -----

if __name__ == '__main__':
    print("=" * 50)
    print("  EduPredict - Abandono Escolar y Exito Academico")
    print("=" * 50)
    print("  http://127.0.0.1:5000\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
