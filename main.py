import os
import re
import fitz  # PyMuPDF
import cv2
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import resample

# === CONFIG ===
DATA_DIR = 'data/'
PDF_FILE = os.path.join(DATA_DIR, '00README.pdf')
MODEL_FILE = 'rf_model.joblib'
CROP_SIZE = 64

# === 1. Extrai labels do PDF ===
def processar_informacoes_completas(linhas):
    dados = []
    padrao_nome = re.compile(r"^mdb\d{3}[lr][lm]$")
    padrao_info = re.compile(r"^([FGD])\s+([A-Z]+)\s+([BM])\s+(\d+)\s+(\d+)\s+(\d+)$")
    padrao_norm = re.compile(r"^([FGD])\s+NORM$")

    i = 0
    while i < len(linhas):
        linha = linhas[i].strip()
        if padrao_nome.match(linha):
            nome_arquivo = linha
            if i + 1 < len(linhas):
                proxima = linhas[i + 1].strip()
                match = padrao_info.match(proxima)
                if match:
                    dados.append({
                        "nome_arquivo": nome_arquivo,
                        "tem_cancer": True,
                        "tecido": match.group(1),
                        "classe": match.group(2),
                        "severidade": match.group(3),
                        "x": int(match.group(4)),
                        "y": int(match.group(5)),
                        "raio": int(match.group(6))
                    })
                elif padrao_norm.match(proxima):
                    dados.append({
                        "nome_arquivo": nome_arquivo,
                        "tem_cancer": False
                    })
            i += 1
        i += 1
    return dados

# === 2. Balanceamento do Dataset ===
def balancear_dataset(df):
    df_majority = df[df.label == 0]
    df_minority = df[df.label == 1]

    df_majority_downsampled = resample(df_majority,
                                       replace=False,
                                       n_samples=len(df_minority),
                                       random_state=42)
    df_balanced = pd.concat([df_majority_downsampled, df_minority])
    return df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# === 3. Funções de recorte ===
def crop_around_point(img, x, y, size):
    h, w = img.shape
    ch, cw = size // 2, size // 2
    x1 = max(0, x - cw)
    y1 = max(0, y - ch)
    x2 = min(w, x1 + size)
    y2 = min(h, y1 + size)
    
    # Ajuste se o crop ficou menor que o tamanho desejado (ex: perto das bordas)
    if (x2 - x1) < size:
        x1 = max(0, x2 - size)
    if (y2 - y1) < size:
        y1 = max(0, y2 - size)
    
    crop = img[y1:y2, x1:x2]
    return cv2.resize(crop, (size, size))

def crop_random(img, size):
    h, w = img.shape
    ch, cw = size // 2, size // 2
    if w - size <= 0 or h - size <= 0:
        # Se a imagem for menor que o crop, redimensiona direto
        return cv2.resize(img, (size, size))
    x = np.random.randint(cw, w - cw)
    y = np.random.randint(ch, h - ch)
    return crop_around_point(img, x, y, size)

# === 4. Carregamento e processamento das imagens ===
def load_and_process_images(df, img_dir, crop_size=64):
    X, y = [], []

    for _, row in df.iterrows():
        img_path = os.path.join(img_dir, row['filename'])
        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        if row['label'] == 1 and pd.notna(row['x']) and pd.notna(row['y']):
            crop = crop_around_point(img, int(row['x']), int(row['y']), crop_size)
        else:
            crop = crop_random(img, crop_size)

        X.append(crop.flatten())
        y.append(row['label'])

    print(f"✅ Total de imagens processadas: {len(X)}")
    return np.array(X), np.array(y)

# === 5. Treinamento com Random Forest + GridSearch ===
def train_rf_gridsearch(X, y):
    X = X / 255.0  # normalização
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("🔍 Buscando melhores hiperparâmetros com GridSearchCV...")

    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
    }

    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy', verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    print("\n🏆 Melhor modelo:")
    print(grid_search.best_params_)

    y_pred = best_model.predict(X_test)

    print("\n=== Avaliação do modelo ===")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    joblib.dump(best_model, MODEL_FILE)
    print(f"💾 Modelo salvo em {MODEL_FILE}")

    return best_model

# === 6. Pipeline principal ===
if __name__ == '__main__':
    print("📖 Lendo PDF...")
    with fitz.open(PDF_FILE) as pdf:
        linhas = []
        for pagina in pdf:
            linhas.extend(pagina.get_text().split("\n"))

    dados = processar_informacoes_completas(linhas)

    df = pd.DataFrame([
        {
            'filename': d['nome_arquivo'][:10] + '.pgm',
            'label': int(d['tem_cancer']),
            'x': d.get('x'),
            'y': d.get('y'),
            'radius': d.get('raio')
        }
        for d in dados
    ])

    print(f"📊 Total de amostras extraídas: {len(df)}")

    df = balancear_dataset(df)
    print(df['label'].value_counts())

    print("🖼️ Processando imagens...")
    X, y = load_and_process_images(df, DATA_DIR, CROP_SIZE)

    if len(X) == 0:
        print("❌ Nenhuma imagem válida foi processada.")
    else:
        print("🤖 Treinando modelo Random Forest com GridSearch...")
        model = train_rf_gridsearch(X, y)
