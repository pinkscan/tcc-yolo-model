from flask import Flask, request, render_template, send_from_directory
import os
import cv2
import numpy as np
import joblib

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pgm'}
MODEL_PATH = 'rf_model.joblib'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Carrega o modelo
model = joblib.load(MODEL_PATH)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(filepath, crop_size=64):
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    h, w = img.shape
    cx, cy = w // 2, h // 2
    half = crop_size // 2

    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(w, x1 + crop_size)
    y2 = min(h, y1 + crop_size)

    if (x2 - x1) < crop_size:
        x1 = max(0, x2 - crop_size)
    if (y2 - y1) < crop_size:
        y1 = max(0, y2 - crop_size)

    crop = img[y1:y2, x1:x2]

    if crop.shape[0] != crop_size or crop.shape[1] != crop_size:
        crop = cv2.resize(crop, (crop_size, crop_size))

    crop = crop.astype(np.float32) / 255.0

    return crop.flatten().reshape(1, -1)

def convert_to_png(pgm_path):
    img = cv2.imread(pgm_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    png_filename = os.path.splitext(os.path.basename(pgm_path))[0] + '.png'
    png_path = os.path.join(app.config['UPLOAD_FOLDER'], png_filename)
    cv2.imwrite(png_path, img)
    return png_filename

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    prob = None
    error = None
    png_filename = None

    if request.method == 'POST':
        if 'image' not in request.files:
            error = 'Nenhum arquivo enviado.'
            return render_template('index.html', error=error)

        file = request.files['image']
        if file.filename == '':
            error = 'Nenhum arquivo selecionado.'
            return render_template('index.html', error=error)

        if file and allowed_file(file.filename):
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            img_processed = preprocess_image(filepath)

            if img_processed is not None:
                probas = model.predict_proba(img_processed)[0]
                prediction = model.predict(img_processed)[0]
                prob = {
                    'cancer': f"{probas[1]*100:.2f}%",
                    'normal': f"{probas[0]*100:.2f}%"
                }

                # Converte para PNG para exibição
                png_filename = convert_to_png(filepath)
            else:
                error = 'Erro ao processar a imagem.'
        else:
            error = 'Formato de arquivo não suportado. Use arquivos .pgm.'

    return render_template('index.html', prediction=prediction, prob=prob, error=error, filename=png_filename)

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
