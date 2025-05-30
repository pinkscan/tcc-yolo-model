from flask import Flask, request, jsonify, send_file
from ultralytics import YOLO
from PIL import Image
import io
import os
import uuid
import cv2
import numpy as np
import base64

app = Flask(__name__)
model = YOLO("best.pt")  # seu modelo YOLO treinado

def preprocessar_imagem_clahe_blur(img_pil):
    img_gray = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    imagem_clahe = clahe.apply(img_gray)
    imagem_blur = cv2.GaussianBlur(imagem_clahe, (3, 3), 0)
    return imagem_blur

def encontrar_mama_bounds(img_array, limiar=120):
    img_cortada = img_array[:, 60:]
    mask = img_cortada > limiar
    coords = np.argwhere(mask)

    if coords.size == 0:
        return None

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    x0 += 60
    x1 += 60
    return x0, y0, x1, y1

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/analisar", methods=["POST"])
def analisar():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    try:
        # Converte e pré-processa a imagem
        img_pil = Image.open(io.BytesIO(file.read())).convert("RGB")
        imagem_processada = preprocessar_imagem_clahe_blur(img_pil)

        bounds = encontrar_mama_bounds(imagem_processada)
        if bounds:
            x0, y0, x1, y1 = bounds
            imagem_processada = imagem_processada[y0:y1, x0:x1]

        temp_img_path = f"/tmp/{uuid.uuid4()}.png"
        cv2.imwrite(temp_img_path, imagem_processada)

        # YOLO detecta
        results = model.predict(source=temp_img_path, save=False, conf=0.1)
        detections = []
        cancer_detectado = False

        # Carrega imagem colorida para desenhar
        img_color = cv2.cvtColor(imagem_processada, cv2.COLOR_GRAY2RGB)

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                classe_nome = model.names[cls]
                detections.append({"classe": classe_nome, "confianca": round(conf, 3)})
                if classe_nome.lower() == "cancer":
                    cancer_detectado = True

                # Desenha caixa
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(img_color, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(img_color, f"{classe_nome} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Converte imagem para base64
        _, buffer = cv2.imencode('.png', img_color)
        img_b64 = base64.b64encode(buffer).decode("utf-8")

        return jsonify({
            "tem_cancer": cancer_detectado,
            "deteccoes": detections,
            "imagem_base64": img_b64
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
