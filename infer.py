from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import io
import os
import uuid

app = Flask(__name__)
model = YOLO("best.pt")

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    temp_filename = f"/tmp/{uuid.uuid4()}.png"

    try:
        # Converte para PNG e salva temporariamente
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
        img.save(temp_filename)

        # Roda a predição
        results = model.predict(source=temp_filename, save=False, imgsz=640, conf=0.01)

        detections = []
        cancer_detectado = False

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                classe_nome = model.names[cls]
                detections.append({
                    "classe": classe_nome,
                    "confianca": round(conf, 3)
                })
                # Verifica se é a classe "cancer" 
                if classe_nome.lower() == "cancer":
                    cancer_detectado = True

        return jsonify({
            "classe": model.names,
            "tem_cancer": cancer_detectado,
            "deteccoes": detections
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
