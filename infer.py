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
    ext = file.filename.lower().split('.')[-1]

    # Cria nome único temporário
    temp_filename = f"/tmp/{uuid.uuid4()}.png"

    try:
        # Converte .pgm para .png (ou qualquer imagem para RGB)
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
        img.save(temp_filename)

        # Predição com o caminho salvo
        results = model.predict(source=temp_filename, save=False, imgsz=640, conf=0.1)

        output = []
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                output.append({
                    "classe": model.names[cls],
                    "confianca": round(conf, 3)
                })

        return jsonify(output)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        # Limpa o arquivo temporário, se existir
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
