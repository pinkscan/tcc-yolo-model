from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import io

app = Flask(__name__)
model = YOLO("best.pt")

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    img = Image.open(io.BytesIO(file.read()))

    results = model(img)

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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
