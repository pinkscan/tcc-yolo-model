from ultralytics import YOLO

# Caminho da imagem PNG
imagem_png = 'mdb001lm.png'

# Carrega o modelo YOLO treinado
model = YOLO('runs/detect/train/weights/best.pt')

# Faz a predição na imagem PNG
resultados = model.predict(source=imagem_png, save=True, imgsz=640, conf=0.1)

# Exibe o resultado
resultados[0].show()
