from ultralytics import YOLO
from PIL import Image
import os

# Caminho da imagem PGM
imagem_pgm = 'mdb001lm.pgm'

# Converte .pgm para .png
imagem_png = imagem_pgm.replace('.pgm', '.png')
Image.open(imagem_pgm).save(imagem_png)

# Carrega o modelo YOLO
model = YOLO('runs/detect/train/weights/best.pt')

# Faz a predição na imagem PNG
resultados = model.predict(source=imagem_png, save=True, imgsz=640, conf=0.1)

# Exibe o resultado
resultados[0].show()

# (Opcional) Remove o arquivo PNG convertido se quiser limpar
# os.remove(imagem_png)
