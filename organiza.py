import os
import shutil
import random
import cv2
import numpy as np

# Diretórios principais
data_dir = "data"
labels_dir = os.path.join(data_dir, "labels")

dataset_dir = "dataset"
images_train_dir = os.path.join(dataset_dir, "images", "train")
images_val_dir = os.path.join(dataset_dir, "images", "val")
labels_train_dir = os.path.join(dataset_dir, "labels", "train")
labels_val_dir = os.path.join(dataset_dir, "labels", "val")

os.makedirs(images_train_dir, exist_ok=True)
os.makedirs(images_val_dir, exist_ok=True)
os.makedirs(labels_train_dir, exist_ok=True)
os.makedirs(labels_val_dir, exist_ok=True)


def preprocessar_imagem_clahe_blur(caminho_pgm):
    imagem = cv2.imread(caminho_pgm, cv2.IMREAD_GRAYSCALE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    imagem_clahe = clahe.apply(imagem)
    imagem_blur = cv2.GaussianBlur(imagem_clahe, (3, 3), 0)
    imagem_norm = cv2.normalize(imagem_blur, None, 0, 255, cv2.NORM_MINMAX)
    return imagem_norm

def encontrar_mama_bounds(img_array, limiar=120):
    img_cortada = img_array[:, 60:]  
    mask = img_cortada > limiar
    coords = np.argwhere(mask)

    if coords.size == 0:
        return None

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)

    x0 += 60  # Corrige os offsets do corte
    x1 += 60

    return x0, y0, x1, y1  # xmin, ymin, xmax, ymax

imagens = [f for f in os.listdir(data_dir) if f.endswith(".pgm")]
random.seed(42)
random.shuffle(imagens)


limite_treino = int(len(imagens) * 0.8)
imagens_treino = imagens[:limite_treino]
imagens_val = imagens[limite_treino:]

def processar_dataset(imagens_lista, dir_imagens_saida, dir_labels_saida):
    for nome_img in imagens_lista:
        caminho_pgm = os.path.join(data_dir, nome_img)
        nome_base = os.path.splitext(nome_img)[0]
        caminho_png_saida = os.path.join(dir_imagens_saida, nome_base + ".png")

        imagem_processada = preprocessar_imagem_clahe_blur(caminho_pgm)

        bounds = encontrar_mama_bounds(imagem_processada)
        if bounds:
            x0, y0, x1, y1 = bounds
            imagem_processada = imagem_processada[y0:y1, x0:x1]  

        cv2.imwrite(caminho_png_saida, imagem_processada)

        label_origem = os.path.join(labels_dir, nome_base + ".txt")
        label_destino = os.path.join(dir_labels_saida, nome_base + ".txt")

        if os.path.exists(label_origem):
            shutil.copy(label_origem, label_destino)
            print(f"✅ {nome_base}: imagem e label processados.")
        else:
            with open(label_destino, "w") as f:
                pass
            print(f"⚠️ {nome_base}: label ausente — criado vazio.")

processar_dataset(imagens_treino, images_train_dir, labels_train_dir)
processar_dataset(imagens_val, images_val_dir, labels_val_dir)

print("\n✅ Dataset finalizado! Imagens melhoradas e salvas em pastas YOLO.")
