import fitz
import re
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def extrair_info_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    linhas = []
    for pagina in doc:
        texto = pagina.get_text("text")
        linhas.extend(texto.split("\n"))
    return linhas

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

def encontrar_mama_bounds(img_array, limiar=200):
    # Corta os primeiros 60 pixels da esquerda (onde está a tarja "R ML")
    img_cortada = img_array[:, 60:]

    mask = img_cortada > limiar
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)

    # Corrige x0 e x1 por causa do corte da esquerda
    x0 += 60
    x1 += 60

    return x0, y0, x1, y1  # xmin, ymin, xmax, ymax


def converter_para_yolo(x, y, raio, largura, altura):
    x_center = x / largura
    y_center = y / altura
    width = (2 * raio) / largura
    height = (2 * raio) / altura
    return x_center, y_center, width, height

def gerar_labels_yolo(dados, diretorio_imagens, diretorio_labels):
    os.makedirs(diretorio_labels, exist_ok=True)
    os.makedirs("bboxes_preview", exist_ok=True)
    os.makedirs("debug_bounds", exist_ok=True)  # <- cria pasta de debug dos bounds

    for d in dados:
        nome_arquivo = d["nome_arquivo"]
        caminho_imagem = os.path.join(diretorio_imagens, nome_arquivo + ".pgm")
        caminho_txt = os.path.join(diretorio_labels, nome_arquivo + ".txt")

        if not os.path.exists(caminho_imagem):
            print(f"⚠️ Imagem não encontrada: {caminho_imagem}")
            continue

        with Image.open(caminho_imagem) as img:
            img_gray = img.convert("L")
            img_array = np.array(img_gray)

            mama_bounds = encontrar_mama_bounds(img_array)
            if mama_bounds is None:
                print(f"⚠️ Mama não detectada: {nome_arquivo}")
                continue

            # === DEBUG: Salva imagem com bounding box da mama ===
            x0, y0, x1, y1 = mama_bounds
            fig, ax = plt.subplots()
            ax.imshow(img_array, cmap='gray')
            rect = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=2, edgecolor='green', facecolor='none')
            ax.add_patch(rect)
            plt.title(f'Bounds encontrados: {nome_arquivo}')
            plt.axis('off')
            plt.savefig(f'debug_bounds/{nome_arquivo}_bounds.png', bbox_inches='tight')
            plt.close()

            # === Continua com geração de labels ===
            mama_crop = img_array[y0:y1, x0:x1]
            largura_crop = x1 - x0
            altura_crop = y1 - y0

            if not d.get("tem_cancer", False):
                with open(caminho_txt, "w") as f:
                    pass
                print(f"✅ Label vazio (sem câncer): {caminho_txt}")
                continue

            # Ajustar coordenadas
            x_orig = d["x"]
            y_orig = d["y"]
            raio = d["raio"]
            x = x_orig - x0
            y = y_orig - y0

            # Verifica se ainda está dentro do recorte
            if not (0 <= x < largura_crop and 0 <= y < altura_crop):
                print(f"⚠️ Anomalia fora da área recortada da mama: {nome_arquivo}")
                continue

            # YOLO
            x_yolo, y_yolo, w_yolo, h_yolo = converter_para_yolo(x, y, raio, largura_crop, altura_crop)
            with open(caminho_txt, "w") as f:
                f.write(f"0 {x_yolo:.6f} {y_yolo:.6f} {w_yolo:.6f} {h_yolo:.6f}\n")
            print(f"✅ Label com câncer gerado: {caminho_txt}")

            # Salva preview com bounding box da lesão
            fig, ax = plt.subplots()
            ax.imshow(mama_crop, cmap='gray')

            # Limita coordenadas da caixa para não sair da imagem
            x_min = max(x - raio, 0)
            y_min = max(y - raio, 0)
            x_max = min(x + raio, largura_crop - 1)
            y_max = min(y + raio, altura_crop - 1)

            width_box = x_max - x_min
            height_box = y_max - y_min

            rect = patches.Rectangle(
                (x_min, y_min), width_box, height_box,
                linewidth=1, edgecolor='r', facecolor='none'
            )
            ax.add_patch(rect)
            plt.title(nome_arquivo)
            plt.axis('off')
            plt.savefig(f"bboxes_preview/{nome_arquivo}.png", bbox_inches='tight')
            plt.close()

# === EXECUÇÃO ===
pdf_path = "00README.pdf"
diretorio_imagens = "data"
diretorio_labels = "data/labels"

linhas = extrair_info_pdf(pdf_path)
dados = processar_informacoes_completas(linhas)
gerar_labels_yolo(dados, diretorio_imagens, diretorio_labels)
