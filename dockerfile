FROM python:3.13

WORKDIR /app

# Instala dependências do sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copia os arquivos
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY infer.py .
COPY best.pt .

EXPOSE 5000
CMD ["python", "infer.py"]
