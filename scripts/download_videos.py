"""Descarga unos pocos vídeos del dataset Nexar Collision Prediction (Hugging Face).

Requisitos:
  - Haber pedido acceso al dataset en la web de Hugging Face (es "gated").
  - Haber hecho `hf auth login` una vez: el token se guarda en ~/.cache/huggingface/
    y la librería lo encuentra sola, así que nunca aparece en el código.

Uso (desde la raíz del proyecto, con el .venv activado):
  python scripts/download_videos.py
"""

from huggingface_hub import hf_hub_download

REPO_ID = "nexar-ai/nexar_collision_prediction"

# Bajamos solo unos pocos ficheros sueltos en vez del dataset entero (~1500 vídeos).
# negative = conducción normal, positive = colisión o casi-colisión.
# Los metadata.csv traen las etiquetas (time_of_event, time_of_alert, escena...),
# que usaremos en la Fase 8 para evaluar si los avisos llegan a tiempo.
# OJO: el campo light_conditions del CSV no distingue día/noche (vídeos de noche
# vienen como "Normal"), así que los de día los elegimos midiendo el brillo de los frames.
FILES = [
    # Día
    "train/negative/01046.mp4",  # ciudad, cielo despejado
    "train/negative/01074.mp4",  # calle suburbana
    "train/negative/01050.mp4",  # autopista con túnel (cambio brusco de luz)
    "train/positive/00021.mp4",  # ciudad, tráfico delante
    "train/positive/00015.mp4",  # autopista, contraluz fuerte
    # Noche (para probar los casos difíciles)
    "train/negative/01040.mp4",
    "train/negative/01042.mp4",
    "train/positive/00000.mp4",
    "train/positive/00003.mp4",
    # Etiquetas
    "train/negative/metadata.csv",
    "train/positive/metadata.csv",
]

for filename in FILES:
    # repo_type="dataset" es obligatorio: por defecto busca en *modelos* y daría un 404.
    # local_dir mantiene las subcarpetas -> data/videos/train/positive/00000.mp4
    # Si el fichero ya está descargado y no ha cambiado, no lo vuelve a bajar.
    path = hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        repo_type="dataset",
        local_dir="data/videos",
    )
    print(f"OK  {path}")
