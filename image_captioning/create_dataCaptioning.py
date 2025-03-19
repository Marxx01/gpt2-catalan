import json
from datasets import load_dataset, concatenate_datasets
import pandas as pd
import requests
from PIL import Image
import io
import os

access_token = ""

# Cargar datasets
ds1 = load_dataset("UCSC-VLAA/Recap-COCO-30K", split='train')
ds1 = ds1.remove_columns([col for col in ds1.column_names if col not in ['image_id', 'coco_url']])

ds2_parts = load_dataset("yerevann/coco-karpathy")
ds2 = concatenate_datasets([ds2_parts[split] for split in ['train', 'validation', 'test', 'restval']])
ds2 = ds2.remove_columns([col for col in ds2.column_names if col not in ['imgid', 'url']])

# Convertir datasets a DataFrame
df_30 = ds1.to_pandas().rename(columns={'image_id': 'imgid', 'coco_url': 'url'})
df_ka = ds2.to_pandas()

# Cargar datos de traducción
with open('translate.json', 'r') as archivo:
    datos = json.load(archivo)

# Función para obtener la URL de la imagen
def get_image_url(image_id):
    id, split = image_id.split('_')
    dataset = df_30 if split == '30k' else df_ka
    return dataset.loc[dataset['imgid'] == int(id), 'url'].values[0]

# Preparar datos
data = pd.DataFrame({
    'ids': list(datos.keys()),
    'captions': ['La image mostra: ' + cap for cap in datos.values()],
    'urls': [get_image_url(id) for id in datos.keys()]
})

# Descargar y procesar imágenes
def process_image(url):
    try:
        respuesta = requests.get(url)
        respuesta.raise_for_status()
        img_data = respuesta.content
    except requests.RequestException:
        print(f"Error al descargar la imagen de {url}")
        img_data = None

    # Guardar y procesar imagen
    if img_data:
        img = Image.open(io.BytesIO(img_data)).resize((384, 384))
    else:
        caption = 'La image mostra: Un home donant menjar a un gos blanc a un pati.'
        data.loc[data['urls'] == url, 'captions'] = caption
        img = Image.open("./imagen_fallo.jpg").resize((384, 384))
    
    byte_arr = io.BytesIO()
    img.save(byte_arr, format="JPEG", quality=100)
    return byte_arr.getvalue()

# Aplicar procesamiento a todas las imágenes
data['images'] = [
    process_image(url) for url in data['urls']
]

# Guardar resultados
data.to_csv('imageCaptioning.csv', index=False)
data.to_parquet('imageCaptioning.parquet', index=False)