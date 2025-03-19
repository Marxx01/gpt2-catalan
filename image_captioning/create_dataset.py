import json
from datasets import load_dataset, concatenate_datasets
import pandas as pd
import requests
from PIL import Image
import io

ds1 = load_dataset("UCSC-VLAA/Recap-COCO-30K")
ds2 = load_dataset("yerevann/coco-karpathy")

with open('translate.json', 'r') as archivo:
    datos = json.load(archivo)

ds1_train = ds1['train']
t = ds2['train']
v = ds2['validation']
te = ds2['test']
rv = ds2['restval']

concatenated_dataset = concatenate_datasets([t, v, te, rv])

ds1 = ds1_train.remove_columns([col for col in ds1_train.column_names if col not in ['image_id', 'coco_url']])

df_30 = ds1.to_pandas()

df_30 = df_30.rename(columns={'image_id': 'imgid', 'coco_url': 'url'})

ds2 = concatenated_dataset.remove_columns([col for col in concatenated_dataset.column_names if col not in ['imgid', 'url']])

df_ka = ds2.to_pandas()

image_id = '53120_30k'

def get_image_url(image_id):
    id, split = image_id.split('_')
    if split == '30k':
        dataset = df_30
    else:
        dataset = df_ka

    url = dataset.loc[dataset['imgid'] == int(id), 'url'].values[0]

    return url

get_image_url(image_id)

l = len(datos)

data = {
    'ids': [None] * l,
    'captions': [None] * l,
    'urls': [None] * l
}

for i, (id, cap) in enumerate(datos.items()):
    data['ids'][i] = id
    data['captions'][i] = cap

    url = get_image_url(id)
    data['urls'][i] = url

data_df = pd.DataFrame(data)

def join_text(text):
    return 'La image mostra: ' + text

data_df["captions"] = data_df["captions"].apply(join_text)

image_list = []

for i, url in enumerate(data_df['urls']):
    respuesta = requests.get(url)
    if respuesta.status_code == 200:
        with open('./imagenes_auxiliares/imagen_descargada.jpg', 'wb') as archivo:
            archivo.write(respuesta.content)
            img = Image.open("./imagenes_auxiliares/imagen_descargada.jpg")
            img = img.resize((384, 384))
            byte_arr = io.BytesIO()
            img.save(byte_arr, format="JPEG", quality=100)  # Ajusta la calidad para reducir tamaño
            bytes_data = byte_arr.getvalue()
            image_list.append(bytes_data)
            
        #print("Imagen descargada y guardada con éxito.")
        
    else:
        print(f"Error al descargar la imagen: {data_df['ids'][i]}. Código de respuesta:", respuesta.status_code)
        with open('./imagenes_auxiliares/imagen_fallo.jpg', 'wb') as archivo:
            archivo.write(respuesta.content)
            img = Image.open("./imagenes_auxiliares/imagen_fallo.jpg")
            img = img.resize((384, 384))
            byte_arr = io.BytesIO()
            img.save(byte_arr, format="JPEG", quality=100)  # Ajusta la calidad para reducir tamaño
            bytes_data = byte_arr.getvalue()
            image_list.append(bytes_data)
            caption = 'La image mostra: Un home donant menjar a un gos blanc a un pati.'
            data_df.loc[i, 'captions'] = caption

data_df['images'] = image_list

data_df.to_csv('imageCaptioning.csv', index=False)
data_df.to_parquet("imageCaptioning.parquet", index=False)