from datasets import load_dataset, concatenate_datasets, Dataset
import ctranslate2
import pyonmttok
from huggingface_hub import snapshot_download
import json
import random

# Cargar datasets
ds1 = load_dataset("UCSC-VLAA/Recap-COCO-30K")
ds2 = load_dataset("yerevann/coco-karpathy")

# Concatenar datasets
t = ds2['train']
v = ds2['validation']
te = ds2['test']
rv = ds2['restval']

concatenated_dataset = concatenate_datasets([t, v, te, rv])

# Procesar captions e ids
c1 = ds1['train']['caption']
ids1 = [f'{id}_30k' for id in ds1['train']['image_id']]

c2 = []
for caps in concatenated_dataset['sentences']:
    g = random.choice(caps)
    c2.append(g)

ids2 = [f'{id}_karpa' for id in concatenated_dataset['imgid']]

c1.extend(c2)
ids1.extend(ids2)

dataset = Dataset.from_dict({'captions': c1, 
                            'image_id': ids1})

# Descargar el modelo
model_dir = snapshot_download(repo_id="softcatala/translate-eng-cat", revision="main")

# Tokenizador
tokenizer = pyonmttok.Tokenizer(mode="none", sp_model_path=model_dir + "/sp_m.model")

# Traductor configurado para usar la GPU 1
translator = ctranslate2.Translator(model_dir, device="cuda", device_index=1)

# Función de traducción
def translate(text):
    tokenized = tokenizer.tokenize(text)
    translated = translator.translate_batch([tokenized[0]])

    cat = tokenizer.detokenize(translated[0][0]['tokens'])

    return cat

# Traducir captions y guardar los resultados
translated = {}
for id, caption in zip(dataset['image_id'], dataset['captions']):
    translated[id] = translate(caption)

# Guardar las traducciones en un archivo JSON
nombre_archivo = 'translate.json'
with open(nombre_archivo, 'w', encoding='utf-8') as file:
    json.dump(translated, file, ensure_ascii=False, indent=4)

print(f"Datos guardados en {nombre_archivo}")
