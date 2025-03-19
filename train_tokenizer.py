from datasets import load_dataset
from transformers import GPT2TokenizerFast  # Establece la configuración del tokenizador de GPT2
from tqdm import tqdm
from itertools import islice
import os

access_token = "" # Introduce tu token de acceso
model = "openai-community/gpt2-large"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
old_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2-large", token = access_token)
catlog = load_dataset("projecte-aina/CATalog", token=access_token, trust_remote_code=True, streaming=True, split='train')
dacsa_train = load_dataset("ELiRF/dacsa", "catalan", token=access_token, trust_remote_code=True, streaming=True, split='train')

# Renombrar columna para consistencia
dacsa_train = dacsa_train.rename_column("article", "text")

# Asignar datasets a variables para entrenamiento y prueba
ds = catlog.filter(lambda example: example["text"] is not None)
ds_test = dacsa_train.filter(lambda example: example["text"] is not None)

len_ds = ds.info.splits['train'].num_examples
len_ds_test = ds_test.info.splits['train'].num_examples

def batch_iterator(dataset, batch_size=50000):
    """
    Iterador que devuelve batches de textos extraídos del dataset.
    """
    iterator = iter(dataset)
    pbar = tqdm(total=len_ds, desc="Procesando dataset")

    while True:
        batch = list(islice(iterator, batch_size))
        if not batch:
            break
        pbar.update(len(batch))
        # Extraer la columna "text" de cada muestra del batch
        yield [sample["text"] for sample in batch]

    pbar.close()

# Configuración
VOCAB_SIZES = [52000]
SAMPLE_SIZE = 2000

for vocab_size in VOCAB_SIZES:
    tokenizer = old_tokenizer.train_new_from_iterator(batch_iterator(ds), vocab_size=vocab_size)

    tokenizer.save_pretrained(f"gpt2-catalan-{vocab_size}")

    # Obtenemos el vocab
    vocab = set(tokenizer.get_vocab().keys())

    texts = [example["text"] for example in islice(ds_test, SAMPLE_SIZE)]
    batch_size = max(1, SAMPLE_SIZE // 4)  # Se fija en 4 para dividir en partes más manejables
    text_batches = [texts[i:i + batch_size] for i in range(0, SAMPLE_SIZE, batch_size)]

    results = []
    for batch in text_batches:
        encodings = tokenizer.encode_batch(batch)
        total_words = sum(len(text.split()) for text in batch)
        total_tokens = sum(len(enc.tokens) for enc in encodings)
        results.append((total_words, total_tokens))

    total_words = sum(result[0] for result in results)
    total_tokens = sum(result[1] for result in results)

    avg_token_length = (total_tokens / total_words) if total_words > 0 else 0

    print(f"✅ vocab_size={vocab_size} | Avg Token Length={avg_token_length:.2f}")
    # Guardar resultados en un JSON
    with open("results.json", "a") as f:
        f.write(f'{{"vocab_size": {vocab_size}, "avg_token_length": {avg_token_length:.2f}}}\n')