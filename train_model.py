import numpy as np
from tqdm import tqdm
import os
import math
import torch
import transformers
from transformers import GPT2LMHeadModel, GPT2Config, AutoTokenizer, AdamW
from datasets import load_dataset
from torch.utils.data import IterableDataset, DataLoader
import wandb

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


class HuggingFaceIterableDataset(IterableDataset):
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset

    def __iter__(self):
        for item in self.dataset:
            yield {
                'input_ids': item['input_ids'].squeeze(0),  # Quitar la dimensión extra
                'attention_mask': item['attention_mask'].squeeze(0),
                'labels': item['labels'].squeeze(0)
            }

def tokenize_function(batch):
    # Tokenizar el texto del artículo con truncamiento y padding hasta un máximo de 1024 tokens
    tokenized_inputs = tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=1024,
        return_tensors="pt"  # Devuelve los resultados como tensores de PyTorch
    )

    # Crear una copia de los input_ids para usarlos como etiquetas (labels)
    labels = tokenized_inputs["input_ids"].clone()

    # Reemplazar los tokens de padding en las etiquetas con -100 para que sean ignorados durante el cálculo de la pérdida
    labels[labels == tokenizer.pad_token] = -100

    # Añadir las etiquetas al diccionario de entradas tokenizadas
    tokenized_inputs["labels"] = labels

    return tokenized_inputs


def compute_metrics(eval_pred: transformers.EvalPrediction):
    loss = eval_pred.loss
    perplexity = math.exp(loss)
    return {"perplexity": perplexity}

def train_one(model: torch.nn.Module, loader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer):
    """Standard PyTorch training, one epoch"""
    model.train()
    losses = []
    for batch in tqdm(loader, desc="Training"):
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)

        optimizer.zero_grad()
        out = model(input_ids = input_ids, attention_mask = attention_mask, labels = labels)
        # loss, logits, past_key_values
        loss = out['loss']
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return np.mean(losses)

def val_one(model: torch.nn.Module, loader: torch.utils.data.DataLoader):
    """Standard PyTorch eval, one epoch"""
    model.eval()
    losses = []
    for batch in tqdm(loader, desc="Validation"):
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)

        with torch.no_grad():
            out = model(input_ids = input_ids, attention_mask = attention_mask, labels = labels)
        # loss, logits, past_key_values
        loss = out['loss']
        losses.append(loss.item())

    return np.mean(losses)

def train_torch(early_stopping_patience=3, early_stopping_threshold=0.005, epochs: int=10, batch_size = 4):
    # Inicializar WandB
    wandb.init(project="gpt2-catalan-model", name="training_run", config={
        "batch_size": batch_size,
        "epochs": epochs,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_threshold": early_stopping_threshold,
    })

    best_loss = 1e9  # Iniciar con un valor de pérdida muy alto
    patience_counter = 0  # Contador para el early stopping
    loader_train = DataLoader(pytorch_dataset, batch_size = batch_size)
    loader_val = DataLoader(pytorch_dataset_val, batch_size = batch_size)

    save_dir = './gpt2-catalan-model'
    os.makedirs(save_dir, exist_ok = True)
    # Training loop
    for i_epoch in range(epochs):
        loss_train = train_one(model, loader_train, optimizer)
        loss_val = val_one(model, loader_val)

        # Calcular perplejidad
        perplexity_train = math.exp(loss_train) if loss_train < 100 else float('inf')  # Evitar overflows en exp()
        perplexity_val = math.exp(loss_val) if loss_val < 100 else float('inf')

        print(f'Epoch {i_epoch} : loss_train={loss_train}, loss_val={loss_val}')
        print(f'Perplexity train: {perplexity_train}, Perplexity val: {perplexity_val}')

        # Registrar métricas en WandB
        wandb.log({
            "loss_train": loss_train,
            "loss_val": loss_val,
            "perplexity_train": perplexity_train,
            "perplexity_val": perplexity_val,
            "epoch": i_epoch
        })

        # Check if the validation loss improved
        if loss_val < best_loss - early_stopping_threshold:
            best_loss = loss_val
            # Guardar el modelo
            model.save_pretrained(save_dir)
            print(f'Validation loss improved. Model saved. New best loss: {best_loss}')
            patience_counter = 0  # Reset the patience counter
        else:
            patience_counter += 1
            print(f'No improvement in validation loss. Patience: {patience_counter}/{early_stopping_patience}')

        # Check if early stopping should be triggered
        if patience_counter >= early_stopping_patience:
            print(f'Early stopping triggered after {i_epoch + 1} epochs.')
            break

    print("Training finished.")

DEVICE = get_device()
access_token = "" # Introduce tu token de acceso aquí
tokenizer = AutoTokenizer.from_pretrained("Marxx01/gpt2-catalan-tokenizer", token = access_token)
tokenizer.pad_token = tokenizer.eos_token

catlog = load_dataset("projecte-aina/CATalog", token=access_token, trust_remote_code=True, streaming=True, split='train')
dacsa_train = load_dataset("ELiRF/dacsa", "catalan", token=access_token, trust_remote_code=True, streaming=True, split='train')

dacsa_train = dacsa_train.rename_column("article", "text")

# Asignar datasets a variables para entrenamiento y prueba
ds = catlog.filter(lambda example: example["text"] is not None)
ds_test = dacsa_train.filter(lambda example: example["text"] is not None)

config = GPT2Config(
    vocab_size=tokenizer.vocab_size,  # Usar el tamaño real del vocabulario
    n_positions=1024,
    n_ctx=1024,
    n_embd=1024,
    n_layer=24,
    n_head=16,
    bos_token_id=tokenizer.bos_token_id,  # Asegurar que coincidan
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,  # Importante si usas padding
)

model = GPT2LMHeadModel(config)  # Modelo inicializado con pesos aleatorios
model.to(DEVICE)

tokenized_dataset_train = ds.map(tokenize_function, batched=True, remove_columns=["id", "score", "strategy", "languages", "url"])
tokenized_dataset_val = ds_test.map(tokenize_function, batched=True, remove_columns=["id", "summary", "article"])

pytorch_dataset = HuggingFaceIterableDataset(tokenized_dataset_train)
pytorch_dataset_val = HuggingFaceIterableDataset(tokenized_dataset_val)

optimizer = AdamW(model.parameters(), lr=5e-5)
criterion = torch.nn.CrossEntropyLoss()

print("Iniciando entrenamiento...🥵🥵")
train_torch(early_stopping_patience=3, early_stopping_threshold=0.005, epochs=10, batch_size = 4)