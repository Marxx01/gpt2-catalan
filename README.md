# README - Catalan Language Model

## Description
This repository contains a language model based on the GPT-2 architecture, trained entirely from scratch for the Catalan language. Unlike other models fine-tuned from existing versions, this model has been developed from the ground up to ensure an optimized and accurate representation of Catalan in all its forms.

## Features
- **Architecture**: Based on GPT-2
- **Training**: From scratch, without using pre-trained models in other languages
- **Tokenizer**: Developed from scratch with a vocabulary of 52,000 tokens
- **Training Corpus**:
  - [ELiRF/dacsa](https://huggingface.co/datasets/ELiRF/dacsa)
  - [projecte-aina/CATalog](https://huggingface.co/datasets/projecte-aina/CATalog)

## Purpose
This model aims to provide a powerful NLP tool optimized for Catalan, with applications in text generation, writing assistance, automatic translation, and other natural language processing tasks.

## Usage
The model can be used with the `transformers` library from Hugging Face:
- [Model link](https://huggingface.co/Marxx01/gpt2_catalan)


```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the tokenizer and the model
model_name = "Marxx01/gpt2_catalan"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Generate sample text
input_text = "El futur de la llengua catalana "
input_ids = tokenizer.encode(input_text, return_tensors="pt")
outputs = model.generate(
    **inputs,
    do_sample = True,
    max_length=150, 
    temperature=0.7, 
    top_p=0.8,  
    top_k=1000, 
    no_repeat_ngram_size=2, 
    num_return_sequences=1
)

print(tokenizer.decode(output[0], skip_special_tokens=True))
```

# README - Model de Llengua Català

## Descripció
Aquest repositori conté un model de llenguatge basat en l'arquitectura GPT-2, entrenat completament des de zero per a la llengua catalana. A diferència d'altres models ajustats a partir de versions preexistents, aquest model ha estat desenvolupat des de la base per garantir una representació precisa i optimitzada del català en totes les seves formes.

## Característiques
- **Arquitectura**: Basada en GPT-2
- **Entrenament**: Des de zero, sense ús de models preentrenats en altres idiomes
- **Tokenitzador**: Desenvolupat des de zero amb un vocabulari de 52.000 tokens
- **Corpus d'entrenament**:
  - [ELiRF/dacsa](https://huggingface.co/datasets/ELiRF/dacsa)
  - [projecte-aina/CATalog](https://huggingface.co/datasets/projecte-aina/CATalog)

## Objectiu
Aquest model té com a objectiu proporcionar una eina de processament del llenguatge natural optimitzada per al català, amb aplicacions en generació de text, assistència en l'escriptura, traducció automàtica i altres tasques relacionades amb NLP.

## Ús
El model es pot utilitzar mitjançant la biblioteca `transformers` de Hugging Face:
- [Model link](https://huggingface.co/Marxx01/gpt2_catalan)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Carregar el tokenitzador i el model
model_name = "Marxx01/gpt2_catalan"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Generar text d'exemple
input_text = "El futur de la llengua catalana "
input_ids = tokenizer.encode(input_text, return_tensors="pt")
outputs = model.generate(
    **inputs,
    do_sample = True,
    max_length=150, 
    temperature=0.7, 
    top_p=0.8,  
    top_k=1000, 
    no_repeat_ngram_size=2, 
    num_return_sequences=1
)

print(tokenizer.decode(output[0], skip_special_tokens=True))
```
