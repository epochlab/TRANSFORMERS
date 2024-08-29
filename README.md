# TRANSFORMERS

**Project ID:** t44Po19k

<div align="center">
<picture>
  <img alt="Stage Skull" src="https://static.wikia.nocookie.net/2001/images/2/21/HAL_closeup.jpg" width="100%">
</picture>
</div>

--------------------------------------------------------------------

### Large Language Models
Abstract: *A **Large Language Model (LLM)** is a machine learning model designed to generate coherent and contextually relevant text by predicting the next token in a sequence based on previous tokens. It operates in a high-dimensional embedding space, where it captures complex patterns in language by processing vast amounts of text data.*

## Installation
    git clone https://github.com/epochlab/TRANSFORMERS
    pip install -r requirements

Update model directory paths in `main.py`

Run:
```
python main.py
```

## Models
Currently we support [Llama2](https://llama.meta.com/llama2/) as our base model.

## System Call
Define your agent profile by stating its objective / persona in `profiles.yml` and identify the agent in `main.py`