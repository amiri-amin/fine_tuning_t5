# T5 Fine-Tuning for Article Title Generation

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776ab.svg)](https://www.python.org/)
[![Transformers](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-Seq2Seq-yellow.svg)](https://huggingface.co/docs/transformers)
[![Model](https://img.shields.io/badge/base%20model-t5--base-ff6f00.svg)](https://huggingface.co/t5-base)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Fine-tune Google's **T5** text-to-text transformer to **generate a headline/title from the
body of an article**. The task is framed as abstractive summarization: given an article's
text, the model produces a short, catchy title.

Training uses a corpus of ~14.7k Medium articles (`articles.csv`) and the Hugging Face
`Seq2SeqTrainer`. A companion script loads a fine-tuned checkpoint and generates a title for
any new text.

---

## Table of contents

- [How it works](#how-it-works)
- [Repository layout](#repository-layout)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Inference](#inference)
- [Training configuration](#training-configuration)
- [Notes &amp; tips](#notes--tips)
- [Citation](#citation)
- [License](#license)

---

## How it works

T5 casts every problem as text-to-text. Here the input is the article body prefixed with the
task tag `summarize:`, and the target is the article's title:

```
   input:  "summarize: <article text …>"   ──►  T5  ──►   output: "<generated title>"
   target: "<original article title>"
```

- **Base model:** [`t5-base`](https://huggingface.co/t5-base)
- **Tokenizer:** `T5Tokenizer` — inputs truncated to **512** tokens, targets to **128**
- **Objective:** sequence-to-sequence cross-entropy via `Seq2SeqTrainer`
- **Split:** 90 % train / 10 % validation (`train_test_split(test_size=0.1)`)

---

## Repository layout

```
fine_tuning_t5/
├── ztesla_t5_test_dimesion.py   # Training script: load CSV → preprocess → fine-tune T5
├── test_fine_tuned_model.py     # Inference script: load a checkpoint → generate a title
├── articles.csv                 # Dataset (~14.7k rows: author, claps, reading_time, link, title, text)
├── requirements.txt             # Python dependencies
├── LICENSE
└── README.md
```

> Fine-tuned checkpoints are written to `./t5-title-generation/` at run time and are **not**
> tracked in git (see `.gitignore`).

---

## Dataset

`articles.csv` contains Medium articles with the columns:

| Column | Description |
|--------|-------------|
| `author` | Article author |
| `claps` | Reader "claps" (engagement) |
| `reading_time` | Estimated reading time |
| `link` | Source URL |
| **`title`** | **Target** — the headline the model learns to generate |
| **`text`** | **Input** — the article body |

Only `text` (input) and `title` (target) are used for fine-tuning; the other columns are
carried along but ignored by the model.

---

## Installation

```bash
# (recommended) create an isolated environment
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/macOS:  source .venv/bin/activate

pip install -r requirements.txt
```

A CUDA-capable GPU is strongly recommended — the training script enables mixed-precision
(`fp16=True`), which requires a GPU. On CPU-only machines, set `fp16=False` and expect
substantially longer training times.

---

## Usage

### Training

```bash
python ztesla_t5_test_dimesion.py
```

This will:
1. Load `articles.csv` and split it 90/10 into train/validation sets.
2. Tokenize inputs (`summarize: <text>`) and targets (`title`).
3. Fine-tune `t5-base` and save checkpoints under `./t5-title-generation/`.

### Inference

Point the tester at one of the saved checkpoints and run it:

```bash
python test_fine_tuned_model.py
```

By default it loads `./t5-title-generation/checkpoint-500/` and generates a title for the
example article embedded in the script. Edit the `new_text` variable (or the checkpoint path)
to try your own input:

```python
model     = T5ForConditionalGeneration.from_pretrained('./t5-title-generation/checkpoint-500/')
tokenizer = T5Tokenizer.from_pretrained('./t5-title-generation/checkpoint-500/')
...
outputs = model.generate(inputs["input_ids"], max_length=128, num_beams=4, early_stopping=True)
```

---

## Training configuration

Defaults set in `Seq2SeqTrainingArguments` (`ztesla_t5_test_dimesion.py`):

| Argument | Value | Note |
|----------|:-----:|------|
| `learning_rate` | `2e-5` | AdamW learning rate |
| `per_device_train_batch_size` | `8` | Train batch size |
| `per_device_eval_batch_size` | `8` | Eval batch size |
| `weight_decay` | `0.01` | L2 regularization |
| `num_train_epochs` | `100` | Reduce for quick experiments |
| `evaluation_strategy` | `epoch` | Evaluate each epoch |
| `save_total_limit` | `5` | Keep the 5 most recent checkpoints |
| `predict_with_generate` | `True` | Use `generate()` for eval metrics |
| `fp16` | `True` | Mixed-precision (requires GPU) |

---

## Notes & tips

- **100 epochs** is a large budget for a base-size model — lower `num_train_epochs` for a
  faster first run, then scale up.
- The inference script hard-codes `checkpoint-500`; update the path to whichever checkpoint
  you want to evaluate (checkpoint numbers depend on batch size and dataset length).
- `T5Tokenizer` requires **SentencePiece** — installed via `requirements.txt`.
- To publish a trained model, push the checkpoint folder to the Hugging Face Hub and load it
  by name (the tester includes commented-out lines showing how).

---

## Citation

If you use this code, please credit this repository and the underlying T5 model:

```bibtex
@software{amiri_t5_title_generation,
  author = {Amiri, Amin},
  title  = {T5 Fine-Tuning for Article Title Generation},
  year   = {2024},
  url    = {https://github.com/amiri-amin/fine_tuning_t5}
}

@article{raffel2020t5,
  title   = {Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer},
  author  = {Raffel, Colin and Shazeer, Noam and Roberts, Adam and Lee, Katherine and Narang, Sharan and Matena, Michael and Zhou, Yanqi and Li, Wei and Liu, Peter J.},
  journal = {Journal of Machine Learning Research},
  year    = {2020}
}
```

## License

Released under the [MIT License](LICENSE) © Amin Amiri.
