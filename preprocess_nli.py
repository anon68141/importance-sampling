"""
Preprocessing script for NLI datasets (SNLI + MNLI).

Steps:
- Load open datasets from Hugging Face using named splits
- Concatenate them into one combined DataFrame
- Compute BERT-based embeddings and predicted labels
- Compute SBERT embedding for premise + hypothesis
- Map predicted label text to numeric codes
- Save processed data as a pickle file
"""

import os
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer


def load_snli_split(split: str = "train") -> pd.DataFrame:
    """Load the SNLI dataset split."""
    splits = {
        "train": "plain_text/train-00000-of-00001.parquet",
        "validation": "plain_text/validation-00000-of-00001.parquet",
        "test": "plain_text/test-00000-of-00001.parquet",
    }
    path = f"hf://datasets/stanfordnlp/snli/{splits[split]}"
    df = pd.read_parquet(path)
    df["source"] = f"snli_{split}"
    return df[["premise", "hypothesis", "label", "source"]]


def load_mnli_split(split: str = "validation_mismatched") -> pd.DataFrame:
    """Load the MNLI dataset split."""
    splits = {
        "train": "data/train-00000-of-00001.parquet",
        "validation_matched": "data/validation_matched-00000-of-00001.parquet",
        "validation_mismatched": "data/validation_mismatched-00000-of-00001.parquet",
    }
    path = f"hf://datasets/nyu-mll/multi_nli/{splits[split]}"
    df = pd.read_parquet(path)
    df["source"] = f"mnli_{split}"
    return df[["premise", "hypothesis", "label", "source"]]


def load_datasets(snli_split: str = "train", mnli_split: str = "validation_mismatched") -> pd.DataFrame:
    """Load and combine SNLI and MNLI splits."""
    df_snli = load_snli_split(snli_split)
    df_mnli = load_mnli_split(mnli_split)
    df = pd.concat([df_snli, df_mnli], ignore_index=True)
    print(f"Loaded {len(df)} total NLI samples ({snli_split}, {mnli_split}).")
    return df


def get_model_and_tokenizer(model_name: str = "textattack/bert-base-uncased-snli"):
    """Load a pretrained NLI model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return model, tokenizer


def get_sbert_model():
    """Load SBERT for sentence embeddings."""
    return SentenceTransformer("sentence-transformers/all-mpnet-base-v2")


def compute_embeddings_and_predictions(df: pd.DataFrame, model, tokenizer, sbert_model) -> pd.DataFrame:
    """Compute CLS and mean embeddings, predicted labels, and SBERT embeddings."""
    label_names = ["contradiction", "entailment", "neutral"]

    cls_embeddings, mean_embeddings, predicted_labels = [], [], []
    sbert_embeddings = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing samples"):
        premise = row["premise"]
        hypothesis = row["hypothesis"]

        inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        last_hidden_state = outputs.hidden_states[-1]

        cls_emb = last_hidden_state[:, 0, :].squeeze().cpu().numpy()
        mask = inputs["attention_mask"].unsqueeze(-1).expand(last_hidden_state.size())
        mean_emb = (torch.sum(last_hidden_state * mask, 1) / mask.sum(1)).squeeze().cpu().numpy()

        probs = F.softmax(outputs.logits, dim=-1)
        pred_label = label_names[probs.argmax().item()]

        cls_embeddings.append(cls_emb)
        mean_embeddings.append(mean_emb)
        predicted_labels.append(pred_label)

        combined_text = f"{premise} [SEP] {hypothesis}"
        sbert_emb = sbert_model.encode(combined_text)
        sbert_embeddings.append(sbert_emb)

    df["cls_embedding"] = cls_embeddings
    df["mean_embedding"] = mean_embeddings
    df["predicted_label_text"] = predicted_labels
    df["sbert_embedding"] = sbert_embeddings

    return df


def map_predicted_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Map predicted label text to numeric codes."""
    mapping = {"entailment": 0, "neutral": 1, "contradiction": 2}
    df["predicted_label"] = df["predicted_label_text"].map(mapping)
    return df


def main(snli_split: str = "test", 
         mnli_split: str = "validation_mismatched", 
         output_path: str = "data/processed/nli_combined.pkl"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df = load_datasets(snli_split=snli_split, mnli_split=mnli_split)
    model, tokenizer = get_model_and_tokenizer()
    sbert_model = get_sbert_model()

    df = compute_embeddings_and_predictions(df, model, tokenizer, sbert_model)
    df = map_predicted_labels(df)

    df.to_pickle(output_path)
    print(f"Saved combined NLI data to {output_path}")


if __name__ == "__main__":
    main()
