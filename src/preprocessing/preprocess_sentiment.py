import os
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pickle

def load_tweeteval_split(split: str = "test") -> pd.DataFrame:
    splits = {
        "train": "sentiment/train-00000-of-00001.parquet",
        "test": "sentiment/test-00000-of-00001.parquet",
        "validation": "sentiment/validation-00000-of-00001.parquet",
    }
    path = f"hf://datasets/cardiffnlp/tweet_eval/{splits[split]}"
    df = pd.read_parquet(path)
    df["source"] = f"tweeteval_{split}"
    df["label_text"] = df["label"].map({0: "negative", 1: "neutral", 2: "positive"})
    df['label'] = df['label_text'].map({'negative': 0, 'positive': 1})
    return df[["text", "label", "label_text", "source"]]


def load_amazon_polarity_split(split: str = "test") -> pd.DataFrame:
    import dask.dataframe as dd
    splits = { 
        "train": "data/train-*.parquet",
        "test": "data/test-00000-of-00001.parquet",
    }
    path = f"hf://datasets/mteb/amazon_polarity/{splits[split]}"
    df = dd.read_parquet(path).compute()
    df["source"] = f"amazon_{split}"
    df['label'] = df['label_text'].map({'negative': 0, 'positive': 1})
    return df[["text", "label", "label_text", "source"]]


def load_datasets(tweet_split: str = "test", amazon_split: str = "test") -> pd.DataFrame:
    df_tweet = load_tweeteval_split(tweet_split)
    df_amazon = load_amazon_polarity_split(amazon_split)
    return pd.concat([df_tweet, df_amazon], ignore_index=True)


def get_model_and_tokenizer(model_name: str = "LiYuan/amazon-review-sentiment-analysis"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model, tokenizer


def compute_embeddings_and_predictions(df: pd.DataFrame, model, tokenizer, batch_size: int = 16, tmp_path: str = None):
    """Compute embeddings and predictions batchwise to save memory."""
    label_names = ['negative', 'negative', 'neutral', 'positive', 'positive']
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if tmp_path and os.path.exists(tmp_path):
        os.remove(tmp_path)

    for i in tqdm(range(0, len(df), batch_size), desc="Processing batches"):
        batch = df.iloc[i:i + batch_size].copy()
        inputs = tokenizer(batch["text"].tolist(), return_tensors="pt", padding=True, truncation=True).to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            last_hidden_state = outputs.hidden_states[-1]

            cls_emb = last_hidden_state[:, 0, :].cpu()
            mask = inputs["attention_mask"].unsqueeze(-1).expand(last_hidden_state.size())
            mean_emb = (torch.sum(last_hidden_state * mask, 1) / mask.sum(1)).cpu()
            probs = F.softmax(outputs.logits, dim=-1).cpu()

        batch["cls_embedding"] = list(cls_emb.numpy())
        batch["mean_embedding"] = list(mean_emb.numpy())
        batch["predicted_label_text"] = [label_names[p.argmax().item()] for p in probs]

        if tmp_path:
            with open(tmp_path, "ab") as f:
                pickle.dump(batch, f)

        del inputs, outputs, last_hidden_state, cls_emb, mean_emb, probs, batch
        torch.cuda.empty_cache()


def read_incremental_pickle(path: str) -> pd.DataFrame:
    """Read concatenated pickle batches written with append mode."""
    frames = []
    with open(path, "rb") as f:
        while True:
            try:
                frames.append(pickle.load(f))
            except EOFError:
                break
    return pd.concat(frames, ignore_index=True)


def remove_neutrals(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df['label_text'] != 'neutral') & (df['predicted_label_text'] != 'neutral')]


def map_predicted_labels(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {"negative": 0, "positive": 1}
    df["predicted_label"] = df["predicted_label_text"].map(mapping)
    return df


def main(tweet_split: str = "test", amazon_split: str = "test", output_path: str = "data/processed/sentiment_combined.pkl"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tmp_path = output_path.replace(".pkl", "_tmp.pkl")

    df = load_datasets(tweet_split=tweet_split, amazon_split=amazon_split)
    model, tokenizer = get_model_and_tokenizer()
    compute_embeddings_and_predictions(df, model, tokenizer, batch_size=16, tmp_path=tmp_path)

    df = read_incremental_pickle(tmp_path)
    df = remove_neutrals(df)
    df = map_predicted_labels(df)

    df.to_pickle(output_path)
    os.remove(tmp_path)
    print(f"Saved combined sentiment data to {output_path}")


if __name__ == "__main__":
    main()
