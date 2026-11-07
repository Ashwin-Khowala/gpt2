
"""
HellaSwag evaluation for GPT-2.
Dataset source: Hugging Face Rowan/hellaswag
"""

import os
import json
import requests
import pyarrow.parquet as pq
import tiktoken
from tqdm import tqdm

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel

# ---------------------------------------------------------
DATA_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "hellaswag"
)

DATASET_URLS = {
    "train": "https://huggingface.co/datasets/Rowan/hellaswag/resolve/main/data/train-00000-of-00001.parquet",
    "val": "https://huggingface.co/datasets/Rowan/hellaswag/resolve/main/data/validation-00000-of-00001.parquet",
    "test": "https://huggingface.co/datasets/Rowan/hellaswag/resolve/main/data/test-00000-of-00001.parquet",
}

enc = tiktoken.get_encoding("gpt2")


# ---------------------------------------------------------
def download_file(url, fname):
    """Download a file safely."""

    temp_file = fname + ".tmp"

    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()

        total = int(resp.headers.get("content-length", 0))

        with open(temp_file, "wb") as f:
            for chunk in tqdm(
                resp.iter_content(chunk_size=1024 * 1024),
                total=(total // (1024 * 1024) + 1),
                desc="Downloading dataset"
            ):
                if chunk:
                    f.write(chunk)

    os.replace(temp_file, fname)



def download(split):
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    parquet_file = os.path.join(
        DATA_CACHE_DIR,
        f"hellaswag_{split}.parquet"
    )

    jsonl_file = os.path.join(
        DATA_CACHE_DIR,
        f"hellaswag_{split}.jsonl"
    )

    if not os.path.exists(parquet_file):
        print(f"Downloading {split} dataset...")
        download_file(DATASET_URLS[split], parquet_file)

    if os.path.getsize(parquet_file) < 100_000:
        raise RuntimeError("Downloaded Parquet file is invalid or incomplete")

    print("Reading Parquet dataset...")

    table = pq.read_table(parquet_file)
    rows = table.to_pylist()

    print("Dataset columns:", table.column_names)
    print("Total rows:", len(rows))

    temp_file = jsonl_file + ".tmp"

    print("Converting Parquet to JSONL...")

    with open(temp_file, "w", encoding="utf-8") as f:
        for row in rows:
            row["label"] = int(row["label"])
            f.write(json.dumps(row) + "\n")

    os.replace(temp_file, jsonl_file)

    # Verify every line.
    count = 0

    with open(jsonl_file, encoding="utf-8") as f:
        for line in f:
            example = json.loads(line)

            assert len(example["endings"]) == 4
            assert 0 <= int(example["label"]) <= 3

            count += 1

    print(f"Successfully verified {count} examples")

    if split == "val" and count != 10042:
        raise RuntimeError(f"Expected 10042 examples, got {count}")

    return jsonl_file

# ---------------------------------------------------------
def render_example(example):

    ctx = example["ctx"]
    label = int(example["label"])
    endings = example["endings"]

    data = {
        "label": label,
        "ctx_tokens": None,
        "ending_tokens": [],
    }

    ctx_tokens = enc.encode(ctx)
    data["ctx_tokens"] = ctx_tokens

    tok_rows = []
    mask_rows = []

    for end in endings:

        end_tokens = enc.encode(" " + end)

        tok_rows.append(ctx_tokens + end_tokens)

        mask_rows.append(
            [0] * len(ctx_tokens) +
            [1] * len(end_tokens)
        )

        data["ending_tokens"].append(end_tokens)

    max_len = max(len(row) for row in tok_rows)

    tokens = torch.zeros((4, max_len), dtype=torch.long)
    mask = torch.zeros((4, max_len), dtype=torch.long)

    for i, (tok_row, mask_row) in enumerate(
        zip(tok_rows, mask_rows)
    ):

        tokens[i, :len(tok_row)] = torch.tensor(tok_row)
        mask[i, :len(mask_row)] = torch.tensor(mask_row)

    return data, tokens, mask, label


# ---------------------------------------------------------
def iterate_examples(split):

    jsonl_file = download(split)

    with open(jsonl_file, encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


# ---------------------------------------------------------
@torch.no_grad()
def evaluate(model_type, device):

    torch.set_float32_matmul_precision("high")

    model = GPT2LMHeadModel.from_pretrained(model_type)
    model.to(device)
    model.eval()

    num_correct_norm = 0
    num_correct = 0
    num_total = 0

    for example in tqdm(iterate_examples("val"), total=10042):

        data, tokens, mask, label = render_example(example)

        tokens = tokens.to(device)
        mask = mask.to(device)

        logits = model(tokens).logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_tokens = tokens[..., 1:].contiguous()

        flat_shift_logits = shift_logits.view(
            -1, shift_logits.size(-1)
        )

        flat_shift_tokens = shift_tokens.view(-1)

        shift_losses = F.cross_entropy(
            flat_shift_logits,
            flat_shift_tokens,
            reduction="none"
        )

        shift_losses = shift_losses.view(tokens.size(0), -1)

        shift_mask = mask[..., 1:].contiguous()

        masked_shift_losses = shift_losses * shift_mask

        sum_loss = masked_shift_losses.sum(dim=1)

        avg_loss = sum_loss / shift_mask.sum(dim=1)

        pred = sum_loss.argmin().item()
        pred_norm = avg_loss.argmin().item()

        num_total += 1

        num_correct += int(pred == label)
        num_correct_norm += int(pred_norm == label)

        if num_total <= 9:
            print("\n---")
            print("Context:", example["ctx"])

            for i, end in enumerate(example["endings"]):
                print(
                    f"{i} (loss: {avg_loss[i].item():.4f}) {end}"
                )

            print(
                f"Predicted: {pred_norm}, Actual: {label}"
            )

    print("\n========== FINAL RESULTS ==========")

    print(f"Total examples: {num_total}")

    print(
        f"Accuracy: {num_correct}/{num_total} "
        f"= {num_correct / num_total:.4f}"
    )

    print(
        f"Normalized accuracy: "
        f"{num_correct_norm}/{num_total} "
        f"= {num_correct_norm / num_total:.4f}"
    )


# ---------------------------------------------------------
if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m", "--model_type",
        type=str,
        default="gpt2"
    )

    parser.add_argument(
        "-d", "--device",
        type=str,
        default="cuda"
    )

    args = parser.parse_args()

    evaluate(args.model_type, args.device)
