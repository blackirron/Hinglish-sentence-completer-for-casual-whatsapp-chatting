"""
Step 3: Tokenize and chunk the corpus for causal LM fine-tuning.
Uses the SAME tokenizer as the base model (don't retrain the
tokenizer -- that would break the pretrained embeddings' meaning).
Concatenates all lines and splits into fixed-length blocks, which
is the standard approach for causal LM fine-tuning (as opposed to
padding each short line individually, which wastes compute).
"""
from transformers import AutoTokenizer
from datasets import Dataset
import os

MODEL_NAME = "surajp/gpt2-hindi"
CORPUS_PATH = "hindi_conversational_corpus.txt"
BLOCK_SIZE = 128  # tokens per training example; 128 is a reasonable size for short conversational lines

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading corpus...")
with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    lines = [l.strip() for l in f if l.strip()]
print(f"Loaded {len(lines)} lines.")

# Join lines with EOS token so the model learns sentence boundaries
print("Tokenizing...")
full_text = tokenizer.eos_token.join(lines)
tokenized = tokenizer(full_text, return_tensors=None)["input_ids"]
print(f"Total tokens: {len(tokenized)}")

# Chunk into fixed-length blocks
def chunk_list(lst, size):
    for i in range(0, len(lst) - size + 1, size):
        yield lst[i:i + size]

blocks = list(chunk_list(tokenized, BLOCK_SIZE))
print(f"Total training blocks of size {BLOCK_SIZE}: {len(blocks)}")

# Build HF Dataset: input_ids == labels for causal LM (next-token prediction)
# attention_mask is all 1s since these are fixed-length blocks with no padding
attention_masks = [[1] * BLOCK_SIZE for _ in blocks]
dataset_dict = {"input_ids": blocks, "labels": [b[:] for b in blocks], "attention_mask": attention_masks}
hf_dataset = Dataset.from_dict(dataset_dict)

# 95/5 train/val split for perplexity tracking during training
split_dataset = hf_dataset.train_test_split(test_size=0.05, seed=42)
print(split_dataset)

split_dataset.save_to_disk("hindi_lm_dataset")
print("Saved tokenized dataset to ./hindi_lm_dataset")

print("\nSanity check - decoding first training block:")
print(tokenizer.decode(split_dataset["train"][0]["input_ids"]))
