"""
Phase 2, Step 3: Tokenize and chunk the WhatsApp-style corpus for HingGPT.
First inspects what special tokens the tokenizer actually has (it's a
WordPiece-style vocab, vocab size 30522 -- likely reused from a BERT
tokenizer, may have [SEP]/[CLS]/[PAD] under different names even though
pad_token_id/eos_token_id read as None) before deciding whether we need
to add a new one.
"""
from transformers import AutoTokenizer
from datasets import Dataset

MODEL_NAME = "l3cube-pune/hing-gpt"
CORPUS_PATH = "hinglish_whatsapp_corpus.txt"
BLOCK_SIZE = 128

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("All special tokens:", tokenizer.all_special_tokens)
print("Special tokens map:", tokenizer.special_tokens_map)
print("pad_token:", tokenizer.pad_token, "| eos_token:", tokenizer.eos_token, "| sep_token:", getattr(tokenizer, "sep_token", None))

# If there's a usable existing special token, prefer that over adding a
# new one (adding a new one means the embedding matrix must grow by 1 row).
separator_token = None
if tokenizer.eos_token is not None:
    separator_token = tokenizer.eos_token
elif tokenizer.sep_token is not None:
    separator_token = tokenizer.sep_token
else:
    print("No usable existing separator token found -- adding a new [EOS] special token.")
    tokenizer.add_special_tokens({"eos_token": "[EOS]", "pad_token": "[EOS]"})
    separator_token = tokenizer.eos_token

if tokenizer.pad_token is None:
    tokenizer.pad_token = separator_token

print("Using separator token:", repr(separator_token))
print("Final tokenizer length:", len(tokenizer))

print("Loading corpus...")
with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    lines = [l.strip() for l in f if l.strip()]
print(f"Loaded {len(lines)} lines.")

print("Tokenizing...")
full_text = separator_token.join(lines)
tokenized = tokenizer(full_text, return_tensors=None)["input_ids"]
print(f"Total tokens: {len(tokenized)}")

def chunk_list(lst, size):
    for i in range(0, len(lst) - size + 1, size):
        yield lst[i:i + size]

blocks = list(chunk_list(tokenized, BLOCK_SIZE))
print(f"Total training blocks of size {BLOCK_SIZE}: {len(blocks)}")

attention_masks = [[1] * BLOCK_SIZE for _ in blocks]
dataset_dict = {"input_ids": blocks, "labels": [b[:] for b in blocks], "attention_mask": attention_masks}
hf_dataset = Dataset.from_dict(dataset_dict)

split_dataset = hf_dataset.train_test_split(test_size=0.05, seed=42)
print(split_dataset)

split_dataset.save_to_disk("hinglish_lm_dataset")
tokenizer.save_pretrained("hinglish_tokenizer")  # save separately -- may have new special tokens added
print("Saved tokenized dataset to ./hinglish_lm_dataset")
print("Saved (possibly modified) tokenizer to ./hinglish_tokenizer")

print("\nSanity check - decoding first training block:")
print(tokenizer.decode(split_dataset["train"][0]["input_ids"]))
