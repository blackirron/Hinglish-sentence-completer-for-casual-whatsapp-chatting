"""
Phase 2, Step 2: Build the WhatsApp-style fine-tuning corpus.
Uses Abhishekcr448/Hinglish-Everyday-Conversations-1M -- synthetic but
purpose-built two-person Hinglish dialogue (closer to chat register
than Twitter's broadcast-style HingCorpus).
"""
from datasets import load_dataset
import re

print("Loading Hinglish-Everyday-Conversations-1M...")
dataset = load_dataset("Abhishekcr448/Hinglish-Everyday-Conversations-1M")
print(dataset)

# Sample a manageable subset -- 1M rows is overkill and would make
# training slow on a free-tier GPU; ~100k gives plenty of signal.
SAMPLE_SIZE = 100_000
split = dataset["train"] if "train" in dataset else list(dataset.keys())[0]
if isinstance(split, str):
    split = dataset[split]

print("Columns:", split.column_names)
print("Total rows available:", len(split))

if len(split) > SAMPLE_SIZE:
    split = split.shuffle(seed=42).select(range(SAMPLE_SIZE))

def clean_line(text):
    if text is None:
        return None
    text = text.strip()
    if len(text) < 2 or len(text) > 300:
        return None
    text = re.sub(r"\s+", " ", text)
    return text.strip()

lines = []
input_col, output_col = split.column_names[0], split.column_names[1]
print(f"Using columns: '{input_col}' (input) / '{output_col}' (output)")

for row in split:
    a = clean_line(row[input_col])
    b = clean_line(row[output_col])
    if a:
        lines.append(a)
    if b:
        lines.append(b)

print(f"Total cleaned lines: {len(lines)}")
lines = list(dict.fromkeys(lines))
print(f"After dedup: {len(lines)}")

with open("hinglish_whatsapp_corpus.txt", "w", encoding="utf-8") as f:
    for line in lines:
        f.write(line + "\n")

print("Saved to hinglish_whatsapp_corpus.txt")
print("\nSample lines:")
for l in lines[:10]:
    print(" -", l)
