"""
Step 4: Fine-tune surajp/gpt2-hindi on the conversational corpus.
Uses HF Trainer with causal LM loss. Tracks eval loss -> perplexity
each epoch so we can see if it's actually learning vs overfitting.
"""
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    Trainer, TrainingArguments, default_data_collator
)
from datasets import load_from_disk
import torch
import math

MODEL_NAME = "surajp/gpt2-hindi"
DATASET_PATH = "hindi_lm_dataset"
OUTPUT_DIR = "hindi_gpt2_finetuned"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# The tokenizer's EOS token was added as a special token at id 50257 --
# one slot past the base embedding matrix's size (rows 0-50256 only).
# Setting pad_token_id to that id (or using EOS at all) is an out-of-bounds
# embedding lookup unless we resize the embedding matrix to match the
# tokenizer's true length first. This is the actual fix -- our earlier
# attempt (just reassigning pad_token_id) pointed at the same broken id.
print("Tokenizer length:", len(tokenizer), "| Model embedding rows before resize:", model.get_input_embeddings().weight.shape[0])
model.resize_token_embeddings(len(tokenizer))
print("Model embedding rows after resize:", model.get_input_embeddings().weight.shape[0])

model.config.pad_token_id = tokenizer.eos_token_id
print("model.config.pad_token_id ->", model.config.pad_token_id, "(now a valid row)")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Training on:", device)

dataset = load_from_disk(DATASET_PATH)
# input_ids, labels, and attention_mask are all present from Step 3.
# default_data_collator converts these python lists to tensors directly --
# no .set_format(type="torch") needed (that path triggers a datasets/
# torchvision version conflict in current Colab environments).

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    learning_rate=5e-5,
    warmup_steps=100,
    weight_decay=0.01,
    fp16=torch.cuda.is_available(),
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=default_data_collator,
)

print("Starting fine-tuning...")
train_result = trainer.train()

eval_results = trainer.evaluate()
perplexity = math.exp(eval_results["eval_loss"])
print(f"\nFinal eval loss: {eval_results['eval_loss']:.4f}")
print(f"Final perplexity: {perplexity:.2f}")

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\nModel saved to ./{OUTPUT_DIR}")
