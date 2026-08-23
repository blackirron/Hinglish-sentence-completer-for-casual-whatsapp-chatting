"""
Phase 2, Step 4: Fine-tune HingGPT on the WhatsApp-style Hinglish corpus.
Loads the MODIFIED tokenizer from Step 3 (has the added [EOS] token) --
not the original model name -- and resizes the embedding matrix to match,
since the tokenizer grew by one token.
"""
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    Trainer, TrainingArguments, default_data_collator
)
from datasets import load_from_disk
import torch
import math

MODEL_NAME = "l3cube-pune/hing-gpt"
TOKENIZER_PATH = "hinglish_tokenizer"  # the modified one from Step 3, not MODEL_NAME
DATASET_PATH = "hinglish_lm_dataset"
OUTPUT_DIR = "hinglish_gpt2_finetuned"

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

print("Tokenizer length:", len(tokenizer), "| Model embedding rows before resize:", model.get_input_embeddings().weight.shape[0])
model.resize_token_embeddings(len(tokenizer))
print("Model embedding rows after resize:", model.get_input_embeddings().weight.shape[0])

model.config.pad_token_id = tokenizer.pad_token_id
print("model.config.pad_token_id ->", model.config.pad_token_id)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print("Training on:", device)

dataset = load_from_disk(DATASET_PATH)

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
