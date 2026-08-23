"""
Phase 2, Step 1: Sanity check for l3cube-pune/hing-gpt.
Confirms the model loads and generates plausible romanized/code-mixed
Hindi-English text before we build the fine-tuning + predictor pipeline
on top of it, same as we did for the pure-Hindi model.
"""
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_NAME = "l3cube-pune/hing-gpt"

print(f"Loading {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Learned from Phase 1: check for the same embedding/pad_token mismatch
# before it silently causes a CUDA device-side assert down the line.
print("Tokenizer length:", len(tokenizer))
print("Model embedding rows:", model.get_input_embeddings().weight.shape[0])
print("Tokenizer pad_token_id:", tokenizer.pad_token_id)
print("Tokenizer eos_token_id:", tokenizer.eos_token_id)
print("Model config pad_token_id:", model.config.pad_token_id)

if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
    print("MISMATCH -- would need resize_token_embeddings() before training, same as Phase 1.")
else:
    print("Tokenizer/embedding sizes match -- no resize needed.")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()
print("Using device:", device)

def generate(prompt, max_new_tokens=25):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.8,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)

test_prompts = [
    "main aaj bahut",
    "kya tum",
    "hum kal",
    "yeh movie",
    "mujhe lagta hai ki",
]

for p in test_prompts:
    print(f"\n--- Prompt: {p!r} ---")
    print("Generated:", generate(p))
