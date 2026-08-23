"""
Phase 2, Step 4b: Fix stale bos/eos token ids left over from before the
embedding resize. Same bug pattern as Phase 1's pad_token_id issue --
model.config still points at the ORIGINAL model's ids (50256), which
are now out of range for the resized (30523-row) embedding matrix.
Run this once, then save over the checkpoint so it's fixed for good.
"""
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "hinglish_gpt2_finetuned"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

print("Before fix:")
print("  bos_token_id:", model.config.bos_token_id)
print("  eos_token_id:", model.config.eos_token_id)
print("  pad_token_id:", model.config.pad_token_id)

model.config.bos_token_id = tokenizer.eos_token_id
model.config.eos_token_id = tokenizer.eos_token_id
model.config.pad_token_id = tokenizer.eos_token_id

print("\nAfter fix:")
print("  bos_token_id:", model.config.bos_token_id)
print("  eos_token_id:", model.config.eos_token_id)
print("  pad_token_id:", model.config.pad_token_id)
print("  (tokenizer length / valid range ceiling:", len(tokenizer), ")")

model.save_pretrained(MODEL_PATH)
tokenizer.save_pretrained(MODEL_PATH)
print("\nSaved corrected config to", MODEL_PATH)
