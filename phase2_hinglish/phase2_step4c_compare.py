"""
Phase 2, Step 4c: Compare base HingGPT vs fine-tuned HingGPT on Hinglish prompts.
"""
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

def load(path):
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path).to(device)
    model.eval()
    return tok, model

print("Loading base model...")
base_tok, base_model = load("l3cube-pune/hing-gpt")
print("Loading fine-tuned model...")
ft_tok, ft_model = load("hinglish_gpt2_finetuned")

def generate(tok, model, prompt, max_new_tokens=25):
    input_ids = tok.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.8,
            repetition_penalty=1.2,  # addresses the repetition-loop issue seen earlier
            no_repeat_ngram_size=2,
            pad_token_id=tok.eos_token_id if tok.eos_token_id is not None else tok.pad_token_id,
        )
    return tok.decode(output[0], skip_special_tokens=True)

test_prompts = ["main aaj bahut", "kya tum", "hum kal", "yeh movie", "mujhe lagta hai ki"]

for p in test_prompts:
    print(f"\n--- Prompt: {p} ---")
    print("Base:      ", generate(base_tok, base_model, p))
    print("Fine-tuned:", generate(ft_tok, ft_model, p))
