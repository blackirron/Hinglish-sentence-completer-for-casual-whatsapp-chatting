"""
Step 4b: Compare base vs fine-tuned model on the same prompts.
Confirms fine-tuning actually improved things before we build on top of it.
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
base_tok, base_model = load("surajp/gpt2-hindi")
print("Loading fine-tuned model...")
ft_tok, ft_model = load("hindi_gpt2_finetuned")

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
            pad_token_id=tok.eos_token_id,
        )
    return tok.decode(output[0], skip_special_tokens=True)

test_prompts = ["भारत एक", "मुझे आज बहुत", "स्कूल जाना", "तुम क्या", "हम कल"]

for p in test_prompts:
    print(f"\n--- Prompt: {p} ---")
    print("Base:      ", generate(base_tok, base_model, p))
    print("Fine-tuned:", generate(ft_tok, ft_model, p))
