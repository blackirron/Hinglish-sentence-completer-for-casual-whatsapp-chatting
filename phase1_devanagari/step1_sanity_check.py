"""
Step 1: Environment + base model sanity check.
Run this in Colab (free GPU) or locally. Confirms the pretrained
Hindi GPT-2 model loads and can generate coherent-ish Hindi text
before we touch fine-tuning.
"""
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_NAME = "surajp/gpt2-hindi"  # GPT-2 architecture, 88M params, pretrained from scratch on Hindi

print(f"Loading {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print("Using device:", device)

def complete_sentence(prompt, max_new_tokens=30, top_k=50, top_p=0.95, temperature=0.8):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)

def next_word_candidates(prompt, k=5):
    """Single forward pass -> top-k next token candidates with probabilities."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(input_ids).logits[0, -1, :]  # logits for the next token
    probs = torch.softmax(logits, dim=-1)
    top_probs, top_ids = torch.topk(probs, k)
    return [(tokenizer.decode([tid]), prob.item()) for tid, prob in zip(top_ids, top_probs)]

if __name__ == "__main__":
    test_prompts = ["भारत एक", "मुझे आज बहुत", "स्कूल जाना"]

    for p in test_prompts:
        print("\n--- Prompt:", p, "---")
        print("Completion:", complete_sentence(p))
        print("Top next-word candidates:")
        for word, prob in next_word_candidates(p):
            print(f"  {word!r}  ({prob:.3f})")

    print("\nVocab size:", tokenizer.vocab_size)
    print("Model params:", sum(p.numel() for p in model.parameters()))
