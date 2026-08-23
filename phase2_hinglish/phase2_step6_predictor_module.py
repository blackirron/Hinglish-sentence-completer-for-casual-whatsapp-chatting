"""
Phase 2, Step 6: Predictor module for the fine-tuned Hinglish model.

Simpler than the Devanagari version's predict_next_word -- this
tokenizer's vocab covers common words as single tokens directly, and
has no boundary marker to extend on anyway (confirmed: encoding is
lossy for word boundaries). So each top-k token IS the candidate word,
no extension loop.

complete_sentence() follows the same design as before: stop after
~max_sentences sentences based on punctuation count, not a fixed
token budget.
"""
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_PATH = "hinglish_gpt2_finetuned"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(device)
model.eval()
print("Using device:", device)  # if this prints 'cpu', that's almost certainly why a run feels slow -- reselect a GPU runtime

SENTENCE_END_CHARS = {".", "?", "!"}  # Roman script -- no Devanagari danda needed here


def predict_next_word(prompt, k=5):
    """
    Returns top-k candidate next words. Each top-k token is treated as
    a complete word directly (see module docstring for why this is the
    right design for this specific tokenizer).
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(input_ids).logits[0, -1, :]
    probs = torch.softmax(logits, dim=-1)
    top_probs, top_ids = torch.topk(probs, k)

    candidates = []
    for tid, prob in zip(top_ids, top_probs):
        word = tokenizer.decode([tid.item()]).strip()
        if word and word not in ("[EOS]",):
            candidates.append((word, prob.item()))
    return candidates


def complete_sentence(prompt, max_sentences=3, max_new_tokens=80, top_k=50, top_p=0.95, temperature=0.8, repetition_penalty=1.2):
    """
    Autoregressive completion capped at ~max_sentences sentences.
    Uses HF generate() with repetition_penalty (the base model showed a
    repetition-loop failure mode without it) and manually checks the
    decoded text for sentence-ending punctuation to decide when to stop,
    regenerating with a growing token budget until we hit the cap or the
    safety ceiling.
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    # Decode the prompt on its own so we can tell prompt punctuation
    # apart from NEWLY generated punctuation below -- otherwise a prompt
    # that already contains sentence-ending punctuation (e.g. pasting a
    # previous completion back in) triggers the stopping condition
    # immediately, before any new text is actually generated.
    prompt_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)

    for budget in range(20, max_new_tokens + 1, 20):
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=budget,
                do_sample=True,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=2,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        generated_only = text[len(prompt_text):]  # only the NEW portion
        sentence_count = sum(generated_only.count(c) for c in SENTENCE_END_CHARS)
        if sentence_count >= max_sentences:
            # Trim to exactly max_sentences worth of NEW text
            cut = len(generated_only)
            count = 0
            for i, ch in enumerate(generated_only):
                if ch in SENTENCE_END_CHARS:
                    count += 1
                    if count == max_sentences:
                        cut = i + 1
                        break
            return (prompt_text + generated_only[:cut]).strip()

    return text.strip()  # fell through to max_new_tokens without hitting max_sentences


if __name__ == "__main__":
    test_prompts = ["main aaj bahut", "kya tum", "hum kal", "yeh movie", "mujhe lagta hai ki"]

    for p in test_prompts:
        print(f"\n=== Prompt: {p!r} ===")
        import time
        t0 = time.time()
        print("Next-word candidates:")
        for word, prob in predict_next_word(p, k=5):
            print(f"  {word!r}  ({prob:.3f})")
        print(f"  (next-word took {time.time() - t0:.1f}s)")
        t1 = time.time()
        print("Sentence completion (up to 3 sentences):")
        print(" ", complete_sentence(p, max_sentences=3))
        print(f"  (completion took {time.time() - t1:.1f}s)")
