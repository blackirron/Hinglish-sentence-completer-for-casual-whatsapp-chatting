"""
Step 5: The core predictor module.

Two functions, both built on the fine-tuned model:
1. predict_next_word()  -- returns top-k WHOLE-WORD candidates (not subword
   fragments -- generates token-by-token until a word boundary, as we
   flagged back in Step 1).
2. complete_sentence()  -- autoregressive generation that stops after
   roughly 2-3 sentences (counts sentence-ending punctuation), not a
   fixed token count, so short prompts don't get needlessly padded out
   and longer prompts don't get cut mid-thought.
"""
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_PATH = "hindi_gpt2_finetuned"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(device)
model.eval()

SENTENCE_END_CHARS = {".", "?", "!", "।"}  # । is the Hindi/Devanagari full stop (danda)
WORD_BOUNDARY_CHARS = {" ", "\n"} | SENTENCE_END_CHARS


def predict_next_word(prompt, k=5, max_token_steps=12):
    """
    Returns top-k whole-word candidates for what comes next.
    Generates greedily token-by-token per candidate branch until hitting
    a word boundary (space/punctuation), since the tokenizer works in
    subwords, not whole words.
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        logits = model(input_ids).logits[0, -1, :]
    probs = torch.softmax(logits, dim=-1)
    # Pull a wider pool than k, since some will get filtered out as
    # mid-word continuation pieces (no leading space) below.
    pool_size = min(k * 4, probs.shape[-1])
    top_probs, top_ids = torch.topk(probs, pool_size)

    candidates = []
    for first_id, first_prob in zip(top_ids, top_probs):
        if len(candidates) >= k:
            break
        first_piece_raw = tokenizer.decode([first_id.item()])
        # Skip continuation-only tokens (no leading space) -- these mean
        # the model is predicting a merge into the PRIOR word (e.g. "कल"
        # -> "कल्पना"), not a genuine new next word. Displayed alone
        # they're meaningless fragments like '्पना'.
        if not first_piece_raw.startswith(" "):
            continue

        current_ids = torch.cat([input_ids, first_id.view(1, 1)], dim=1)
        word_piece = first_piece_raw

        # Look ahead: keep extending as long as the NEXT predicted token
        # is a continuation piece (no leading space -- same word). Stop as
        # soon as the next token would start a new word or hit punctuation.
        # (We must check the NEXT token's boundary, not the current one --
        # the current/first token itself always starts with a leading space
        # by BPE convention, which is not itself a stopping signal.)
        VIRAMA = "\u094d"  # ् -- Devanagari conjunct joiner; a word can never validly end on a bare virama, it always expects a following consonant
        steps = 1
        while steps < max_token_steps:
            with torch.no_grad():
                next_logits = model(current_ids).logits[0, -1, :]
            next_id = torch.argmax(next_logits).item()
            next_piece = tokenizer.decode([next_id])
            looks_like_boundary = (
                next_piece.startswith(" ")
                or any(c in next_piece for c in SENTENCE_END_CHARS)
                or next_id == tokenizer.eos_token_id
            )
            # Even if this looks like a boundary, don't stop if the word
            # so far ends in a bare virama -- that's grammatically
            # incomplete (e.g. "संयुक्" needs "त" to become "संयुक्त").
            if looks_like_boundary and not word_piece.endswith(VIRAMA):
                break
            word_piece += next_piece
            current_ids = torch.cat([current_ids, torch.tensor([[next_id]]).to(device)], dim=1)
            steps += 1

        word = word_piece.strip(" .?!।\n")
        if word:
            candidates.append((word, first_prob.item()))

    # Dedupe (different token paths can land on the same surface word)
    seen = set()
    deduped = []
    for word, prob in candidates:
        if word not in seen:
            seen.add(word)
            deduped.append((word, prob))
    return deduped


def complete_sentence(prompt, max_sentences=3, max_new_tokens=80, top_k=50, top_p=0.95, temperature=0.8):
    """
    Generates a completion capped at roughly max_sentences sentences,
    with max_new_tokens as a hard safety ceiling (in case punctuation
    never appears). Input length isn't force-padded -- a short prompt
    naturally tends to produce a short-ish completion from this model.
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated = input_ids

    sentence_count = 0
    new_tokens = 0

    with torch.no_grad():
        while new_tokens < max_new_tokens and sentence_count < max_sentences:
            logits = model(generated).logits[0, -1, :]
            logits = logits / temperature

            # top-p (nucleus) sampling
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            cutoff = cumulative_probs > top_p
            cutoff[1:] = cutoff[:-1].clone()
            cutoff[0] = False
            sorted_logits[cutoff] = float("-inf")
            probs = torch.softmax(sorted_logits, dim=-1)
            next_token_sorted_idx = torch.multinomial(probs, 1)
            next_id = sorted_indices[next_token_sorted_idx]

            generated = torch.cat([generated, next_id.view(1, 1)], dim=1)
            new_tokens += 1

            piece = tokenizer.decode([next_id.item()])
            if any(c in piece for c in SENTENCE_END_CHARS):
                sentence_count += 1

            if next_id.item() == tokenizer.eos_token_id:
                break

    return tokenizer.decode(generated[0], skip_special_tokens=True)


if __name__ == "__main__":
    test_prompts = ["भारत एक", "मुझे आज बहुत", "तुम क्या", "हम कल"]

    for p in test_prompts:
        print(f"\n=== Prompt: {p!r} ===")
        print("Next-word candidates:")
        for word, prob in predict_next_word(p, k=5):
            print(f"  {word!r}  ({prob:.3f})")
        print("Sentence completion (up to 3 sentences):")
        print(" ", complete_sentence(p, max_sentences=3))
