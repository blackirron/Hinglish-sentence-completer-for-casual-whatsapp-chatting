"""
Critical check: does tokenizer.decode() correctly preserve whether a
multi-token word had NO space (compound, one original word) vs whether
adjacent tokens were genuinely separate space-delimited words? This
determines whether decode() can be trusted as our source of truth for
word boundaries, since raw token strings carry no marker at all here.
"""
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("hinglish_gpt2_finetuned")

# Case A: "whatsappkarna" -- ONE original word, no internal space, that
# happened to split into 2 tokens ['whatsapp', 'karna']
ids_a = tokenizer.encode("whatsappkarna", add_special_tokens=False)
print("Case A -- 'whatsappkarna' (one word, no space):")
print("  ids:", ids_a)
print("  decode:", repr(tokenizer.decode(ids_a)))

# Case B: "whatsapp karna" -- TWO separate words, WITH a space
ids_b = tokenizer.encode("whatsapp karna", add_special_tokens=False)
print("\nCase B -- 'whatsapp karna' (two words, with space):")
print("  ids:", ids_b)
print("  decode:", repr(tokenizer.decode(ids_b)))

print("\nAre the id sequences identical?", ids_a == ids_b)

# Case C: a full natural sentence, decode incrementally token by token
# to see exactly where/how decode() inserts spacing
sentence = "kya tum whatsappkarna chahte ho"
ids_c = tokenizer.encode(sentence, add_special_tokens=False)
print("\nCase C -- full sentence incremental decode:")
for i in range(1, len(ids_c) + 1):
    print(f"  after {i} tokens: {tokenizer.decode(ids_c[:i])!r}")
