"""
Phase 2, Step 5 (prep): Check HingGPT's tokenizer convention before
reusing the Hindi model's word-boundary logic. WordPiece-style vocabs
typically use ## to mark continuation pieces, not a leading space to
mark word starts -- need to confirm which applies here before building
predict_next_word().
"""
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("hinglish_gpt2_finetuned")

test_words = ["kya", "tum", "karna", "chahiye", "bahut", "khushi", "conversation"]
for w in test_words:
    ids = tokenizer.encode(w, add_special_tokens=False)
    pieces = [tokenizer.decode([i]) for i in ids]
    print(f"{w!r} -> ids {ids} -> pieces {pieces}")

print()
sentence = "kya tum bahut khush ho"
ids = tokenizer.encode(sentence, add_special_tokens=False)
print("Full sentence tokenization:")
for i in ids:
    print(f"  id={i}  piece={tokenizer.decode([i])!r}  raw_token={tokenizer.convert_ids_to_tokens([i])}")
