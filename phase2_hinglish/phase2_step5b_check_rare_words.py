"""
Check tokenizer behavior on rare/compound/misspelled words -- where it
must fall back to splitting into subwords, and where the boundary-marking
convention (## / leading space / none) will actually show up.
"""
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("hinglish_gpt2_finetuned")

test_words = [
    "antidisestablishmentarianism",  # deliberately absurd, guaranteed to split
    "khushiyan",                      # less common inflected Hindi word
    "yaarrrr",                        # informal elongation, common in chat text
    "supercalifragilisticexpialidocious",
    "whatsappkarna",                  # made-up compound, likely OOV
]
for w in test_words:
    ids = tokenizer.encode(w, add_special_tokens=False)
    raw_tokens = tokenizer.convert_ids_to_tokens(ids)
    print(f"{w!r} -> ids {ids} -> raw_tokens {raw_tokens}")
