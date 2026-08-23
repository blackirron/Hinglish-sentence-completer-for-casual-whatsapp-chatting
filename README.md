# Hindi & Hinglish Next-Word Predictor / Sentence Completer

CS224N final project (custom project). Two fine-tuned GPT-2 language models for next-word prediction and sentence completion:

1. **`phase1_devanagari/`** — standard-script Hindi, fine-tuned on conversational subtitle dialogue.
2. **`phase2_hinglish/`** — romanized, code-mixed Hindi-English ("Hinglish"), fine-tuned on synthetic two-person WhatsApp-style conversation, aimed at eventual keyboard-suggestion use.

Full writeup with methodology, results, and analysis: [`report/report.pdf`](report/report.pdf).

## Results

| System | Base model | Val. loss | Perplexity |
|---|---|---|---|
| Devanagari (Hindi) | [`surajp/gpt2-hindi`](https://huggingface.co/surajp/gpt2-hindi) | 1.891 | 6.62 |
| Hinglish (code-mixed) | [`l3cube-pune/hing-gpt`](https://huggingface.co/l3cube-pune/hing-gpt) | 2.379 | 10.79 |

Both fine-tuned models show a clear qualitative shift from formal/broadcast register (the base models) toward casual conversational register (the fine-tuned models) — see the report for side-by-side examples.

## Repo structure

```
phase1_devanagari/
  step1_sanity_check.py       # load & sanity-check the base Hindi model
  step2_build_corpus.py       # build the fine-tuning corpus (OpenSubtitles Hindi)
  step3_tokenize_dataset.py   # tokenize + chunk for causal LM fine-tuning
  step4_finetune.py           # fine-tune with HF Trainer
  step4b_compare_models.py    # base vs fine-tuned comparison
  step5_predictor_module.py   # next-word prediction + sentence completion
  step6_cli.py                # interactive CLI

phase2_hinglish/
  phase2_step1_sanity_check.py
  phase2_step2_build_corpus.py           # build corpus (Hinglish-Everyday-Conversations-1M)
  phase2_step3_tokenize.py               # tokenize (handles missing pad/eos token)
  phase2_step4_finetune.py               # fine-tune (handles embedding resize)
  phase2_step4b_fix_config.py            # fix stale bos/eos token ids post-resize
  phase2_step4c_compare.py               # base vs fine-tuned comparison
  phase2_step5_check_tokenizer_convention.py   # diagnostic: word-boundary marking
  phase2_step5b_check_rare_words.py            # diagnostic: OOV word splitting
  phase2_step5c_check_decode_spacing.py        # diagnostic: decode() spacing behavior
  phase2_step6_predictor_module.py       # next-word prediction + sentence completion
  phase2_step7_cli.py                    # interactive CLI

report/
  report.pdf        # full writeup (ACL-style)
  report.tex         # LaTeX source
  references.bib      # bibliography
```

## Running it

Built and tested on Google Colab (free-tier T4 GPU). Each script is meant to be run in sequence, in the same Colab session (or with intermediate artifacts like the corpus/dataset/model checkpoint saved to disk between steps).

```bash
pip install -r requirements.txt
```

Then run the phase you want, in order (`step1` → `step6`/`step7`), either as separate `.py` files in the same working directory or pasted into sequential Colab cells.

## Known limitations

- **Devanagari next-word predictor** occasionally truncates longer multi-token words before they're grammatically complete (e.g. "संयुक्त" → "संयुक्"). Partially mitigated with a well-formedness check (a word can't validly end on a bare virama); a residual failure mode from greedy decoding remains — beam search would likely resolve it fully.
- **Hinglish tokenizer** has no word-boundary marking at all and is lossy for word boundaries at the encoding level (confirmed: "whatsappkarna" and "whatsapp karna" encode identically). The predictor works around this by treating each top-k token as a complete word directly, since the vocabulary covers most common words as single tokens — but rare/OOV words can occasionally produce a malformed split (e.g. "Lion" → "L ion").
- The Hinglish model's validation loss had not plateaued at 3 epochs (unlike the Hindi model) — likely has headroom for further fine-tuning.
- Full write-up of these and other findings, including an engineering post-mortem on tokenizer/config pitfalls hit along the way, is in the report.

## Acknowledgments

Base models: [`surajp/gpt2-hindi`](https://huggingface.co/surajp/gpt2-hindi), [`l3cube-pune/hing-gpt`](https://huggingface.co/l3cube-pune/hing-gpt) (L3Cube-HingCorpus). Fine-tuning data: OpenSubtitles (via OPUS), `Abhishekcr448/Hinglish-Everyday-Conversations-1M`.
