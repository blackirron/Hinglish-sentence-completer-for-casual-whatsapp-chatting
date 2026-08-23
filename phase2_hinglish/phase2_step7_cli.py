"""
Phase 2, Step 7: Interactive CLI for the Hinglish word predictor / completer.
Same structure as the Phase 1 (Devanagari Hindi) CLI.
"""
from phase2_step6_predictor_module import predict_next_word, complete_sentence

def print_banner():
    print("=" * 60)
    print("     HINGLISH WORD PREDICTOR / SENTENCE COMPLETER")
    print("=" * 60)

def print_next_word_results(prompt, candidates):
    print(f"\nNext-word predictions for: '{prompt}'")
    if not candidates:
        print("  (no confident candidates found)")
        return
    for i, (word, prob) in enumerate(candidates, 1):
        print(f"  {i}. {word}  ({prob:.3f})")

def print_completion_result(prompt, completion):
    print(f"\nCompletion for: '{prompt}'")
    print(f"  {completion}")

def main():
    print_banner()
    while True:
        prompt = input("\nEnter a partial Hinglish sentence (or 'q' to quit): ").strip()
        if prompt.lower() == "q":
            print("Goodbye.")
            break
        if not prompt:
            print("Please enter some text.")
            continue

        print("\nChoose mode:")
        print("1. Predict next word")
        print("2. Complete the sentence")
        print("q. Quit")
        mode = input("> ").strip()

        if mode == "q":
            print("Goodbye.")
            break
        elif mode == "1":
            candidates = predict_next_word(prompt, k=5)
            print_next_word_results(prompt, candidates)
        elif mode == "2":
            completion = complete_sentence(prompt, max_sentences=3)
            print_completion_result(prompt, completion)
        else:
            print("Invalid choice, try again.")

if __name__ == "__main__":
    main()
