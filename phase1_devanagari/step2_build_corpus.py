"""
Step 2 (v2): Build the fine-tuning corpus from OpenSubtitles Hindi data.
Downloads the raw Moses-format en-hi bitext directly from OPUS
(bypasses the HF `datasets` script-loading issue entirely) and
keeps only the Hindi side for our conversational corpus.
"""
import urllib.request
import zipfile
import re
import os

URL = "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/moses/en-hi.txt.zip"
ZIP_PATH = "en-hi.txt.zip"

if not os.path.exists(ZIP_PATH):
    print("Downloading OpenSubtitles en-hi (moses format) from OPUS...")
    urllib.request.urlretrieve(URL, ZIP_PATH)
    print("Download complete.")
else:
    print("Zip already downloaded, skipping.")

print("Extracting...")
with zipfile.ZipFile(ZIP_PATH, "r") as z:
    print("Files in archive:", z.namelist())
    z.extractall(".")

# The Hindi-side file is typically named OpenSubtitles.en-hi.hi
hi_file_candidates = [f for f in os.listdir(".") if f.endswith(".hi")]
if not hi_file_candidates:
    raise FileNotFoundError("Couldn't find the .hi file after extraction — check the printed file list above and adjust the filename.")
hi_file = hi_file_candidates[0]
print("Using Hindi file:", hi_file)

def clean_line(text):
    text = text.strip()
    if len(text) < 3 or len(text) > 200:
        return None
    text = re.sub(r"^[-–—]\s*", "", text)
    text = re.sub(r"\[.*?\]", "", text)
    text = text.strip()
    if not text:
        return None
    return text

lines = []
with open(hi_file, "r", encoding="utf-8") as f:
    for raw_line in f:
        cleaned = clean_line(raw_line)
        if cleaned:
            lines.append(cleaned)

print(f"Total cleaned lines: {len(lines)}")

lines = list(dict.fromkeys(lines))  # dedupe, preserve order
print(f"After dedup: {len(lines)}")

with open("hindi_conversational_corpus.txt", "w", encoding="utf-8") as f:
    for line in lines:
        f.write(line + "\n")

print("Saved to hindi_conversational_corpus.txt")
print("\nSample lines:")
for l in lines[:10]:
    print(" -", l)
