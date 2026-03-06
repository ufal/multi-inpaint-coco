import ipdb
import re

import nltk
nltk.download('punkt_tab')

from tqdm import tqdm

from datasets import load_dataset, Dataset
dataset = load_dataset("phiyodr/inpaintCOCO", split="test")

edited_file = "data/annotations/translated.edited.ro.docx.txt"
with open(edited_file, "r") as fin:
    content = fin.read()

inpaint_biling_sents = re.findall(r"Inpaint \w+ \d+: (.+)\n", content)
inpaint_en_sents = inpaint_biling_sents[0::2]

altered = 0

altered_words = 0
altered_chars = 0

for sample, inpaint_en in tqdm(zip(dataset, inpaint_en_sents)):

    inpaint_orig = sample["inpaint_caption"]
    if inpaint_orig.strip() != inpaint_en.strip():
        altered += 1

        original_words = nltk.word_tokenize(inpaint_orig)
        inpaint_en_words = nltk.word_tokenize(inpaint_en)

        altered_words += nltk.edit_distance(original_words, inpaint_en_words)
        altered_chars += nltk.edit_distance(inpaint_orig, inpaint_en)

print(f"Altered, inpaint sentences: {altered}")
print(f"Altered words: {altered_words}")
print(f"Altered chars: {altered_chars}")

