import random
import pandas as pd

from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from sklearn.metrics import accuracy_score


def run_sample(model, tokenizer, item, lang):
    coco_caption = item[f"coco_caption_{lang}"]
    inpaint_caption = item[f"inpaint_caption_{lang}"]

    label = random.sample([0, 1], 1)[0]
    caption = coco_caption if label == 0 else inpaint_caption
    
    # ISSUE: this is model specific
    image_token = tokenizer.boi_token + tokenizer.eoi_token

    messages = [
        {
        "role": "system",
        "content": [{"type": "text", "text": "You are a helpful assistant, which receives two images and a caption, and you need to decide which image the caption describes better."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "url": item["coco_image"]},
                {"type": "image", "url": item["inpaint_image"]},
                # {"type": "text", "text": f"Given the two images {image_token}{image_token}, which one follows the following caption, in the best way \"{caption}\"? Answer left or right."}
                {"type": "text", "text": f"Based on the two images {image_token}{image_token}, ordered from left to right, which follows the following caption, in the best way \"{caption}\"? Answer left or right."}
            ]
        }
    ]
    tokenized_chat = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt"
    )

    outputs = model.generate(tokenized_chat.to(model.device), max_new_tokens=10)
    output_text = tokenizer.decode(outputs[0][tokenized_chat.shape[1]:], skip_special_tokens=True)
    
    # Replace all newlines with spaces
    output_text = output_text.replace("\n", " ").strip()
    return label, output_text

def get_gemma3n_answer(output_text):
    output_text = output_text.lower()
    if "left" in output_text:
        return 0
    elif "right" in output_text:
        return 1
    else:
        return -1

def main(multiling_ds_path, model_name, lang):
    dataset = load_from_disk(multiling_ds_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

    results = []
    labels = []
    preds = []

    for item in tqdm(dataset):
        label, output_text = run_sample(model, tokenizer, item, lang)
        extracted_answer = get_gemma3n_answer(output_text)

        labels.append(label)
        preds.append(extracted_answer)

        results.append({
            "coco_caption": item["coco_caption"],
            "inpaint_caption": item["inpaint_caption"],
            "label": label,
            "extracted_answer": extracted_answer,
            "output_text": output_text
        })

    model_name_path = model_name.replace("/", "_")
    pd.DataFrame(results).to_csv(f"data/eval.results.{model_name_path}.{lang}.csv", index=False)

    acc = accuracy_score(labels, preds)
    with open(f"data/eval.acc.{model_name_path}.{lang}.csv", "w") as fout:
        fout.write(f"Accuracy: {acc}\n")


if __name__ == "__main__":
    # stabilize the seed for the random label
    random.seed(42)

    if "snakemake" in globals():
        from snakemake.script import snakemake
        multiling_ds_path = snakemake.input[0]
        model_name = snakemake.params.model_name
        lang = snakemake.params.lang

    main(multiling_ds_path, model_name, lang)