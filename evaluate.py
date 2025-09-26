import random
import pandas as pd
import ipdb
import torch

from datasets import load_from_disk
from transformers import pipeline
from tqdm import tqdm
from sklearn.metrics import accuracy_score


def get_prompt_fn_by_id(task, prompt_id):

    if task == "2img":
        if prompt_id == "prompt_0":
            prompt_fn = lambda caption: f"Based on the two images, ordered from left to right, which follows the following caption, in the best way \"{caption}\"? Left or right? Answer in English."

        elif prompt_id == "prompt_1":
            prompt_fn = lambda caption: f"Given the two images, ordered from left to right, and the caption \"{caption}\", does the caption describe better the image in the right? Yes or no? Answer in English."

        elif prompt_id == "prompt_2":
            prompt_fn = lambda caption: f"Based on two images: left and right, which follows the caption \"{caption}\" ? Answer with 'left' or 'right' and nothing else."
            
    else:
        if prompt_id == "prompt_0":
            prompt_fn = lambda caption_1, caption_2: f"Based the 2 captions: left \"{caption_1}\" and right \"{caption_2}\"?, which follows the image? Left or right? Answer in English."
        
    return prompt_fn

def run_sample(model_pipe, item, lang, task, prompt_fn):
    coco_caption = item[f"coco_caption_{lang}"]
    inpaint_caption = item[f"inpaint_caption_{lang}"]

    label = random.sample([0, 1], 1)[0]
    caption = coco_caption if label == 0 else inpaint_caption
    image = item["coco_image"] if label == 0 else item["inpaint_image"]
    
    if task == "2img":
        prompt_text = prompt_fn(caption)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": item["coco_image"]},
                    {"type": "image", "image": item["inpaint_image"]},
                    {"type": "text", "text": prompt_text}
                ]
            }
        ]
    else:
        prompt_text = prompt_fn(coco_caption, inpaint_caption)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text}
                ]
            }
        ]

    outputs = model_pipe(
            text=messages,
            max_new_tokens=20,
            return_full_text=False
        )
    decoded_str = outputs[0]["generated_text"]
    decoded_str = decoded_str.replace("\n", " ").strip()
    return label, decoded_str

def get_gemma3n_answer_prompt_0(output_text):
    output_text = output_text.lower()
    if "left" not in output_text and "right" not in output_text:
        return -1
    elif "left" in output_text:
        return 0
    elif "right" in output_text:
        return 1
    
def get_gemma3n_answer_prompt_1(output_text):
    output_text = output_text.lower()
    if "yes" not in output_text and "no" not in output_text:
        return -1
    elif "no" in output_text:
        return 0
    elif "yes" in output_text:
        return 1
    
prompt_fn_map = {
    "prompt_0": get_gemma3n_answer_prompt_0,
    "prompt_1": get_gemma3n_answer_prompt_1,
    "prompt_2": get_gemma3n_answer_prompt_0
}

def main(multiling_ds_path, model_name, lang, task, prompt_id):
    dataset = load_from_disk(multiling_ds_path)
    model_pipe = pipeline("image-text-to-text", model=model_name, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto")

    prompt_fn = get_prompt_fn_by_id(task, prompt_id)
    answer_extractor_fn = prompt_fn_map.get(prompt_id)

    results = []
    labels = []
    preds = []

    for item in tqdm(dataset):
        label, output_text = run_sample(model_pipe, item, lang, task, prompt_fn)
        extracted_answer = answer_extractor_fn(output_text)

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
    pd.DataFrame(results).to_csv(f"data/eval.results.{model_name_path}.{lang}.{task}.{prompt_id}.csv", index=False)

    acc = accuracy_score(labels, preds)
    with open(f"data/eval.acc.{model_name_path}.{lang}.{task}.{prompt_id}.csv", "w") as fout:
        fout.write(f"Accuracy: {acc}\n")


if __name__ == "__main__":
    # stabilize the seed for the random label
    random.seed(42)

    if "snakemake" in globals():
        from snakemake.script import snakemake
        multiling_ds_path = snakemake.input[0]
        model_name = snakemake.params.model_name
        lang = snakemake.params.lang
        task = snakemake.params.task
        prompt_id = snakemake.params.prompt_id

    main(multiling_ds_path, model_name, lang, task, prompt_id)