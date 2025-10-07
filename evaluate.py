import random
from unicodedata import normalize
import pandas as pd
import ipdb
import torch

from datasets import load_from_disk
from transformers import pipeline, AutoModel, AutoProcessor
from tqdm import tqdm

snapshot_map_generation = {
    "google_gemma-3-12b-it": "google/gemma-3-12b-it",
    "qwen-7b": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen3g-8b": "Qwen/Qwen3Guard-Gen-8B",
    "qwen_omni-7b": "Qwen/Qwen2.5-Omni-7B",
    "eurovllm-9b": "utter-project/EuroVLM-9B-Preview",
    "llama4_scout": "meta-llama_Llama-4-Scout-17B-16E-Instruct",
}

snapshot_map_similarity = {
    # "nllb-clip-base": "visheratin/nllb-clip-base",
    "siglip2-base": "google/siglip2-base-patch16-224"
}


def get_prompt_fn_by_id(task, prompt_id):

    if task == "2img":
        if prompt_id == "prompt_0":
            prompt_fn = lambda caption: f"Based on the two images, ordered from left to right, which follows the following caption, in the best way \"{caption}\"? Left or right? Answer in English."

        elif prompt_id == "prompt_1":
            prompt_fn = lambda caption: f"Given the two images, ordered from left to right, and the caption \"{caption}\", does the caption describe better the image in the right? Yes or no? Answer in English."

        elif prompt_id == "prompt_2":
            prompt_fn = lambda caption: f"Based on two images, which follows the caption \"{caption}\" ? Answer with 'left' or 'right' and nothing else."
            
    else:
        if prompt_id == "prompt_0":
            prompt_fn = lambda caption_1, caption_2: f"Based the 2 captions: left \"{caption_1}\" and right \"{caption_2}\", which follows the image? Left or right? Answer in English."
        
        elif prompt_id == "prompt_1":
            prompt_fn = lambda caption_1, caption_2: f"Given the 2 captions: left \"{caption_1}\" and right \"{caption_2}\", does the image follow better the image in the right? Yes or no? Answer in English."

        elif prompt_id == "prompt_2":
            prompt_fn = lambda caption_1, caption_2: f"Based the 2 captions: left \"{caption_1}\" and right \"{caption_2}\", which follows the image? Answer with 'left' or 'right' and nothing else."
        
    return prompt_fn

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

def run_generation_sample(model_pipe, item, lang, task, prompt_fn):
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

def evaluate_by_generation(dataset, model_name, lang, task, prompt_id):
    model_snapshot = snapshot_map_generation.get(model_name)
    model_pipe = pipeline("image-text-to-text", model=model_snapshot, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto")

    prompt_fn = get_prompt_fn_by_id(task, prompt_id)
    answer_extractor_fn = prompt_fn_map.get(prompt_id)

    results = []
    for item in tqdm(dataset):
        label, output_text = run_generation_sample(model_pipe, item, lang, task, prompt_fn)
        extracted_answer = answer_extractor_fn(output_text)

        results.append({
            "concept": item["concept"],
            "coco_caption": item["coco_caption"],
            "inpaint_caption": item["inpaint_caption"],
            "label": label,
            "extracted_answer": extracted_answer,
            "output_text": output_text
        })
    return results

# imported from hard-negative-mining github
def run_similarity_sample(item, model, processor, lang, task, to_normalize=True):
    coco_caption = item[f"coco_caption_{lang}"]
    inpaint_caption = item[f"inpaint_caption_{lang}"]

    label = random.sample([0, 1], 1)[0]
    caption = coco_caption if label == 0 else inpaint_caption
    image = item["coco_image"] if label == 0 else item["inpaint_image"]
    
    if task == "2img":
        images = [item["coco_image"], item["inpaint_image"]]
        captions = [caption]
    else:
        images = [image]
        captions = [coco_caption, inpaint_caption]

    inputs = processor(text=captions, images=images, return_tensors="pt", padding=True, return_attention_mask=True).to(model.device)
    with torch.no_grad():
        # Txt encoder incl. projections
        texts_feats = model.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])

        # Img encoder incl. projections
        image_feats = model.get_image_features(pixel_values=inputs["pixel_values"])

    if to_normalize:
        image_feats /= image_feats.norm(dim=-1, keepdim=True)
        texts_feats /= texts_feats.norm(dim=-1, keepdim=True)

    similarity = (01.0 * image_feats @ texts_feats.T).flatten()
    return label, similarity

def evaluate_by_similarity(dataset, model_name, lang, task):
    model = AutoModel.from_pretrained(snapshot_map_similarity[model_name], device_map="cuda")
    processor = AutoProcessor.from_pretrained(snapshot_map_similarity[model_name])

    results = []
    for item in tqdm(dataset):
        label, similarity = run_similarity_sample(item, model, processor, lang, task)
        extracted_answer = torch.argmax(similarity).item()

        results.append({
            "concept": item["concept"],
            "coco_caption": item["coco_caption"],
            "inpaint_caption": item["inpaint_caption"],
            "label": label,
            "extracted_answer": extracted_answer,
            "similarity": similarity.cpu().tolist()
        })
    
    return results

def main(multiling_ds_path, model_name, lang, task, prompt_id):
    dataset = load_from_disk(multiling_ds_path)
    if model_name in snapshot_map_generation:
        results = evaluate_by_generation(dataset, model_name, lang, task, prompt_id)
    else:
        results = evaluate_by_similarity(dataset, model_name, lang, task)
    pd.DataFrame(results).to_csv(f"data/evaluation/results.{model_name}.{lang}.{task}.{prompt_id}.csv", index=False)


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