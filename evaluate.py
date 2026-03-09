import random
import pandas as pd
import torch

from datasets import load_from_disk
from transformers import pipeline, AutoModel, AutoProcessor
from tqdm import tqdm
from open_clip import create_model_from_pretrained, get_tokenizer

snapshot_map_generation = {
    "google_gemma-3-12b-it": "google/gemma-3-12b-it",
    "qwen-7b": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen3g-8b": "Qwen/Qwen3Guard-Gen-8B",
    # "qwen_omni-7b": "Qwen/Qwen2.5-Omni-7B", NOT WORKING YET
    "eurovllm-9b": "utter-project/EuroVLM-9B-Preview",
    "llama4_scout": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
}

snapshot_map_similarity = {
    "siglip2-base": "google/siglip2-base-patch16-224",
    "siglip2-large": "google/siglip2-large-patch16-256",
    "siglip2-so400m": "google/siglip2-so400m-patch16-256",
    "siglip2-giant": "google/siglip2-giant-opt-patch16-256",
    "mexma-siglip2": "visheratin/mexma-siglip2",
    "nllb-siglip-base": "nllb-clip-base-siglip",
    "nllb-siglip-large": "nllb-clip-large-siglip"
    # "nllb-clip-large": "visheratin/nllb-clip-large", NOT WORKING YET
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

def compute_task_specific_random_input(item, lang, task):
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

    return label, captions, images

def compute_task_specific_full_input(item, lang, task):
    dec_langs = lang.split("_")
    if len(dec_langs) == 1:
        lang1 = lang2 = lang
    elif len(dec_langs) == 2:
        lang1, lang2 = dec_langs
    else:
        raise ValueError(f"Invalid language format: {lang}")
    
    coco_caption = item[f"coco_caption_{lang1}"]
    inpaint_caption = item[f"inpaint_caption_{lang2}"]

    coco_image = item["coco_image"]
    inpaint_image = item["inpaint_image"]

    labels = []
    captions_list = []
    images_list = []

    if task == "2img":
        for label_id, caption in enumerate([coco_caption, inpaint_caption]):
            
            # natural order
            images_list.append([coco_image, inpaint_image])
            captions_list.append([caption])
            labels.append(label_id)
            
            # swapped order
            images_list.append([inpaint_image, coco_image])
            captions_list.append([caption])
            labels.append(1 - label_id)
            
    else:
        for label_id, image in enumerate([coco_image, inpaint_image]):
            
            # natural order
            captions_list.append([coco_caption, inpaint_caption])
            images_list.append([image])
            labels.append(label_id)
            
            # swapped order
            captions_list.append([inpaint_caption, coco_caption])
            images_list.append([image])
            labels.append(1 - label_id)

    return labels, captions_list, images_list


def run_generation_sample(model_pipe, item, lang, task, prompt_fn):
    # label, captions, images = compute_task_specific_random_input(item, lang, task)
    labels, captions_list, images_list = compute_task_specific_full_input(item, lang, task)

    decoded_list = []
    for captions, images in zip(captions_list, images_list):
        if task == "2img":
            prompt_text = prompt_fn(captions[0])
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": images[0]},
                        {"type": "image", "image": images[1]},
                        {"type": "text", "text": prompt_text}
                    ]
                }
            ]
        else:
            prompt_text = prompt_fn(captions[0], captions[1])
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": images[0]},
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
        decoded_list.append(decoded_str)
    
    return labels, decoded_list

def evaluate_by_generation(dataset, model_name, lang, task, prompt_id):
    model_snapshot = snapshot_map_generation.get(model_name)
    model_pipe = pipeline("image-text-to-text", model=model_snapshot, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto")

    prompt_fn = get_prompt_fn_by_id(task, prompt_id)
    answer_extractor_fn = prompt_fn_map.get(prompt_id)

    results = []
    for item in tqdm(dataset):
        labels, output_texts = run_generation_sample(model_pipe, item, lang, task, prompt_fn)
        extracted_answers = [answer_extractor_fn(output_text) for output_text in output_texts]

        for label, extracted_answer, output_text in zip(labels, extracted_answers, output_texts):
            results.append({
                "concept": item["concept"],
                "coco_caption": item["coco_caption"],
                "inpaint_caption": item["inpaint_caption"],
                "label": label,
                "extracted_answer": extracted_answer,
                "output_text": output_text
            })
    return results

def import_nllb_siglip(model_name):
    model, transform = create_model_from_pretrained(snapshot_map_similarity[model_name], "v1", device="cuda")
    tokenizer = get_tokenizer(snapshot_map_similarity[model_name])
    model = model.eval()
    return model, transform, tokenizer

def process_nllb_siglip(processor, captions, images):
    transform, tokenizer = processor
    inputs_ids = tokenizer(captions).to("cuda")
    pixel_values = torch.vstack([transform(img).unsqueeze(0).to("cuda") for img in images])

    return {
        "input_ids": inputs_ids,
        "pixel_values": pixel_values,
    }

def get_clip_similarity(model, inputs, to_normalize=True):   
    image_feats = model.get_image_features(pixel_values=inputs["pixel_values"])      
    texts_feats = model.get_text_features(input_ids=inputs["input_ids"])

    if to_normalize:
        image_feats /= image_feats.norm(dim=-1, keepdim=True)
        texts_feats /= texts_feats.norm(dim=-1, keepdim=True)

    similarity = (1.0 * image_feats @ texts_feats.T).flatten()
    return similarity

def get_siglip_similarity(model, inputs):
    outputs = model(**inputs)
    similarity = outputs.logits_per_image.flatten()
    return similarity

def get_mexma_similarity(model, inputs):
    image_logits, _ = model.get_logits(inputs["input_ids"], inputs["attention_mask"], inputs["pixel_values"])
    similarity = image_logits.flatten()
    return similarity

def get_nllb_siglip_similarity(model, inputs):
    image_features, text_features, logit_scale_exp, logit_bias = model(inputs["pixel_values"], inputs["input_ids"])
    similarity = logit_scale_exp * image_features @ text_features.T + logit_bias
    return similarity.flatten()


def run_similarity_sample(item, model, model_name, processor, lang, task):
    # label, captions, images = compute_task_specific_random_input(item, lang, task)
    labels, captions_list, images_list = compute_task_specific_full_input(item, lang, task)

    similarities = []
    for captions, images in zip(captions_list, images_list):
        if model_name.startswith("nllb-siglip"):
            inputs = process_nllb_siglip(processor, captions, images)
        elif model_name.startswith("siglip2"):
            inputs = processor(text=captions, images=images, return_tensors="pt", padding="max_length", max_length=64, truncation=True).to(model.device)
        else:
            inputs = processor(text=captions, images=images, return_tensors="pt", padding=True).to(model.device)

        # imported from hard-negative-mining github
        with torch.inference_mode():
            if model_name.startswith("mexma"):
                similarity = get_mexma_similarity(model, inputs)
            elif model_name.startswith("nllb-siglip"):
                similarity = get_nllb_siglip_similarity(model, inputs)
            elif model_name.startswith("siglip2"):
                similarity = get_siglip_similarity(model, inputs)
            elif model_name.startswith("clip"):
                similarity = get_clip_similarity(model, inputs)
        
        similarities.append(similarity)
    return labels, similarities

def evaluate_by_similarity(dataset, model_name, lang, task):
    if model_name.startswith("nllb-siglip"):
        model, transform, tokenizer = import_nllb_siglip(model_name)
        processor = (transform, tokenizer)
    else:
        model = AutoModel.from_pretrained(snapshot_map_similarity[model_name], device_map="cuda", trust_remote_code=True).eval()
        processor = AutoProcessor.from_pretrained(snapshot_map_similarity[model_name])

    results = []
    for item in tqdm(dataset):
        labels, similarities = run_similarity_sample(item, model, model_name, processor, lang, task)
        extracted_answers = [torch.argmax(similarity).item() for similarity in similarities]

        for label, extracted_answer, similarity in zip(labels, extracted_answers, similarities):
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