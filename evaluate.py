import random
import ipdb
import pandas as pd
import torch

from datasets import load_from_disk
from transformers import pipeline, AutoModel, AutoProcessor, AutoModelForCausalLM, Llama4ForConditionalGeneration
from tqdm import tqdm
from open_clip import create_model_from_pretrained, get_tokenizer
from utils import snapshot_map_generation, snapshot_map_similarity

def get_prompt_fn_by_id(task, prompt_id):

    if task == "2img":
        if prompt_id == "prompt_0":
            prompt_fn = lambda caption: f"Based on the two images, ordered from left to right, which follows the following caption, in the best way \"{caption}\"? Left or right? Answer in English."

        elif prompt_id == "prompt_1":
            prompt_fn = lambda caption: f"Given the two images, ordered from left to right, and the caption \"{caption}\", does the caption describe better the image in the right? Yes or no? Answer in English."

        elif prompt_id == "prompt_2":
            prompt_fn = lambda caption: f"Based on two images, which follows the caption \"{caption}\" ? Answer with 'left' or 'right' and nothing else."

        elif prompt_id == "prompt_3":
            prompt_fn = lambda caption: f"Based on two images, which follows the caption \"{caption}\" ? Answer with 'left' or 'right' and nothing else."
        
        elif prompt_id == "prompt_4":
            prompt_fn = lambda caption: f"À partir des deux images, qui suivent la description \"{caption}\" ? Répondez par 'gauche' ou 'droite' et rien d'autre."
            
    else:
        if prompt_id == "prompt_0":
            prompt_fn = lambda caption_1, caption_2: f"Based the 2 captions: left \"{caption_1}\" and right \"{caption_2}\", which follows the image? Left or right? Answer in English."
        
        elif prompt_id == "prompt_1":
            prompt_fn = lambda caption_1, caption_2: f"Given the 2 captions: left \"{caption_1}\" and right \"{caption_2}\", does the image follow better the image in the right? Yes or no? Answer in English."

        elif prompt_id == "prompt_2":
            prompt_fn = lambda caption_1, caption_2: f"Based the 2 captions: left \"{caption_1}\" and right \"{caption_2}\", which follows the image? Answer with 'left' or 'right' and nothing else."
        
        elif prompt_id == "prompt_3":
            prompt_fn = lambda caption_1, caption_2: f"Based the 2 captions: \n (left) \"{caption_1}\" \n (right) \"{caption_2}\" \n Which one follows the image? Answer with 'left' or 'right' and nothing else."
        
        elif prompt_id == "prompt_4":
            prompt_fn = lambda caption_1, caption_2: f"À partir des deux descriptions suivantes: \n (gauche) \"{caption_1}\" \n (droite) \"{caption_2}\" \n Laquelle correspond à l'image? Répondez par 'gauche' ou 'droite', et rien d'autre."

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

def get_gemma3n_answer_prompt_4(output_text):
    output_text = output_text.lower()
    if "gauche" not in output_text and "droite" not in output_text:
        return -1
    elif "gauche" in output_text:
        return 0
    elif "droite" in output_text:
        return 1
    
prompt_fn_map = {
    "prompt_0": get_gemma3n_answer_prompt_0,
    "prompt_1": get_gemma3n_answer_prompt_1,
    "prompt_2": get_gemma3n_answer_prompt_0,
    "prompt_3": get_gemma3n_answer_prompt_0,
    "prompt_4": get_gemma3n_answer_prompt_4,
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

def compute_task_specific_full_input(item, pair_item, lang, task):
    dec_langs = lang.split("_")
    if len(dec_langs) == 1:
        lang1 = lang2 = lang
    elif len(dec_langs) == 2:
        lang1, lang2 = dec_langs
    else:
        raise ValueError(f"Invalid language format: {lang}")
    
    coco_caption = item[f"coco_caption_{lang1}"]
    inpaint_caption = pair_item[f"inpaint_caption_{lang2}"]

    coco_image = item["coco_image"]
    inpaint_image = pair_item["inpaint_image"]

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

def compute_encoder_specific_input(item, pair_item, lang):
    dec_langs = lang.split("_")
    if len(dec_langs) == 1:
        lang1 = lang2 = lang
    elif len(dec_langs) == 2:
        lang1, lang2 = dec_langs
    else:
        raise ValueError(f"Invalid language format: {lang}")
    
    coco_caption = item[f"coco_caption_{lang1}"]
    inpaint_caption = pair_item[f"inpaint_caption_{lang2}"]

    coco_image = item["coco_image"]
    inpaint_image = pair_item["inpaint_image"]

    return [coco_caption, inpaint_caption], [coco_image, inpaint_image]

def apply_custom_pipeline(model_pipe, messages, images):
    processor, model = model_pipe

    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=[images] if images else None,
        padding="longest",
        return_tensors="pt",
        tokenize=True,
        return_dict=True,
    ).to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=20,
    )
    generated_ids = output[0][inputs["input_ids"].shape[-1] :]
    decoded_str = processor.batch_decode([generated_ids], skip_special_tokens=True)[0]
    return decoded_str


def run_generation_sample(model_pipe, model_name, item, pair_item, lang, task, prompt_fn):
    # label, captions, images = compute_task_specific_random_input(item, lang, task)
    labels, captions_list, images_list = compute_task_specific_full_input(item, pair_item, lang, task)

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

        if model_name.startswith("jina"):
            decoded_str = apply_custom_pipeline(model_pipe, messages, images)
        elif model_name == "llama4_scout":
            decoded_str = apply_custom_pipeline(model_pipe, messages, images)
        else:
            outputs = model_pipe(
                text=messages,
                max_new_tokens=20,
                return_full_text=False
            )
            decoded_str = outputs[0]["generated_text"]
        
        decoded_str = decoded_str.replace("\n", " ").strip()
        decoded_list.append(decoded_str)
    
    return labels, decoded_list, captions_list

def load_custom_pipeline(model_name):
    processor = AutoProcessor.from_pretrained(model_name, use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', torch_dtype=torch.bfloat16, trust_remote_code=True)
    return (processor, model)

def evaluate_by_generation(dataset, model_name, lang, task, prompt_id, selection):
    model_snapshot = snapshot_map_generation.get(model_name)
    
    if model_name.startswith("jina"):
        model_pipe = load_custom_pipeline(model_snapshot)
    elif model_name == "llama4_scout":
        processor = AutoProcessor.from_pretrained(model_snapshot)
        model = Llama4ForConditionalGeneration.from_pretrained(
            model_snapshot,
            attn_implementation="flex_attention",
            device_map="cuda",
            torch_dtype=torch.bfloat16,
        ).eval()
        model_pipe = (processor, model)
    else:
        model_pipe = pipeline("image-text-to-text", model=model_snapshot, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto")

    prompt_fn = get_prompt_fn_by_id(task, prompt_id)
    answer_extractor_fn = prompt_fn_map.get(prompt_id)

    if selection == "random_pairs":
        with open("data/random_ids.txt", "r") as fin:
            pair_ids = [int(line.strip()) for line in fin.read().split("\n") if line.strip()]
    else:
        pair_ids = list(range(len(dataset)))

    results = []
    for item, pair_id in tqdm(zip(dataset, pair_ids)):
        pair_item = dataset[pair_id]

        labels, output_texts, captions_list = run_generation_sample(model_pipe, model_name, item, pair_item, lang, task, prompt_fn)
        extracted_answers = [answer_extractor_fn(output_text) for output_text in output_texts]

        for label, extracted_answer, output_text, captions in zip(labels, extracted_answers, output_texts, captions_list):
            results.append({
                "concept": item["concept"],
                "coco_caption": captions[0],
                "inpaint_caption": captions[1] if len(captions) > 1 else captions[0],
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

    similarity = (1.0 * image_feats @ texts_feats.T)
    return similarity

def get_siglip_similarity(model, inputs):
    _, captions = inputs
    outputs = model(*inputs)
    
    similarity = torch.zeros((2, 2), dtype=torch.float32)
    for img_id, output_img in enumerate(outputs):
        for score_dict in output_img:
            txt_id = captions.index(score_dict["label"])
            similarity[txt_id][img_id] = score_dict["score"]
    
    return similarity

def get_mexma_similarity(model, inputs):
    image_logits, _ = model.get_logits(inputs["input_ids"], inputs["attention_mask"], inputs["pixel_values"])
    similarity = image_logits
    return similarity

def get_nllb_siglip_similarity(model, inputs):
    image_features, text_features, logit_scale_exp, logit_bias = model(inputs["pixel_values"], inputs["input_ids"])
    similarity = logit_scale_exp * image_features @ text_features.T + logit_bias
    return similarity


def run_similarity_efficient_sample(item, pair_item, model, model_name, processor, lang, task):
    captions, images = compute_encoder_specific_input(item, pair_item, lang)
    similarities = []
    
    # Lowercase is necessary for every SigLIP Text Encoder
    lower_captions = [caption.lower() for caption in captions]

    if model_name.startswith("nllb-siglip"):
        inputs = process_nllb_siglip(processor, lower_captions, images)
    elif model_name.startswith("siglip2"):
        inputs = [images, lower_captions]
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
    
    # always like this
    labels = [0, 1, 1, 0]
    
    if task == "2img":
        similarities += [
            similarity[:, 0],
            similarity[:, 0].flip(0),
            similarity[:, 1],
            similarity[:, 1].flip(0)
        ]
    else:
        similarities += [
            similarity[0, :],
            similarity[0, :].flip(0),
            similarity[1, :],
            similarity[1, :].flip(0)
        ]
    
    display_captions = [captions for _ in range(4)]
    return labels, similarities, display_captions

def evaluate_by_similarity(dataset, model_name, lang, task, selection):
    if model_name.startswith("nllb-siglip"):
        model, transform, tokenizer = import_nllb_siglip(model_name)
        processor = (transform, tokenizer)
    elif model_name.startswith("siglip2"):
        model = pipeline(model=snapshot_map_similarity[model_name], task="zero-shot-image-classification")
        processor = None
    else:
        model = AutoModel.from_pretrained(snapshot_map_similarity[model_name], device_map="cuda", trust_remote_code=True).eval()
        processor = AutoProcessor.from_pretrained(snapshot_map_similarity[model_name])

    if selection == "random_pairs":
        with open("data/random_ids.txt", "r") as fin:
            pair_ids = [int(line.strip()) for line in fin.read().split("\n")]
    else:
        pair_ids = list(range(len(dataset)))

    results = []
    for item, pair_id in tqdm(zip(dataset, pair_ids)):
        pair_item = dataset[pair_id]

        labels, similarities, caption_list = run_similarity_efficient_sample(item, pair_item, model, model_name, processor, lang, task)
        extracted_answers = [torch.argmax(similarity).item() for similarity in similarities]

        for label, extracted_answer, similarity, captions in zip(labels, extracted_answers, similarities, caption_list):
            results.append({
                "concept": item["concept"],
                "coco_caption": captions[0],
                "inpaint_caption": captions[1],
                "label": label,
                "extracted_answer": extracted_answer,
                "similarity": similarity.cpu().tolist()
            })
    
    return results

def main(multiling_ds_path, model_name, lang, task, prompt_id, selection):
    dataset = load_from_disk(multiling_ds_path)
    if model_name in snapshot_map_generation:
        results = evaluate_by_generation(dataset, model_name, lang, task, prompt_id, selection)
    else:
        results = evaluate_by_similarity(dataset, model_name, lang, task, selection)

    prefix = "_random_pairs." if selection == "random_pairs" else "."
    pd.DataFrame(results).to_csv(f"data/evaluation/results{prefix}{model_name}.{lang}.{task}.{prompt_id}.csv", index=False)


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
        selection = snakemake.params.selection if "selection" in snakemake.params.keys() else "normal"

    main(multiling_ds_path, model_name, lang, task, prompt_id, selection)