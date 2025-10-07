import random
from unicodedata import normalize
import pandas as pd
import ipdb
import torch
import torch.nn.functional as F

from datasets import load_from_disk
from transformers import pipeline, AutoModel, AutoProcessor
from tqdm import tqdm

snapshot_map_generation = {
    "google_gemma-3-12b-it": "google/gemma-3-12b-it",
    "qwen-7b": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen3g-8b": "Qwen/Qwen3Guard-Gen-8B",
    "qwen_omni-7b": "Qwen/Qwen2.5-Omni-7B",
    "eurovllm-9b": "utter-project/EuroVLM-9B-Preview",
    "llama4_scout": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
}

snapshot_map_similarity = {
    "siglip2-base": "google/siglip2-base-patch16-224",
    "siglip2-large": "google/siglip2-large-patch16-256",
    "siglip2-so400m": "google/siglip2-so400m-patch16-256",
    "siglip2-giant": "google/siglip2-giant-opt-patch16-256"
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

def resize_positional_embeddings_2d(pos_embed, target_length):
    """
    Bilinearly resize positional embeddings following SiGLIP paper approach.
    
    Args:
        pos_embed: [seq_len, hidden_size] positional embeddings
        target_length: target sequence length
    
    Returns:
        resized_pos_embed: [target_length, hidden_size] resized positional embeddings
    """
    
    # Convert 1D sequence to 2D grid for bilinear interpolation
    seq_len, hidden_size = pos_embed.shape
    
    # Assume square grid for original embeddings (common in SiGLIP)
    orig_size = int(seq_len ** 0.5)
    if orig_size * orig_size != seq_len:
        # If not perfect square, find closest factorization
        # This handles cases where seq_len might not be exactly square
        orig_h = int(seq_len ** 0.5)
        orig_w = seq_len // orig_h
        if orig_h * orig_w != seq_len:
            orig_h = orig_w = int(seq_len ** 0.5)
    else:
        orig_h = orig_w = orig_size
    
    # Target grid dimensions
    target_size = int(target_length ** 0.5)
    if target_size * target_size != target_length:
        target_h = int(target_length ** 0.5)
        target_w = target_length // target_h
        if target_h * target_w != target_length:
            target_h = target_w = int(target_length ** 0.5)
    else:
        target_h = target_w = target_size
    
    # Reshape to 2D grid: [hidden_size, orig_h, orig_w]
    pos_embed_2d = pos_embed.T.view(hidden_size, orig_h, orig_w).unsqueeze(0)
    
    # Bilinear interpolation with anti-aliasing (recompute_scale_factor=False for stability)
    resized_pos_embed_2d = F.interpolate(
        pos_embed_2d,
        size=(target_h, target_w),
        mode='bilinear',
        align_corners=False,
        antialias=True  # Anti-aliasing as mentioned in the paper
    )
    
    # Reshape back to 1D: [target_length, hidden_size]
    resized_pos_embed = resized_pos_embed_2d.squeeze(0).view(hidden_size, -1).T
    
    # Ensure we have exactly target_length embeddings
    if resized_pos_embed.size(0) != target_length:
        # Fallback: simple linear interpolation if 2D approach doesn't work perfectly
        pos_embed_1d = pos_embed.T.unsqueeze(0).unsqueeze(0)  # [1, hidden_size, seq_len, 1]
        resized_pos_embed_1d = F.interpolate(
            pos_embed_1d,
            size=(target_length, 1),
            mode='bilinear',
            align_corners=False,
            antialias=True
        )
        resized_pos_embed = resized_pos_embed_1d.squeeze(0).squeeze(-1).T  # [target_length, hidden_size]
    
    return resized_pos_embed

def get_text_features_with_resized_pos_embeddings(model, input_ids, attention_mask):
    """
    Get text features with bilinearly resized positional embeddings following SiGLIP paper.
    """
    seq_length = input_ids.size(1)
    text_model = model.text_model
    embeddings_layer = text_model.embeddings
    
    # Get token embeddings
    inputs_embeds = embeddings_layer.token_embedding(input_ids)
    
    # Get original positional embeddings
    original_pos_embed = embeddings_layer.position_embedding.weight  # [orig_max_length, hidden_size]
    
    # Resize positional embeddings to match sequence length
    if seq_length > original_pos_embed.size(0):
        resized_pos_embed = resize_positional_embeddings_2d(original_pos_embed, seq_length)
        
        # Create position ids for the full sequence
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(input_ids.size(0), -1)
        
        # Apply resized positional embeddings
        position_embeddings = resized_pos_embed[position_ids]
    else:
        # Use standard positional embeddings for shorter sequences
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(input_ids.size(0), -1)
        position_embeddings = embeddings_layer.position_embedding(position_ids)
    
    # Combine token and position embeddings
    embeddings = inputs_embeds + position_embeddings
    
    encoder_outputs = text_model.encoder(
        inputs_embeds=embeddings,
        attention_mask=attention_mask.to(dtype=torch.bool),
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    )
    
    # Pool the output (first token for SiGLIP)
    last_hidden_state = encoder_outputs.last_hidden_state
    pooled_output = last_hidden_state[:, 0, :]
    
    # Apply text projection
    if hasattr(model, 'text_projection') and model.text_projection is not None:
        text_features = model.text_projection(pooled_output)
    else:
        text_features = pooled_output
    
    return text_features

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
        seq_length = inputs["input_ids"].size(1)
        if model.config.model_type == "siglip" and seq_length > 64:
            # Use bilinearly resized positional embeddings following SiGLIP paper
            texts_feats = get_text_features_with_resized_pos_embeddings(
                model, inputs["input_ids"], inputs["attention_mask"]
            )
        else:
            texts_feats = model.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])

        # Img encoder incl. projections
        image_feats = model.get_image_features(pixel_values=inputs["pixel_values"])

    if to_normalize:
        image_feats /= image_feats.norm(dim=-1, keepdim=True)
        texts_feats /= texts_feats.norm(dim=-1, keepdim=True)

    similarity = (01.0 * image_feats @ texts_feats.T).flatten()
    return label, similarity

def evaluate_by_similarity(dataset, model_name, lang, task):
    model = AutoModel.from_pretrained(snapshot_map_similarity[model_name], device_map="cuda").eval()
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