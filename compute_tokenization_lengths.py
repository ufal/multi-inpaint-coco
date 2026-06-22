#!/usr/bin/env python3
"""
Compute tokenization lengths for each language and model in the dataset.
This script loads all models (encoder and decoder) and tokenizes captions
in each language to compute token counts.

This script is designed to be called from Snakemake via the script: directive.
"""

import json
import os

import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoProcessor
from open_clip import get_tokenizer
from tqdm import tqdm


from utils import (
    snapshot_map_generation as SNAPSHOT_MAP_GENERATION,
    snapshot_map_similarity as SNAPSHOT_MAP_ENCODER,
)


def load_tokenizer(model_name, model_type):
    """Load tokenizer for a given model."""
    if model_type == "encoder":
        model_path = SNAPSHOT_MAP_ENCODER[model_name]
        if model_name.startswith("nllb-siglip"):
            # For NLLB-CLIP models, use open_clip tokenizer
            try:
                tokenizer = get_tokenizer(model_path)
            except Exception as e:
                print(f"Warning: Could not load open_clip tokenizer for {model_name}: {e}")
                print(f"Trying with AutoTokenizer instead...")
                tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        else:
            # For SigLIP models, try AutoProcessor first, then AutoTokenizer
            try:
                processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
                tokenizer = processor.tokenizer
            except Exception as e:
                print(f"Warning: Could not load processor for {model_name}: {e}")
                print(f"Trying with AutoTokenizer instead...")
                tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    else:  # decoder
        model_path = SNAPSHOT_MAP_GENERATION[model_name]
        try:
            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            tokenizer = processor.tokenizer
        except Exception as e:
            print(f"Warning: Could not load processor for {model_name}: {e}")
            print(f"Trying with AutoTokenizer instead...")
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    return tokenizer


def count_tokens(tokenizer, text):
    """Count tokens in a text string."""
    try:
        # Handle different tokenizer types
        if hasattr(tokenizer, 'encode'):
            tokens = tokenizer.encode(text)
        elif hasattr(tokenizer, '__call__'):
            tokens = tokenizer(text)
        else:
            raise ValueError("Tokenizer doesn't have encode or __call__ method")

        # Get token count depending on return type
        if isinstance(tokens, list):
            return len(tokens)
        elif isinstance(tokens, torch.Tensor):
            # If the tokens[0] ends with a sequence of ones, remove them
            if tokens.ndim == 2 and tokens.shape[1] > 0 and (tokens[:, -1] == 1).all():
                for i, val in enumerate(tokens[0]):
                    if val == 1:
                        return i
            return tokens.shape[-1]
        elif hasattr(tokens, 'input_ids'):
            if isinstance(tokens.input_ids, list):
                return len(tokens.input_ids)
            else:
                return tokens.input_ids.shape[-1]
        else:
            return len(tokens)
    except Exception as e:
        print(f"Error counting tokens: {e}")
        return 0


def compute_tokenization_stats(dataset, tokenizer, model_name, language):
    """Compute tokenization statistics for a given model and language."""
    coco_lengths = []
    inpaint_lengths = []

    coco_caption_field = f"coco_caption_{language}"
    inpaint_caption_field = f"inpaint_caption_{language}"
    n_sig_overflow = 0

    for item in tqdm(dataset, desc=f"Tokenizing {model_name} - {language}", leave=False):
        coco_caption = item[coco_caption_field]
        inpaint_caption = item[inpaint_caption_field]

        coco_len = count_tokens(tokenizer, coco_caption)
        inpaint_len = count_tokens(tokenizer, inpaint_caption)

        if coco_len >= 64:
            n_sig_overflow += 1
        if inpaint_len >= 64:
            n_sig_overflow += 1

        coco_lengths.append(coco_len)
        inpaint_lengths.append(inpaint_len)

    return {
        "model": model_name,
        "language": language,
        "coco_mean": sum(coco_lengths) / len(coco_lengths),
        "coco_min": min(coco_lengths),
        "coco_max": max(coco_lengths),
        "coco_total": sum(coco_lengths),
        "inpaint_mean": sum(inpaint_lengths) / len(inpaint_lengths),
        "inpaint_min": min(inpaint_lengths),
        "inpaint_max": max(inpaint_lengths),
        "inpaint_total": sum(inpaint_lengths),
        "combined_mean": (sum(coco_lengths) + sum(inpaint_lengths)) / (len(coco_lengths) + len(inpaint_lengths)),
        "combined_total": sum(coco_lengths) + sum(inpaint_lengths),
        "num_samples": len(dataset),
        "num_sig_overflow": n_sig_overflow,
    }


if __name__ == "__main__":
    # When called from Snakemake, access via snakemake object
    dataset_path = snakemake.input[0]
    model_name = snakemake.params.model_name
    model_type = snakemake.params.model_type
    language = snakemake.params.lang
    output_path = snakemake.output[0]

    # Load dataset
    print(f"Loading dataset from {dataset_path}...")
    dataset = load_from_disk(dataset_path)

    # Load tokenizer
    print(f"Loading tokenizer for {model_name} ({model_type})...")
    tokenizer = load_tokenizer(model_name, model_type)

    # Compute statistics
    print(f"Computing tokenization statistics...")
    stats = compute_tokenization_stats(dataset, tokenizer, model_name, language)

    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"Results saved to {output_path}")
    print(f"Combined mean tokens: {stats['combined_mean']:.2f}")
