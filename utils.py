
all_languages = [
    "ar", 
    "az",
    "cs", 
    "de",
    "el",
    "en",
    "hi",
    "it", 
    "ja",
    "ro", 
    "ru",
    "sk", 
    "uk",
    "vi",
]

snapshot_map_generation = {
    "google_gemma-3-12b-it": "google/gemma-3-12b-it",
    "qwen-7b": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen3-8b": "Qwen/Qwen3-VL-8B-Instruct",
    "aya-8b": "CohereLabs/aya-vision-8b",
    "jina": "jinaai/jina-vlm",
    "eurovllm-9b": "utter-project/EuroVLM-9B-Preview",
    "llama4_scout": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
}

snapshot_map_similarity = {
    "nllb-siglip-base": "nllb-clip-base-siglip",
    "nllb-siglip-large": "nllb-clip-large-siglip",
    "mexma-siglip2": "visheratin/mexma-siglip2",
    "siglip2-base": "google/siglip2-base-patch16-224",
    "siglip2-large": "google/siglip2-large-patch16-256",
    "siglip2-so400m": "google/siglip2-so400m-patch16-256",
    "siglip2-giant": "google/siglip2-giant-opt-patch16-256",
}

generative_models = ["google_gemma-3-12b-it", "qwen-7b", "qwen3-8b", "aya-8b", "jina"]
all_models = generative_models + list(snapshot_map_similarity.keys())