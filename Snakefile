import base64
import itertools
from io import BytesIO

HUNYAN_ONLY_LANGS = ["zh", "fr", "pt", "es", "tr", "ko", "th", "ms", "id", "tl",
    "hi", "pl", "nl", "km", "my", "fa", "gu", "ur", "te", "mr", "he", "bn", "ta",
    "bo", "kk", "mn", "ug", "yue"]

LANGUAGE_NAMES = {
    "am": "Amharic",
    "ar": "Arabic",
    "az": "Azerbaijani",
    "be": "Belarusian",
    "bg": "Bulgarian",
    "bn": "Bengali",
    "bo": "Tibetan",
    "cs": "Czech",
    "de": "German",
    "es": "Spanish",
    "fa": "Persian",
    "fr": "French",
    "gu": "Gujarati",
    "he": "Hebrew",
    "hi": "Hindi",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "kk": "Kazakh",
    "km": "Khmer",
    "ko": "Korean",
    "mn": "Mongolian",
    "mr": "Marathi",
    "ms": "Malay",
    "my": "Burmese",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sk": "Slovak",
    "ta": "Tamil",
    "te": "Telugu",
    "th": "Thai",
    "tl": "Filipino",
    "tr": "Turkish",
    "ug": "Uyghur",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "vi": "Vietnamese",
    "yue": "Cantonese",
    "zh": "Chinese",
}

LOCAL_DS_PATH = "data/inpaintCOCO_v2"
LOCAL_TRANSLATED_PATH = "data/inpaintCOCO_multilingual"

FIRST_LANGUAGES = ["cs", "ro"]
SECOND_LANGUAGES = ["de", "it", "az", "el", "ja", "sk", "vi", "uk", "ar", "ru", "hi"]
FINAL_LANGUAGES = FIRST_LANGUAGES + SECOND_LANGUAGES
TARGET_LANGUAGES = FINAL_LANGUAGES + ["en"]

LANGUAGE_STATS = {
    "tier": {
        "en": 5, "cs": 4, "ro": 3, "vi": 4, "de": 5,
        "sk": 3, "it": 4, "az": 1, "ja": 5, "uk": 3,
        "ar": 5, "el": 3, "hi": 4, "ru": 4,
    },
    "speakers_M": {
        "en": 380, "cs":  10.7, "ro":  24.3, "vi":  76.0, "de":  76.5,
        "sk":   6.0, "it":  64.8, "az":  24, "ja": 128.0, "uk":  26.9,
        "ar": 335.0, "el":  15.0, "hi": 341.0, "ru": 154.0,
    },
    "fineweb2_GB": {
        "en": 200_000_000.0, "cs":    221.5, "ro":    199.9, "vi":    343.4, "de":  1_654.8,
        "sk":         91.7, "it":    793.8, "az":     28.9, "ja":  1_650.2, "uk":    273.7,
        "ar":        315.2, "el":    238.4, "hi":    129.9, "ru":  6_400.1,
    },
    "wikipedia_articles": {
        "en": 6_900_000, "cs":   556_000, "ro":   434_000, "vi": 1_310_000, "de": 2_980_000,
        "sk":   259_000, "it": 1_862_000, "az":   215_000, "ja": 1_410_000, "uk": 1_330_000,
        "ar": 1_220_000, "el":   207_000, "hi": 1_620_000, "ru": 1_990_000,
    },
    "featural_sim_en": {
        "en": 1.000, "cs": 0.436, "ro": 0.537, "vi": 0.525, "de": 0.597,
        "sk": 0.436, "it": 0.524, "az": 0.451, "ja": 0.421, "uk": 0.541,
        "ar": 0.432, "el": 0.500, "hi": 0.513, "ru": 0.503,
    },
    "featural_sim_zh": {
        "en": 0.451, "cs": 0.385, "ro": 0.469, "vi": 0.423, "de": 0.450,
        "sk": 0.385, "it": 0.421, "az": 0.362, "ja": 0.459, "uk": 0.465,
        "ar": 0.348, "el": 0.403, "hi": 0.463, "ru": 0.469,
    },
}

GENERATIVE_MODEL_NAMES = [
    "google_gemma-3-12b-it",
    "qwen-7b",
    "qwen3-8b",
    "aya-8b",
    "jina",
    # "llama4_scout"
]

ENCODER_MODEL_NAMES = [
    "nllb-siglip-base",
    "nllb-siglip-large",
    "mexma-siglip2",
    "siglip2-base",
    "siglip2-large",
    "siglip2-so400m",
    "siglip2-giant"
]

rule all:
    input:
        "data/evaluation/strict_acc_bootstrap.csv",
        # "data/evaluation/eval.random_pairs.bilingual.csv",
        # "data/evaluation/eval.all.csv",
        # "data/evaluation/models_agreement.all.csv",
        # "data/evaluation/languages_agreement.all.csv",
        # "data/evaluation/eval.multilingual.2.csv",
        # "data/evaluation/mistakes_bilingual.csv"
        # "data/tokenization/tokenization_lengths.csv",
        # "data/evaluation/accuracy_correlations.csv",


# This rule loads the original InpaintCOCO dataset and applies the edits that
# were done during processing Czech and Romanian (some images deleted, some
# captions edited).
rule fix_original_dataset_and_export:
    input:
        "data/annotations/translated.edited.cs.docx.txt",  # Source of edited English Inpaint sentences
        "data/keep.singleshot.ids.txt"
    output:
        "data/texts.tsv",
        directory(LOCAL_DS_PATH),
    resources:
        mem="48G",
        cpus_per_task=8,
        tasks=1,
    run:
        import re
        import pandas as pd

        from datasets import load_dataset, Dataset
        dataset = load_dataset("phiyodr/inpaintCOCO")

        with open(input[0], "r") as f:
            new_content = f.read()
            inpaint_biling_sents = re.findall(r"Inpaint \w+ \d+:(.+)\n", new_content)
            inpaint_en_sents = inpaint_biling_sents[0::2]

        with open(input[1], "r") as f:
            keeping_ids = list(map(int, f.read().split("\n")))

        # Updating the new inpaint english values
        fixed_ds = dataset["test"]
        fixed_ds = fixed_ds.map(lambda example, idx: {**example, "inpaint_caption": inpaint_en_sents[idx]}, 
                                with_indices=True, batch_size=16, writer_batch_size=16)

        # Ignoring the bad samples
        fixed_ds = fixed_ds.select(keeping_ids)
        fixed_ds.save_to_disk(LOCAL_DS_PATH)

        # exporting the right texts.txt
        with open(output[0], "w") as f:
            f.write("coco_caption\tinpaint_caption\n")
            for item in fixed_ds:
                coco_caption = item["coco_caption"].strip()
                inpaint_caption = item["inpaint_caption"].strip()
                f.write(f"{coco_caption}\t{inpaint_caption}\n")


# Google translate is in a separate file because we use async calls that are
# tricky from withing Snakemake
rule google_translate:
    input:
        "data/texts.tsv"
    output:
        "data/translated.google.{lang}.tsv"
    shell:
        """
        python3 translate_google.py {wildcards.lang} > {output}
        """


def gpt4_translate(client, lang, sentence):
    response = client.chat.completions.create(
        model="gpt-4",  # You can also use "gpt-4-turbo" for the latest version
        messages=[
            {"role": "system", "content": "You are the professional translator in the world that translate sentences very accurately and in a way that sounds natural in the target languages."},
            {"role": "user", "content": f"Translate this into {LANGUAGE_NAMES[lang]}: {sentence}"},
        ],
        temperature=0.0,
        max_tokens=100
    )

    return response.choices[0].message.content.strip()


rule gpt4_translation:
    input:
        "data/texts.tsv"
    output:
        "data/translated.gpt4.{lang}.tsv"
    run:
        from openai import OpenAI

        with open("oai_api_key.txt", "r") as file:
            api_key = file.read().strip()

        client = OpenAI(api_key=api_key)

        with open(input[0], "r") as f_in, open(output[0], "w") as f_out:
            for line in f_in:
                coco_caption, inpaint_caption = line.strip().split("\t")
                if not inpaint_caption:
                    inpaint_caption = "?"
                translated_coco = gpt4_translate(client, wildcards.lang, coco_caption)
                translated_inpaint = gpt4_translate(client, wildcards.lang, inpaint_caption)
                print(f"{translated_coco}\t{translated_inpaint}", file=f_out)


def run_hunyuan_translate(tokenizer, model, language, sentence):
    messages = [
        {"role": "user",
        "content": f"Translate the following segment into {LANGUAGE_NAMES[language]}, without additional explanation. Keep the original sentence length and structure and be as concise as possible.\n\n{sentence}"},
    ]
    tokenized_chat = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt"
    )

    outputs = model.generate(tokenized_chat.to(model.device), max_new_tokens=500)
    output_text = tokenizer.decode(outputs[0][tokenized_chat.shape[1]:], skip_special_tokens=True)
    # Replace all newlines with spaces
    output_text = output_text.replace("\n", " ").strip()
    return output_text


rule hunyuan_translation:
    input:
        "data/texts.tsv"
    output:
        "data/translated.hunyuan.{lang}.tsv"
    resources:
        mem="48G",
        cpus_per_task=4,
        slurm_partition="gpu-troja,gpu-ms",
        slurm_extra="--gres=gpu:3 --constraint='gpuram40G|gpuram48G|gpuram64G|gpuram95G'"
    run:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from tqdm import tqdm

        model_name_or_path = "tencent/Hunyuan-MT-7B"
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto")

        with open(input[0], "r") as f_in, open(output[0], "w") as f_out:
            lines = f_in.readlines()
            for line in tqdm(lines, desc="Translating with Hunyuan"):
                coco_caption, inpaint_caption = line.strip().split("\t")
                if not inpaint_caption:
                    inpaint_caption = "?"
                translated_coco = run_hunyuan_translate(tokenizer, model, wildcards.lang, coco_caption)
                translated_inpaint = run_hunyuan_translate(tokenizer, model, wildcards.lang, inpaint_caption)
                print(f"{translated_coco}\t{translated_inpaint}", file=f_out)


def images_as_base64(pil_image):
    pil_image = pil_image.resize((150, 150))

    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    html = f"<img src='data:image/jpeg;base64,{img_str}'/>"
    return html


# This rule generates an HTML file that contains the entire dataset (including
# images in 150x150px base64-encoded format) and its machine translation.
# It is then used to generate editable docx.
rule generate_html:
    input:
        translated="data/translated.{system}.{lang}.tsv",
        dataset=LOCAL_DS_PATH,
    output:
        "data/translated.{system}.{lang}.html"
    resources:
        mem="4G",
        cpus_per_task=1,
    run:
        from datasets import load_from_disk

        f_out = open(output[0], "w")
        print("<html><body>", file=f_out)

        dataset = load_from_disk(input.dataset)
        with open(input.translated, "r") as f:
            # The [1:] is there because the translation file has a header line
            translation = [l.strip("\n").split("\t") for l in f.readlines()][1:]

        assert len(translation) == len(dataset)
        assert all(len(t) == 2 for t in translation), "Translation file should have two columns: COCO and Inpaint captions."

        for i, (item, (tgt_coco, tgt_inpaint)) in enumerate(
                zip(dataset, translation)):
            orig_img_html = images_as_base64(item["coco_image"])
            print(f"<p>{orig_img_html}</p>", file=f_out)
            print(f"<p><b>COCO en {i + 1}:</b> {item['coco_caption']}</p>", file=f_out)
            print(f"<p><b>COCO trans {i + 1}:</b> {tgt_coco}</p>", file=f_out)

            inpaint_img_html = images_as_base64(item["inpaint_image"])
            print(f"<p>{inpaint_img_html}</p>", file=f_out)
            print(f"<p><b>Inpaint en {i + 1}:</b> {item['inpaint_caption']}</p>", file=f_out)
            print(f"<p><b>Inpaint trans {i + 1}:</b> {tgt_inpaint}</p>", file=f_out)

        print("</body></html>", file=f_out)


# This generates editable docx with machine translation of the dataset that is
# later used for manual post-editing in Google doc.
rule convert_to_docx:
    input:
        "data/translated.{lang}.html"
    output:
        "data/translated.{lang}.docx"
    resources:
        mem="8G",
        cpus_per_task=4,
    shell:
        """
        pandoc -s {input} -o {output}
        """


# Get the list of deleted items from the original dataset and check if it was
# the same in Czech and Romanian.
rule check_dataset_edits:
    input:
        "data/annotations/translated.edited.cs.docx.txt",
        "data/annotations/translated.edited.ro.docx.txt",
    output:
        "data/ignore.ids.txt",
        "data/translated.final.cs.tsv",
        "data/translated.final.ro.tsv",
    script:
        "check_dataset_edits.py"


rule convert_postedited_to_tsv:
    input:
        "data/annotations/translated.edited.{lang}.docx.txt"
    output:
        "data/translated.final.{lang}.tsv"
    wildcard_constraints:
        lang="|".join(SECOND_LANGUAGES)
    shell:
        """
        grep trans {input} | sed 's/COCO trans [0-9]*: //;s/Inpaint trans [0-9]*: //' > {output}
        """

def update_item(item, idx, languages, translations):
        new_item = dict(item)
        new_item["coco_caption_en"] = new_item.pop("coco_caption")
        new_item["inpaint_caption_en"] = new_item.pop("inpaint_caption")
        for lang, trans in zip(languages, translations):
            new_item[f"coco_caption_{lang}"] = trans[idx * 2]
            new_item[f"inpaint_caption_{lang}"] = trans[idx * 2 + 1]
        return new_item

rule finalize_dataset:
    input:
        dataset=LOCAL_DS_PATH,
        translations=expand("data/translated.final.{lang}.tsv", lang=TARGET_LANGUAGES[:-1])
    output:
        directory(LOCAL_TRANSLATED_PATH)
    resources:
        mem="48G",
        cpus_per_task=4,
        slurm_partition="gpu-amd",
        slurm_extra="--gres=gpu:1 --constraint='gpuram64G'"
    run:
        from datasets import load_from_disk
        dataset = load_from_disk(input.dataset)

        with open("data/ignore.again.ids.txt", "r") as f:
            ignoring_ids = list(map(int, f.read().split("\n")))

        translations = []
        for trans_file in input.translations:
            with open(trans_file) as f:
                translations.append([line.strip() for line_id, line in enumerate(f) if (line_id // 2 + 1) not in ignoring_ids])

        languages = TARGET_LANGUAGES[:-1]
        dataset = dataset.map(update_item, with_indices=True, batch_size=16, writer_batch_size=16,
                              fn_kwargs={"languages": languages, "translations": translations})
        dataset.save_to_disk(LOCAL_TRANSLATED_PATH)

rule evaluate_dataset:
    input:
        LOCAL_TRANSLATED_PATH
    output:
        "data/evaluation/results.{model_name}.{lang}.{task}.{prompt_id}.csv"
    wildcard_constraints:
        model_name="|".join(GENERATIVE_MODEL_NAMES + ENCODER_MODEL_NAMES)
    params:
        model_name=lambda wildcards: wildcards.model_name,
        lang=lambda wildcards: wildcards.lang,
        task=lambda wildcards: wildcards.task,
        prompt_id=lambda wildcards: wildcards.prompt_id
    resources:
        mem="64G",
        cpus_per_task=4,
        slurm_partition="gpu-amd",
        slurm_extra="--gres=gpu:1 --constraint='gpuram64G'"
    script:
        "evaluate.py"


rule gather_evals:
    input:
        expand(
            "data/evaluation/results.{model_name}.{lang}.{task}.{prompt_id}.csv",
            model_name=GENERATIVE_MODEL_NAMES, 
            lang=TARGET_LANGUAGES,
            task=["2img", "2txt"],
            prompt_id=["prompt_4"]
        ),
        expand(
            "data/evaluation/results.{model_name}.{lang}.{task}.{prompt_id}.csv",
            model_name=ENCODER_MODEL_NAMES, 
            lang=TARGET_LANGUAGES,
            task=["2img", "2txt"],
            prompt_id=["similarity"]
        )
    params:
        task="monolingual"
    output:
        "data/evaluation/eval.all.csv",
        "data/evaluation/strict_accuracy.csv",
        "data/evaluation/2img_accuracy.csv",
        "data/evaluation/2txt_accuracy.csv",
        "data/evaluation/image_order_mistakes.csv"
    script:
        "gather.py"


def get_multilingual_sets(languages, size=2):
    bilingual_sets = ["_".join(combination) for combination in itertools.permutations(languages, r=size)]
    return bilingual_sets

rule correlate_model_pairs_multilingual:
    input:
       expand(
            "data/evaluation/results.{model_name}.{lang}.{task}.{prompt_id}.csv",
            model_name=GENERATIVE_MODEL_NAMES, 
            lang=TARGET_LANGUAGES,
            task=["2img", "2txt"],
            prompt_id=["prompt_4"]
        ),
        expand(
            "data/evaluation/results.{model_name}.{lang}.{task}.{prompt_id}.csv",
            model_name=ENCODER_MODEL_NAMES, 
            lang=TARGET_LANGUAGES,
            task=["2img", "2txt"],
            prompt_id=["similarity"]
        )
    output:
        "data/evaluation/models_agreement.all.csv",
        "data/evaluation/languages_agreement.all.csv",
        "data/evaluation/models_corr.csv",
        "data/evaluation/language_corr.csv"
    params:
        model_names=ENCODER_MODEL_NAMES+GENERATIVE_MODEL_NAMES,
        languages=TARGET_LANGUAGES,
        tasks=["2img", "2txt"],
        prompt_id="prompt_4"
    resources:
        mem="16G",
        cpus_per_task=4,
    script:
        "corelate.py"


rule gather_multilingual_evals:
    input:
        expand(
            "data/evaluation/results.{model_name}.{lang_set}.2txt.{prompt_id}.csv",
            model_name=GENERATIVE_MODEL_NAMES, 
            lang_set=get_multilingual_sets(TARGET_LANGUAGES, 2),
            prompt_id=["prompt_4"]
        ),
        expand(
            "data/evaluation/results.{model_name}.{lang_set}.2txt.{prompt_id}.csv",
            model_name=ENCODER_MODEL_NAMES, 
            lang_set=get_multilingual_sets(TARGET_LANGUAGES, 2),
            prompt_id=["similarity"]
        )
    params:
        task="bilingual"
    output:
        "data/evaluation/eval.multilingual.2.csv",
        "data/evaluation/acc_bilingual.csv",
        "data/evaluation/mistakes_bilingual.csv"
    script:
        "gather.py"


# Rule to compute tokenization lengths for each model-language pair
rule compute_tokenization_length:
    input:
        LOCAL_TRANSLATED_PATH
    output:
        "data/tokenization/token_lengths.{model_name}.{model_type}.{lang}.json"
    params:
        model_name=lambda wildcards: wildcards.model_name,
        model_type=lambda wildcards: wildcards.model_type,
        lang=lambda wildcards: wildcards.lang
    resources:
        mem="16G",
        cpus_per_task=2,
    script:
        "compute_tokenization_lengths.py"


# Rule to gather all tokenization results into a single CSV file
rule gather_tokenization_lengths:
    input:
        expand(
            "data/tokenization/token_lengths.{model_name}.decoder.{lang}.json",
            model_name=GENERATIVE_MODEL_NAMES,
            lang=TARGET_LANGUAGES
        ),
        expand(
            "data/tokenization/token_lengths.{model_name}.encoder.{lang}.json",
            model_name=ENCODER_MODEL_NAMES,
            lang=TARGET_LANGUAGES
        )
    output:
        "data/tokenization/tokenization_lengths.csv"
    run:
        import json
        import pandas as pd
        
        all_results = []
        for json_file in input:
            with open(json_file, 'r') as f:
                data = json.load(f)
                all_results.append(data)
        
        df = pd.DataFrame(all_results)
        df.to_csv(output[0], index=False)
        print(f"Tokenization lengths saved to {output[0]}")
        print(f"Total entries: {len(df)}")


rule compute_accuracy_correlations:
    input:
        strict_accuracy="data/evaluation/strict_accuracy.csv",
        tokenization="data/tokenization/tokenization_lengths.csv"
    output:
        "data/evaluation/accuracy_correlations.csv"
    params:
        language_stats=LANGUAGE_STATS
    resources:
        mem="8G",
        cpus_per_task=1,
    run:
        from scipy.stats import pearsonr, spearmanr
        import pandas as pd
        
        # Load data
        strict_acc_df = pd.read_csv(input.strict_accuracy, index_col=0)
        tokenization_df = pd.read_csv(input.tokenization)
        
        # Pivot tokenization data to have model-language pairs
        tok_pivot = tokenization_df.pivot_table(
            index='language', 
            columns='model', 
            values='combined_mean'
        )
        
        # Prepare results storage
        results = []
        
        # Get all models from strict_accuracy (columns)
        models = strict_acc_df.columns
        
        for model in models:
            # Get strict accuracy for this model
            model_accuracy = strict_acc_df[model]
            
            # Filter to languages that have both accuracy and tokenization data
            if model in tok_pivot.columns:
                # Get tokenization data for this model
                model_tokenization = tok_pivot[model]
                
                # Find common languages
                common_langs = list(set(model_accuracy.index) & set(model_tokenization.index))
                
                if len(common_langs) > 2:  # Need at least 3 points for correlation
                    # Align data
                    acc_values = model_accuracy.loc[common_langs]
                    tok_values = model_tokenization.loc[common_langs]
                    
                    # Remove any NaN values
                    valid_mask = ~(acc_values.isna() | tok_values.isna())
                    acc_values = acc_values[valid_mask]
                    tok_values = tok_values[valid_mask]
                    
                    if len(acc_values) > 2:
                        # Compute correlation with tokenization
                        pearson_tok, p_pearson_tok = pearsonr(acc_values, tok_values)
                        spearman_tok, p_spearman_tok = spearmanr(acc_values, tok_values)
                        
                        results.append({
                            'model': model,
                            'property': 'combined_mean_tokens',
                            'pearson_r': pearson_tok,
                            'pearson_p': p_pearson_tok,
                            'spearman_r': spearman_tok,
                            'spearman_p': p_spearman_tok,
                            'n_languages': len(acc_values)
                        })
            
            # Correlate with each language property in LANGUAGE_STATS
            for prop_name, prop_values in params.language_stats.items():
                # Find languages that have both accuracy and this property
                common_langs = list(set(model_accuracy.index) & set(prop_values.keys()))
                
                if len(common_langs) > 2:  # Need at least 3 points for correlation
                    # Get accuracy values for common languages
                    acc_values = model_accuracy.loc[common_langs]
                    
                    # Get property values for common languages
                    prop_vals = pd.Series([prop_values[lang] for lang in common_langs], 
                                         index=common_langs)
                    
                    # Remove any NaN values
                    valid_mask = ~acc_values.isna()
                    acc_values = acc_values[valid_mask]
                    prop_vals = prop_vals[valid_mask]
                    
                    if len(acc_values) > 2:
                        # Compute correlations
                        pearson_r, p_pearson = pearsonr(acc_values, prop_vals)
                        spearman_r, p_spearman = spearmanr(acc_values, prop_vals)
                        
                        results.append({
                            'model': model,
                            'property': prop_name,
                            'pearson_r': pearson_r,
                            'pearson_p': p_pearson,
                            'spearman_r': spearman_r,
                            'spearman_p': p_spearman,
                            'n_languages': len(acc_values)
                        })
        
        # Convert to DataFrame and save
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(['model', 'property'])
        results_df.to_csv(output[0], index=False)
        
        print(f"Correlation results saved to {output[0]}")
        print(f"Total correlations computed: {len(results_df)}")
        print(f"\nModels analyzed: {results_df['model'].nunique()}")
        print(f"Properties analyzed: {results_df['property'].nunique()}")


rule create_random_permutation:
    output:
        "data/random_ids.txt"
    resources:
        mem="8G",
        cpus_per_task=1,
    run:
        import random
        new_ids = []

        for cur_id in range(953):
            new_id = cur_id
            while new_id == cur_id:
                new_id = random.choice(list(range(953)))
            new_ids.append(str(new_id))

        with open(output[0], "w") as f:
            f.write("\n".join(new_ids))
        

rule evaluate_random_pairs:
    input:
        LOCAL_TRANSLATED_PATH,
        "data/random_ids.txt"
    output:
        "data/evaluation/results_random_pairs.{model_name}.{lang}.{task}.{prompt_id}.csv"
    wildcard_constraints:
        model_name="|".join(GENERATIVE_MODEL_NAMES + ENCODER_MODEL_NAMES),
    params:
        model_name=lambda wildcards: wildcards.model_name,
        lang=lambda wildcards: wildcards.lang,
        task=lambda wildcards: wildcards.task,
        prompt_id=lambda wildcards: wildcards.prompt_id,
        selection="random_pairs"
    resources:
        mem="64G",
        cpus_per_task=4,
        slurm_partition="gpu-ms,gpu-troja",
        slurm_extra="--gres=gpu:3 --constraint='gpuram40G|gpuram48G' --exclude=tdll-3gpu3"
    script:
        "evaluate.py"


rule gather_random_pairs_evals:
    input:
        expand(
            "data/evaluation/results_random_pairs.{model_name}.{lang_set}.2txt.{prompt_id}.csv",
            model_name=GENERATIVE_MODEL_NAMES, 
            lang_set=get_multilingual_sets(TARGET_LANGUAGES, 2),
            prompt_id=["prompt_3"]
        ),
        expand(
            "data/evaluation/results_random_pairs.{model_name}.{lang_set}.2txt.{prompt_id}.csv",
            model_name=ENCODER_MODEL_NAMES, 
            lang_set=get_multilingual_sets(TARGET_LANGUAGES, 2),
            prompt_id=["similarity"]
        )
    params:
        task="random_pairs"
    output:
        "data/evaluation/eval.random_pairs.bilingual.csv",
        "data/evaluation/acc_random_pairs_bilingual.csv",
        "data/evaluation/mistakes_random_pairs_bilingual.csv"
    script:
        "gather.py"


rule bootstrap_strict_accuracy:
    input:
        expand(
            "data/evaluation/results.{model_name}.{lang}.{task}.{prompt_id}.csv",
            model_name=GENERATIVE_MODEL_NAMES, 
            lang=TARGET_LANGUAGES,
            task=["2img", "2txt"],
            prompt_id=["prompt_3"]
        ),
        expand(
            "data/evaluation/results.{model_name}.{lang}.{task}.{prompt_id}.csv",
            model_name=ENCODER_MODEL_NAMES, 
            lang=TARGET_LANGUAGES,
            task=["2img", "2txt"],
            prompt_id=["similarity"]
        )
    params:
        task="monolingual"
    output:
        "data/evaluation/strict_acc_bootstrap.csv",
    script:
        "stat_test.py"