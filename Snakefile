import base64
import itertools
from io import BytesIO

LANGUAGES_FOR_ANNOTS = ["cs", "sk", "de", "ro", "it", "uk", "ru", "vi", "am", "ja", "ar"]
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
SECOND_LANGUAGES = ["de", "it", "az", "el", "ja", "sk", "vi", "uk","ar"]
FINAL_LANGUAGES = FIRST_LANGUAGES + SECOND_LANGUAGES
TARGET_LANGUAGES = FINAL_LANGUAGES + ["en"]

LANGUAGE_STATS = {
    "tier": {
        "en": 5, "cs": 4, "ro": 3, "vi": 4, "de": 5,
        "sk": 3, "it": 4, "az": 1, "ja": 5, "uk": 3,
        "ar": 5, "el": 3, "hi": 4, "ru": 4,
    },
    "speakers_M": {
        "en": 753.4, "cs":  10.7, "ro":  24.3, "vi":  76.0, "de":  76.5,
        "sk":   6.0, "it":  64.8, "az":   9.2, "ja": 128.0, "uk":  26.9,
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
    "eurovllm-9b"
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
        "data/evaluation/eval.multilingual.2.csv"
        # "data/evaluation/eval.all.csv",
        # "data/evaluation/models_agreement.all.csv",
        # "data/evaluation/languages_agreement.all.csv",
        # expand("data/translated.hunyuan.{lng}.tsv", lng=HUNYAN_ONLY_LANGS),


# This rule loads the original InpaintCOCO dataset and applies the edits that
# were done during processing Czech and Romanian (some images deleted, some
# captions edited).
rule fix_original_dataset_and_export:
    input:
        "data/translated.edited.cs.docx.txt", # Source of edited English Inpaint sentences
        "data/ignore.ids.txt"
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
            ignoring_ids = list(map(int, f.read().split("\n")))

        # Updating the new inpaint english values
        fixed_ds = dataset["test"]
        fixed_ds = fixed_ds.map(lambda example, idx: {**example, "inpaint_caption": inpaint_en_sents[idx]}, 
                                with_indices=True, batch_size=16, writer_batch_size=16)

        # Ignoring the bad samples
        valid_indices = [i for i in range(len(fixed_ds)) if i not in ignoring_ids]
        fixed_ds = fixed_ds.select(valid_indices)

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

        translations = []
        for trans_file in input.translations:
            with open(trans_file) as f:
                translations.append([line.strip() for line in f])
        languages = TARGET_LANGUAGES[:-1]
        dataset = dataset.map(update_item, with_indices=True, batch_size=16, writer_batch_size=16,
                              fn_kwargs={"languages": languages, "translations": translations})
        dataset.save_to_disk(LOCAL_TRANSLATED_PATH)

rule evaluate_dataset:
    input:
        LOCAL_TRANSLATED_PATH
    output:
        "data/evaluation/results.{model_name}.{lang}.{task}.{prompt_id}.csv"
    params:
        model_name=lambda wildcards: wildcards.model_name,
        lang=lambda wildcards: wildcards.lang,
        task=lambda wildcards: wildcards.task,
        prompt_id=lambda wildcards: wildcards.prompt_id
    resources:
        mem="48G",
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
            prompt_id=["prompt_2"]
        ),
        expand(
            "data/evaluation/results.{model_name}.{lang}.{task}.{prompt_id}.csv",
            model_name=ENCODER_MODEL_NAMES, 
            lang=TARGET_LANGUAGES,
            task=["2img", "2txt"],
            prompt_id=["similarity"]
        )
    output:
        "data/evaluation/eval.all.csv"
    script:
        "gather.py"


def get_multilingual_sets(languages, size=2):
    bilingual_sets = ["_".join(combination) for combination in itertools.combinations(languages, r=size)]
    return bilingual_sets

rule correlate_model_pairs_multilingual:
    input:
       expand(
            "data/evaluation/results.{model_name}.{lang}.{task}.{prompt_id}.csv",
            model_name=GENERATIVE_MODEL_NAMES, 
            lang=TARGET_LANGUAGES,
            task=["2img", "2txt"],
            prompt_id=["prompt_2"]
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
        "data/evaluation/languages_agreement.all.csv"
    params:
        model_names=ENCODER_MODEL_NAMES+GENERATIVE_MODEL_NAMES,
        languages=TARGET_LANGUAGES,
        tasks=["2img", "2txt"],
        prompt_id="prompt_2"
    resources:
        mem="16G",
        cpus_per_task=4,
    script:
        "corelate.py"


rule gather_multilingual_evals:
    input:
        expand(
            "data/evaluation/results.{model_name}.{lang_set}.{task}.{prompt_id}.csv",
            model_name=GENERATIVE_MODEL_NAMES, 
            lang_set=get_multilingual_sets(TARGET_LANGUAGES, 2),
            task=["2img", "2txt"],
            prompt_id=["prompt_2"]
        ),
        expand(
            "data/evaluation/results.{model_name}.{lang_set}.{task}.{prompt_id}.csv",
            model_name=ENCODER_MODEL_NAMES, 
            lang_set=get_multilingual_sets(TARGET_LANGUAGES, 2),
            task=["2img", "2txt"],
            prompt_id=["similarity"]
        )
    output:
        "data/evaluation/eval.multilingual.2.csv"
    script:
        "gather.py"
