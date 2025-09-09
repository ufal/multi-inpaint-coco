import base64
from io import BytesIO

LANGUAGES = ["cs", "sk", "de", "ro", "it", "uk", "ru"]

LANGUAGE_NAMES = {
    "cs": "Czech",
    "sk": "Slovak",
    "de": "German",
    "ro": "Romanian",
    "it": "Italian",
    "uk": "Ukrainian",
    "ru": "Russian"
}

LOCAL_DS_PATH = "data/inpaintCOCO_v2"

rule all:
    input:
        expand("data/translated.google.{lang}.docx", lang=LANGUAGES),
        expand("data/translated.gpt4.{lang}.docx", lang=["ro", "cs"]),


# This rule loads the original InpaintCOCO dataset and applies the edits that
# were done during processing Czech and Romanian (some images deleted, some
# captions edited).
rule fix_original_dataset_and_export:
    input:
        "data/translated.final.cs.docx.txt", # Source of edited English Inpaint sentences
        "data/ignore.ids.txt"
    output:
        "data/texts.tsv"
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
        "content": f"Translate the following segment into {LANGUAGE_NAMES[language]}, without additional explanation.\n\n{sentence}"},
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
        slurm_extra="--gres=gpu:1 --constraint='gpuram48G|gpuram64G|gpuram95G'"
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
    run:
        from datasets import load_from_disk

        f_out = open(output[0], "w")
        print("<html><body>", file=f_out)

        dataset = load_from_disk(input.dataset)
        with open(input.translated, "r") as f:
            # The [1:] is there because the translation file has a header line
            translation = [l.strip().split("\t") for l in f.readlines()][1:]

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
    shell:
        """
        pandoc -s {input} -o {output}
        """


rule convert_postedited_to_tsv:
    input:
        "postedited.{lang}.txt"
    output:
        "final.{lang}.tsv"
    shell:
        """
        grep trans {input} | sed 's/COCO trans [0-9]*: //;s/Inpaint trans [0-9]*: //'| sed 'N;s/\n/\t/' > {output}
        """

# Get the list of deleted items from the original dataset and check if it was
# the same in Czech and Romanian.
rule check_dataset_edits:
    input:
        "data/translated.final.cs.docx.txt",
        "data/translated.final.ro.docx.txt"
    output:
        "data/ignore.ids.txt",
        directory(LOCAL_DS_PATH)
    script:
        "check_dataset_edits.py"
