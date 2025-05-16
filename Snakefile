import base64
from io import BytesIO

LANGUAGES = ["cs", "sk", "de", "ro", "it", "uk", "ru"]

rule all:
    input:
        expand("translated.{lang}.docx", lang=LANGUAGES)


rule get_english:
    output:
        "texts.tsv"
    run:
        from datasets import load_dataset
        dataset = load_dataset("phiyodr/inpaintCOCO")

        with open(output[0], "w") as f:
            f.write("coco_caption\tinpaint_caption\n")
            for item in dataset["test"]:
                coco_caption = item["coco_caption"].strip()
                inpaint_caption = item["inpaint_caption"].strip()
                if not inpaint_caption:
                    inpaint_caption = "?"
                f.write(f"{coco_caption}\t{inpaint_caption}\n")


rule google_translate:
    input:
        "texts.tsv"
    output:
        "translated.{lang}.tsv"
    shell:
        """
        python3 translate.py {input} {wildcards.lang} > {output}
        """


def images_as_base64(pil_image):
    pil_image = pil_image.resize((150, 150))

    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    html = f"<img src='data:image/jpeg;base64,{img_str}'/>"
    return html


rule generate_html:
    input:
        "translated.{lang}.tsv"
    output:
        "translated.{lang}.html"
    run:
        from datasets import load_dataset

        f_out = open(output[0], "w")
        print("<html><body>", file=f_out)

        dataset = load_dataset("phiyodr/inpaintCOCO")
        with open(input[0], "r") as f:
            translation = [l.strip().split("\t") for l in f.readlines()]

        for i, (item, (tgt_coco, tgt_inpaint)) in enumerate(
                zip(dataset["test"], translation)):
            orig_img_html = images_as_base64(item["coco_image"])
            print(f"<p>{orig_img_html}</p>", file=f_out)
            print(f"<p><b>COCO en {i + 1}:</b> {item['coco_caption']}</p>", file=f_out)
            print(f"<p><b>COCO trans {i + 1}:</b> {tgt_coco}</p>", file=f_out)

            inpaint_img_html = images_as_base64(item["inpaint_image"])
            print(f"<p>{inpaint_img_html}</p>", file=f_out)
            print(f"<p><b>Inpaint en {i + 1}:</b> {item['inpaint_caption']}</p>", file=f_out)
            print(f"<p><b>Inpaint trans {i + 1}:</b> {tgt_inpaint}</p>", file=f_out)

        print("</body></html>", file=f_out)


rule convert_to_docx:
    input:
        "translated.{lang}.html"
    output:
        "translated.{lang}.docx"
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
