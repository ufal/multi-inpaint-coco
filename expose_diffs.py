import re
import ipdb


def extract_items_from_content(full_content):
    coco_biling_sents = re.findall(r"COCO \w+ \d+:(.+)\n", full_content)
    coco_en_sents = coco_biling_sents[0::2]
    coco_fe_sents = coco_biling_sents[1::2]

    inpaint_biling_sents = re.findall(r"Inpaint \w+ \d+:(.+)\n", full_content)
    inpaint_en_sents = inpaint_biling_sents[0::2]
    inpaint_fe_sents = inpaint_biling_sents[1::2]

    items = [
        {
            "coco_en": coco_en.strip(),
            "coco_fe": coco_fe.strip(),
            "inpaint_en": inpaint_en.strip(),
            "inpaint_fe": inpaint_fe.strip()
        }
        for coco_en, coco_fe, inpaint_en, inpaint_fe in zip(coco_en_sents, coco_fe_sents, inpaint_en_sents, inpaint_fe_sents)
    ]
    return items

def extract_ro_marked_ids(full_content):
    sample_ids = re.findall(r"COCO trans (\d+): .+\n+\#MARKED", full_content)
    sample_ids = list(map(int, sample_ids))
    return sample_ids

def extract_cs_marked_ids(full_content):
    sample_ids = re.findall(r"COCO trans (\d+): .+ DELETE\n", full_content)
    sample_ids = list(map(int, sample_ids))
    return sample_ids

def export_diffs(ro_items, ro_marked_ids, cs_items, cs_marked_ids):
    common_ids = list(set(ro_marked_ids) & set(cs_marked_ids))
    common_ids.sort()

    only_ro_ids = list(set(ro_marked_ids) - set(cs_marked_ids))
    only_ro_ids.sort()

    only_cs_ids = list(set(cs_marked_ids) - set(ro_marked_ids))
    only_cs_ids.sort()

    with open("data/ro_cs_diffs.md", "w", encoding="utf-8") as fout:
        fout.write(f"# Aggreed IDs for deletion ({len(common_ids)}):\n")
        for sample_id in common_ids:
            ro_item = ro_items[sample_id - 1]
            fout.write(f"COCO en {sample_id}: {ro_item['coco_en']}\n")
        fout.write("\n\n")

        fout.write(f"# Only detected by Andrei ({len(only_ro_ids)}):\n")
        for sample_id in only_ro_ids:
            ro_item = ro_items[sample_id - 1]
            fout.write(f"COCO en {sample_id}: {ro_item['coco_en']}\n")
        fout.write("\n\n")

        fout.write(f"# Only detected by Jindrich ({len(only_cs_ids)}):\n")
        for sample_id in only_cs_ids:
            cs_item = cs_items[sample_id - 1]
            fout.write(f"COCO en {sample_id}: {cs_item['coco_en']}\n")
        fout.write("\n\n")

def export_en_captions_diffs(ro_items, ro_marked_ids, cs_items, cs_marked_ids):
    ignored_ids = list(set(ro_marked_ids) | set(cs_marked_ids))
    ignored_ids.sort()

    with open("data/en_captions_diffs.md", "w", encoding="utf-8") as fout:
        for ro_item, cs_item in zip(ro_items, cs_items):
            if ro_item["coco_en"] != cs_item["coco_en"] or ro_item["inpaint_en"] != cs_item["inpaint_en"]:
                sample_id = ro_items.index(ro_item) + 1

                if sample_id not in ignored_ids:
                    if ro_item["coco_en"] != cs_item["coco_en"]:
                        fout.write(f"COCO en (RO) {sample_id}: {ro_item['coco_en']}\n")
                        fout.write(f"COCO en (CS) {sample_id}: {cs_item['coco_en']}\n")
                    else:
                        fout.write(f"Inpaint en (RO) {sample_id}: {ro_item['inpaint_en']}\n")
                        fout.write(f"Inpaint en (CS) {sample_id}: {cs_item['inpaint_en']}\n")
                    fout.write("\n")


def main():
    with open("data/translated.final.ro.docx.txt", "r", encoding="utf-8") as f:
        ro_content = f.read()
    ro_items = extract_items_from_content(ro_content)
    ro_marked_ids = extract_ro_marked_ids(ro_content)

    with open("data/translated.final.cs.docx.txt", "r", encoding="utf-8") as f:
        cs_content = f.read()
    cs_items = extract_items_from_content(cs_content)
    cs_marked_ids = extract_cs_marked_ids(cs_content)

    export_diffs(ro_items, ro_marked_ids, cs_items, cs_marked_ids)
    export_en_captions_diffs(ro_items, ro_marked_ids, cs_items, cs_marked_ids)

    with open("data/ignore.ids.txt", "w") as f:
        ignoring_ids = list(set(ro_marked_ids) | set(cs_marked_ids))
        ignoring_ids.sort()

        f.write("\n".join(map(str, ignoring_ids)))

if __name__ == "__main__":
    main()
