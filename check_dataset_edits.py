"""Get diff from the original InpaintCOCO dataset.

During the initial post-editing of Czech and Romanian translations, we decided
to throw away some examples and to post-edit some of the captions of inpainted
images. This script extracts what we post-edited based on exports from Google
Docs where we did the post-editing.
"""

import re
import ipdb


def parse_doc_content(full_content: str) -> list[dict[str, str]]:
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


def check_deletion_diff(ro_items, ro_marked_ids, cs_items, cs_marked_ids):
    import sys
    
    common_ids = list(set(ro_marked_ids) & set(cs_marked_ids))
    common_ids.sort()

    only_ro_ids = list(set(ro_marked_ids) - set(cs_marked_ids))
    only_ro_ids.sort()

    only_cs_ids = list(set(cs_marked_ids) - set(ro_marked_ids))
    only_cs_ids.sort()

    # Check if there are any differences
    if only_ro_ids or only_cs_ids:
        error_msg = "Differences found between Romanian and Czech marked IDs:\n"
        
        if only_ro_ids:
            error_msg += f"Only detected by Andrei ({len(only_ro_ids)}):\n"
            for sample_id in only_ro_ids:
                ro_item = ro_items[sample_id - 1]
                error_msg += f"COCO en {sample_id}: {ro_item['coco_en']}\n"
            error_msg += "\n"
        
        if only_cs_ids:
            error_msg += f"Only detected by Jindrich ({len(only_cs_ids)}):\n"
            for sample_id in only_cs_ids:
                cs_item = cs_items[sample_id - 1]
                error_msg += f"COCO en {sample_id}: {cs_item['coco_en']}\n"
            error_msg += "\n"
        
        print(error_msg, file=sys.stderr)

def write_final_items(items, marked_ids, fout):
    final_items = [item for idx, item in enumerate(items) if (idx + 1) not in marked_ids]
    for idx, item in enumerate(final_items):
        fout.write(f"COCO en {idx+1}: {item['coco_en']}\n" + \
                   f"COCO translated {idx+1}: {item['coco_fe']}\n\n\n" + \
                   f"Inpaint en {idx+1}: {item['inpaint_en']}\n" + \
                   f"Inpaint translated {idx+1}: {item['inpaint_fe']}\n\n\n")

def main_cs_ro():
    with open("data/annotations/translated.edited.ro.docx.txt", "r", encoding="utf-8") as f:
        ro_content = f.read()
    ro_items = parse_doc_content(ro_content)
    ro_marked_ids = extract_ro_marked_ids(ro_content)

    with open("data/annotations/translated.edited.cs.docx.txt", "r", encoding="utf-8") as f:
        cs_content = f.read()
    cs_items = parse_doc_content(cs_content)
    cs_marked_ids = extract_cs_marked_ids(cs_content)

    # just for cs-ro
    check_deletion_diff(ro_items, ro_marked_ids, cs_items, cs_marked_ids)

    with open("data/ignore.ids.txt", "w") as f:
        ignoring_ids = list(set(ro_marked_ids) | set(cs_marked_ids))
        ignoring_ids.sort()

        f.write("\n".join(map(str, ignoring_ids)))

    with open("data/annotations/translated.final.ro.txt", "w", encoding="utf-8") as f:
        write_final_items(ro_items, ignoring_ids, f)

    with open("data/annotations/translated.final.cs.txt", "w", encoding="utf-8") as f:
        write_final_items(cs_items, ignoring_ids, f)

if __name__ == "__main__":
    main_cs_ro()
