import os

import ipdb
import pandas as pd

from sklearn.metrics import accuracy_score, confusion_matrix

all_languages = ["cs", "ro", "de", "it", "az", "el", "ja", "sk", "vi", "uk", "ar", "en", "hi", "ru"]

def compute_accuracy_and_uncertainty_by_concept(df):
    """ Computes the accuracy for each concept
    """
    metrics = {}
    for concept in df['concept'].unique():
        concept_mask = df['concept'] == concept
        concept_df = df[concept_mask]

        accuracy = accuracy_score(concept_df['label'], concept_df['extracted_answer'])
        metrics[f"{concept}_acc"] = accuracy

        uncertain_mask = concept_df['extracted_answer'] == -1
        uncertainty = uncertain_mask.sum() / len(uncertain_mask)
        metrics[f"{concept}_uncert"] = uncertainty
    return metrics

def compute_metrics(df, filename):
    """ Computes not only the accuracy overall, but also for each concept: entity, color, size.
    The uncertainty is the number of model outputs which are not left or right, but rather none, or blanc answers.
    """
    accuracy = accuracy_score(df['label'], df['extracted_answer'])
    uncertain_mask = df['extracted_answer'] == -1
    uncertainty = uncertain_mask.sum() / len(uncertain_mask)

    properties = {
        "filename": filename,
        "accuracy": accuracy,
        "uncertainty": uncertainty
    }
    metrics = compute_accuracy_and_uncertainty_by_concept(df)
    properties.update(metrics)
    return properties

def collect_if_available(ds_dict, lang, model_name, value):
    if lang not in ds_dict:
        ds_dict[lang] = {}
    ds_dict[lang][model_name] = value

def convert_and_export_dataframe(ds_dict, output_all_file , name):
    df = pd.DataFrame(ds_dict).T.sort_index()
    output_conjunction_file = output_all_file.replace("eval.", f"{name}.")
    df.to_csv(output_conjunction_file, index=True)

def compute_conjunction_strict(filenames, output_all_file):
    """ Computes the accuracy of the conjunction of results for the 2img and 2txt tasks, and exports the results in a grid: models x language
    """
    conjunction_all = {}
    conjunction_txt = {}
    conjunction_img = {}

    single_task_filenames = [file for file in filenames if "2img" in file]
    for filename_2img in single_task_filenames:
        df_img = pd.read_csv(filename_2img)

        filename_2txt = filename_2img.replace("2img", "2txt")
        df_2txt = pd.read_csv(filename_2txt)

        # labels are always 0, 1, 1, 0

        img_ans = df_img["extracted_answer"].values
        img_conj = (1 - img_ans[0::4]) & img_ans[1::4] & img_ans[2::4] & (1 - img_ans[3::4])
        # sim(i_coco, t_coco) > sim(i_inp, t_coco) ∧ sim(i_inp, t_inp) > sim(i_coco, t_coco)

        txt_ans = df_2txt["extracted_answer"].values
        txt_conj = (1 - txt_ans[0::4]) & txt_ans[1::4] & txt_ans[2::4] & (1 - txt_ans[3::4])
        # sim(i_coco, t_coco) > sim(i_coco, t_inp) ∧ sim(i_inp, t_inp) > sim(i_inp, t_coco)

        mask = img_conj & txt_conj

        img_acc = img_conj.sum() / len(img_conj)
        txt_acc = txt_conj.sum() / len(txt_conj)
        conjuction_acc = mask.sum() / len(mask)
        
        params_list = os.path.basename(filename_2img).split(".")
        model_name = params_list[1]
        lang = params_list[2]

        collect_if_available(conjunction_all, lang, model_name, conjuction_acc)
        collect_if_available(conjunction_img, lang, model_name, img_acc)
        collect_if_available(conjunction_txt, lang, model_name, txt_acc)

    convert_and_export_dataframe(conjunction_all, output_all_file, "conj_all")
    convert_and_export_dataframe(conjunction_img, output_all_file, "conj_img")
    convert_and_export_dataframe(conjunction_txt, output_all_file, "conj_txt")

def compute_fp_of_parity_samples(labels, answers, indices):
    """ Computes the False Positive rates for various samples with indices % 4
    """
    labels_merged = []
    answers_merged = []
    for idx in indices:
        labels_merged += labels[idx::4]
        answers_merged += answers[idx::4]

    conf_merged = confusion_matrix(labels_merged, answers_merged)
    fp_merged = conf_merged[0][1]
    return fp_merged

def compute_image_order_fp_diff(df):
    """ Computes the False Positive rates for samples with images given and the dataset order and in the swapped order, and then computes the difference. 
    """
    labels = df["label"].tolist()
    answers = df["extracted_answer"].tolist()

    fp_normal = compute_fp_of_parity_samples(labels, answers, indices=[0, 2])
    fp_swapped = compute_fp_of_parity_samples(labels, answers, indices=[1, 3])

    fp_diff_rate = (fp_normal - fp_swapped) / len(labels)
    return fp_diff_rate

def compute_order_fp_rates(filenames, output_all_file):
    """ Computes the difference of False Positive rates in 2img task, regarding the image order. Exports the numbers in a grid: models x language
    """
    fp_rates = {} 

    single_task_filenames = [file for file in filenames if "2img" in file]
    for filename in single_task_filenames:
        df = pd.read_csv(filename)
        fp_diff_rate = compute_image_order_fp_diff(df)
        
        params_list = os.path.basename(filename).split(".")
        model_name = params_list[1]
        lang = params_list[2]

        collect_if_available(fp_rates, lang, model_name, fp_diff_rate)

    convert_and_export_dataframe(fp_rates, output_all_file, "order_fp_diff")


def compute_lang_acc_and_fp_rates(filenames, output_all_file):
    pass

def main(filenames, output_all_file, task):
    gathered = []
    for filename in filenames:
        df = pd.read_csv(filename)

        properties = compute_metrics(df, filename)
        gathered.append(properties)

    gathered = pd.DataFrame(gathered)
    gathered.to_csv(output_all_file, index=False)

    if task == "monolingual":
        compute_conjunction_strict(filenames, output_all_file)
        compute_order_fp_rates(filenames, output_all_file)
    elif task == "bilingual":
        compute_lang_acc_and_fp_rates(filenames, output_all_file)


if __name__ == "__main__":
    if "snakemake" in globals():
        from snakemake.script import snakemake
        filenames = snakemake.input
        output_all_file = snakemake.output[0]
        task = snakemake.params.task
        
    main(filenames, output_all_file, task)