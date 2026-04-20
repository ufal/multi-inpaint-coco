import os

import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, confusion_matrix
from utils import all_languages


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

def convert_and_export_dataframe(ds_dict, output_all_file, name):
    df = pd.DataFrame(ds_dict).T.sort_index()
    output_conjunction_file = output_all_file.replace("eval.all", f"{name}")
    df.to_csv(output_conjunction_file, index=True)

def compute_bilingual_strict_acc(df):
    txt_ans = df["extracted_answer"].values
    txt_conj = (1 - txt_ans[0::4]) & txt_ans[1::4] & txt_ans[2::4] & (1 - txt_ans[3::4])
    # sim(i_coco, t_coco) > sim(i_coco, t_inp) ∧ sim(i_inp, t_inp) > sim(i_inp, t_coco)
    
    txt_acc = txt_conj.sum() / len(txt_conj)
    return txt_acc

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
        df_txt = pd.read_csv(filename_2txt)

        # labels are always 0, 1, 1, 0

        img_ans = df_img["extracted_answer"].values
        img_conj = (1 - img_ans[0::4]) & img_ans[1::4] & img_ans[2::4] & (1 - img_ans[3::4])
        # sim(i_coco, t_coco) > sim(i_inp, t_coco) ∧ sim(i_inp, t_inp) > sim(i_coco, t_coco)

        txt_ans = df_txt["extracted_answer"].values
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

    convert_and_export_dataframe(conjunction_all, output_all_file, "strict_accuracy")
    convert_and_export_dataframe(conjunction_img, output_all_file, "2img_accuracy")
    convert_and_export_dataframe(conjunction_txt, output_all_file, "2txt_accuracy")


def compute_image_order_mistakes_diff(df):
    """ Computes the mistakes difference for samples with images given and the dataset order and in the swapped order. 
    Labels are always 0, 1, 1, 0:
    - natural order: [0::4] # of 1s, [2::4] # of 0s --> natr_mistakes
    - swapped order: [1::4] # of 0s, [3::4] # of 1s --> swap_mistakes
    """
    answers = df["extracted_answer"].values

    natr_mistakes = answers[0::4].sum() + (1 - answers[2::4]).sum()
    swap_mistakes = (1 - answers[1::4]).sum() + answers[3::4].sum()

    mistakes_dif = (natr_mistakes - swap_mistakes) / len(answers[::4])
    return mistakes_dif

def compute_order_mistake_rates(filenames, output_all_file):
    """ Computes the difference of False Positive rates in 2img task, regarding the image order. Exports the numbers in a grid: models x language
    """
    fp_rates = {} 

    single_task_filenames = [file for file in filenames if "2img" in file]
    for filename in single_task_filenames:
        df = pd.read_csv(filename)
        fp_diff_rate = compute_image_order_mistakes_diff(df)
        
        params_list = os.path.basename(filename).split(".")
        model_name = params_list[1]
        lang = params_list[2]

        collect_if_available(fp_rates, lang, model_name, fp_diff_rate)

    convert_and_export_dataframe(fp_rates, output_all_file, "image_order_mistakes")

def compute_lang_pair_filename_mapping(filenames):
    lang_pair_filename = {}
    for filename in filenames:
        params_list = os.path.basename(filename).split(".")
        key = params_list[2]
        
        if key not in lang_pair_filename:
            lang_pair_filename[key] = []
        lang_pair_filename[key].append(filename)
    return lang_pair_filename


def compute_mistakes(df_dir, df_rev):
    """ Computing
    - df_dir - [0::4] # of 1s, [3::4] # of 1s (FP)
    - df_rev - [1::4] # of 0s, [2::4] # of 0s (FN)
    ---> .sum()
    """
    ans_dir = df_dir["extracted_answer"].values
    ans_rev = df_rev["extracted_answer"].values

    # number of 1s 
    num_dir = ans_dir[0::4].sum() + ans_dir[3::4].sum()
    
    # number of 0s
    num_rev = (1 - ans_rev[1::4]).sum() + (1 - ans_rev[2::4]).sum()
    
    confusions = (num_dir + num_rev) / len(ans_dir)
    return confusions


def compute_bilingual_pairwise_mistakes(lang_pair_filename, output_all_file):
    lang_metrics_mat = [[{} for _ in all_languages] for _ in all_languages]
    for lang_key, filenames in lang_pair_filename.items():

        lang_1, lang_2 = lang_key.split("_")
        idx_1 = all_languages.index(lang_1)
        idx_2 = all_languages.index(lang_2)

        for filename in filenames:
            model_name = os.path.basename(filename).split(".")[1]
            df_dir = pd.read_csv(filename)
            
            rev_filename = filename.replace(f"{lang_1}_{lang_2}", f"{lang_2}_{lang_1}")
            df_rev = pd.read_csv(rev_filename)

            if model_name not in lang_metrics_mat[idx_1][idx_2] and model_name not in lang_metrics_mat[idx_2][idx_1]:
                
                dir_mistakes = compute_mistakes(df_dir, df_rev)
                rev_mistakes = compute_mistakes(df_rev, df_dir)

                lang_metrics_mat[idx_1][idx_2].update({
                    model_name: dir_mistakes - rev_mistakes
                })
                
                lang_metrics_mat[idx_2][idx_1].update({
                    model_name: rev_mistakes - dir_mistakes
                })
                
    lang_mat = np.array([[np.mean(list(metrics.values())) if len(metrics.values()) > 0 else 1.0 for metrics in row] for row in lang_metrics_mat])           
    df = pd.DataFrame(lang_mat, index=all_languages, columns=all_languages)

    lang_mat_path = output_all_file.replace("eval.multilingual.2", "mistakes_bilingual")
    df.to_csv(lang_mat_path)

def compute_bilingual_pairwise_acc(lang_pair_filename, output_all_file):
    lang_metrics_mat = [[{} for _ in all_languages] for _ in all_languages]
    for lang_key, filenames in lang_pair_filename.items():

        lang_1, lang_2 = lang_key.split("_")
        idx_1 = all_languages.index(lang_1)
        idx_2 = all_languages.index(lang_2)

        for filename in filenames:
            model_name = os.path.basename(filename).split(".")[1]
            df = pd.read_csv(filename)
            
            lang_metrics_mat[idx_1][idx_2].update({
                model_name: compute_bilingual_strict_acc(df)
            })

    lang_mat = np.array([[np.mean(list(metrics.values())) if len(metrics.values()) > 0 else 1.0 for metrics in row] for row in lang_metrics_mat])        
    df = pd.DataFrame(lang_mat, index=all_languages, columns=all_languages)

    lang_mat_path = output_all_file.replace("eval.multilingual.2", "acc_bilingual")
    df.to_csv(lang_mat_path)

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
        compute_order_mistake_rates(filenames, output_all_file)

    elif task == "bilingual":
        lang_pair_filename = compute_lang_pair_filename_mapping(filenames)
        compute_bilingual_pairwise_acc(lang_pair_filename, output_all_file)
        compute_bilingual_pairwise_mistakes(lang_pair_filename, output_all_file)


if __name__ == "__main__":
    if "snakemake" in globals():
        from snakemake.script import snakemake
        filenames = snakemake.input
        output_all_file = snakemake.output[0]
        task = snakemake.params.task
        
    main(filenames, output_all_file, task)