import pandas as pd
import ipdb
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

from glob import glob
from sklearn.metrics import accuracy_score, confusion_matrix

GENERATIVE_MODEL_NAMES = [
    "google_gemma-3-12b-it",
    "qwen-7b",
    # "eurovllm-9b"
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

def plot_2d_language_agreement(df, prefix):
    languages = sorted(list(set(df["lang_1"].tolist()) | set(df["lang_2"].tolist())))
    lang_mat = np.ones((len(languages), len(languages)), dtype=np.float32)

    grouped_df = df.groupby(["task", "lang_1", "lang_2"])
    for key, group in grouped_df:
        task, lang_1, lang_2 = key
        mean_corr = group["cohen_kappa"].mean()

        idx_1 = languages.index(lang_1)
        idx_2 = languages.index(lang_2)
        
        if task == "2img":
            # above
            lang_mat[idx_1, idx_2] = mean_corr
        else:
            # below
            lang_mat[idx_2, idx_1] = mean_corr

    plt.figure(figsize=(10, 10))
    ax = sns.heatmap(lang_mat, xticklabels=languages, yticklabels=languages, annot=True, fmt=".2f", cmap="Blues")
    ax.xaxis.tick_top()

    fig_path = os.path.join("data", "evaluation", f"{prefix}.png")
    plt.savefig(fig_path)


def compute_partial_bilingual_agreement():
    # TODO: refactor this from filenames to csv file

    languages = ["cs", "ro", "de", "it", "az", "el", "ja", "sk", "vi", "uk", "ar", "en"]
    lang_mat = [[[] for _ in range(len(languages))] for _ in range(len(languages))]

    bilingual_filenames = glob("data/evaluation/results.*.csv")
    for filename in bilingual_filenames:
        metadata = os.path.basename(filename).split(".")
        langs = metadata[2]
        task = metadata[3]

        if len(langs.split("_")) != 2:
            continue
        lang_1, lang_2 = langs.split("_")
        idx_1 = languages.index(lang_1)
        idx_2 = languages.index(lang_2)
        
        df = pd.read_csv(filename)
        accuracy = accuracy_score(df['label'], df['extracted_answer'])
        
        # ISSUE: collect them as a list and then average
        if task == "2img":
            # above
            lang_mat[idx_1][idx_2].append(accuracy)
        else:
            # below
            lang_mat[idx_2][idx_1].append(accuracy)

    # Average the collected accuracy values
    for i in range(len(languages)):
        for j in range(len(languages)):
            if lang_mat[i][j]:
                lang_mat[i][j] = sum(lang_mat[i][j]) / len(lang_mat[i][j])
            else:
                lang_mat[i][j] = 1.0

    plt.figure(figsize=(10, 10))
    ax = sns.heatmap(lang_mat, xticklabels=languages, yticklabels=languages, annot=True, fmt=".2f", cmap="Blues")
    ax.xaxis.tick_top()

    fig_path = os.path.join("data", "evaluation", f"partial_bilingual_results.png")
    plt.savefig(fig_path)


def compute_fp_of_parity_samples(labels, answers, indices):
    labels_merged = []
    answers_merged = []
    for idx in indices:
        labels_merged += labels[idx::4]
        answers_merged += answers[idx::4]

    conf_merged = confusion_matrix(labels_merged, answers_merged)
    fp_merged = conf_merged[0][1]
    return fp_merged

def compute_fp_rates_inner(df):
    labels = df["label"].tolist()
    answers = df["extracted_answer"].tolist()
    
    # 0, 1 - first caption
    fp_01 = compute_fp_of_parity_samples(labels, answers, indices=[0, 1])
    # 2, 3 - second caption
    fp_23 = compute_fp_of_parity_samples(labels, answers, indices=[2, 3])
    fp_diff_captions = fp_01 - fp_23

    # 0, 2 - dataset order
    fp_02 = compute_fp_of_parity_samples(labels, answers, indices=[0, 2])
    # 1, 3 - swapped order
    fp_13 = compute_fp_of_parity_samples(labels, answers, indices=[1, 3])
    fp_diff_order = fp_02 - fp_13

    return fp_diff_captions, fp_diff_order

def analize_inner_results():
    # TODO: refactor this from filenames to csv file
    languages = ["cs", "ro", "de", "it", "az", "el", "ja", "sk", "vi", "uk", "ar", "en"]
    captions_mat = [[[] for _ in range(len(languages))] for _ in range(len(languages))]
    orders_mat = [[[] for _ in range(len(languages))] for _ in range(len(languages))]

    bilingual_filenames = glob("data/evaluation/results.*.csv")
    for filename in bilingual_filenames:
        metadata = os.path.basename(filename).split(".")
        langs = metadata[2]
        task = metadata[3]

        if len(langs.split("_")) != 2:
            continue

        lang_1, lang_2 = langs.split("_")
        idx_1 = languages.index(lang_1)
        idx_2 = languages.index(lang_2)
        
        df = pd.read_csv(filename)
        fp_diff_captions, fp_diff_order = compute_fp_rates_inner(df)
        
        # ISSUE: collect them as a list and then average
        if task == "2img":
            # above
            captions_mat[idx_1][idx_2].append(fp_diff_captions)
            orders_mat[idx_1][idx_2].append(fp_diff_order)
        else:
            # below
            captions_mat[idx_2][idx_1].append(fp_diff_captions)
            orders_mat[idx_2][idx_1].append(fp_diff_order)


    # Average the collected accuracy values
    for name, lang_mat in zip(["captions", "orders"], [captions_mat, orders_mat]):
        for i in range(len(languages)):
            for j in range(len(languages)):
                if lang_mat[i][j]:
                    lang_mat[i][j] = sum(lang_mat[i][j]) / len(lang_mat[i][j])
                else:
                    lang_mat[i][j] = 1.0

        plt.figure(figsize=(10, 10))
        ax = sns.heatmap(lang_mat, xticklabels=languages, yticklabels=languages, annot=True, fmt=".2f", cmap="Blues")
        ax.xaxis.tick_top()

        fig_path = os.path.join("data", "evaluation", f"partial_{name}_bilingual_results.png")
        plt.savefig(fig_path)

def main():
    agreement_path = "data/evaluation/languages_agreement.all.csv"
    df = pd.read_csv(agreement_path)
    plot_2d_language_agreement(df, prefix="avg_all_lang_agreement")

    mask = df["model"].isin(GENERATIVE_MODEL_NAMES)
    gen_df = df[mask].reset_index(drop=True)
    plot_2d_language_agreement(gen_df, prefix="avg_generative_lang_agreement")

    mask =  df["model"].isin(ENCODER_MODEL_NAMES)
    enc_df = df[mask].reset_index(drop=True)
    plot_2d_language_agreement(enc_df, prefix="avg_encoder_lang_agreement")

    compute_partial_bilingual_agreement()
    analize_inner_results()
    
    # TODO: compute everything with mask of ids: data/ignore.again.ids.txt

if __name__ == "__main__":
    main()
