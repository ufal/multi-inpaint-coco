import pandas as pd
import numpy as np

from sklearn.metrics import cohen_kappa_score
from utils import all_languages

generative_models = ["google_gemma-3-12b-it", "qwen-7b", "qwen3-8b", "aya-8b", "jina"]

def compute_models_agreement(model_names, languages, tasks, prompt_id):
    correlations = []
    for task in tasks:
        for i, model_name_1 in enumerate(model_names):
            prompt_1 = prompt_id if model_name_1 in generative_models else "similarity"

            for model_name_2 in model_names[i+1:]:
                prompt_2 = prompt_id if model_name_2 in generative_models else "similarity"

                for lang in languages:
                    lang_res_0_path = f"data/evaluation/results.{model_name_1}.{lang}.{task}.{prompt_1}.csv"
                    lang_res_1_path = f"data/evaluation/results.{model_name_2}.{lang}.{task}.{prompt_2}.csv"

                    df_en = pd.read_csv(lang_res_0_path)
                    df_fe = pd.read_csv(lang_res_1_path)

                    ck_score = cohen_kappa_score(df_fe["extracted_answer"], df_en["extracted_answer"])
                    correlations.append({
                        "task": task,
                        "model_1": model_name_1,
                        "model_2": model_name_2,
                        "language": lang,
                        "cohen_kappa": ck_score,
                    })
    return correlations

def compute_languages_agreement(model_names, languages, tasks, prompt_id):
    correlations = []
    for task in tasks:
        for i, lang_1 in enumerate(languages):
            for lang_2 in languages[i+1:]:
                for model_name in model_names:
                    prompt = prompt_id if model_name in generative_models else "similarity"

                    lang_res_0_path = f"data/evaluation/results.{model_name}.{lang_1}.{task}.{prompt}.csv"
                    lang_res_1_path = f"data/evaluation/results.{model_name}.{lang_2}.{task}.{prompt}.csv"

                    df_en = pd.read_csv(lang_res_0_path)
                    df_fe = pd.read_csv(lang_res_1_path)

                    ck_score = cohen_kappa_score(df_fe["extracted_answer"], df_en["extracted_answer"])
                    correlations.append({
                        "task": task,
                        "model": model_name,
                        "lang_1": lang_1,
                        "lang_2": lang_2,
                        "cohen_kappa": ck_score,
                    })
    return correlations

def compute_and_export_model_agreement_2d(df, models_agg_file):
    all_models = list(set(df["model_1"].unique()) | set(df["model_2"].unique()))
    model_mat = np.ones((len(all_models), len(all_models)), dtype=np.float32)

    language_based_mat = {lang: np.ones((len(all_models), len(all_models)), dtype=np.float32) for lang in df["language"].unique()}
    grouped_df = df.groupby(["task", "model_1", "model_2"])
    
    for key, group in grouped_df:
        task, model_1, model_2 = key
        mean_corr = group["cohen_kappa"].mean()

        idx_1 = all_models.index(model_1)
        idx_2 = all_models.index(model_2)

        if idx_1 > idx_2:
            idx_1, idx_2 = idx_2, idx_1

        if task == "2img":
            # above
            model_mat[idx_1, idx_2] = mean_corr
            for _, row in group.iterrows():
                language_based_mat[row["language"]][idx_1, idx_2] = row["cohen_kappa"]
        else:
            # below
            model_mat[idx_2, idx_1] = mean_corr

            for _, row in group.iterrows():
                language_based_mat[row["language"]][idx_2, idx_1] = row["cohen_kappa"]
    
    df = pd.DataFrame(model_mat, index=all_models, columns=all_models)
    model_mat_path = models_agg_file.replace("models_agreement.all", "models_corr")
    df.to_csv(model_mat_path)

    for lang, mat in language_based_mat.items():
        df = pd.DataFrame(mat, index=all_models, columns=all_models)
        language_based_mat_path = models_agg_file.replace("models_agreement.all", f"models_corr.{lang}")
        df.to_csv(language_based_mat_path)

def compute_and_export_language_agreement_2d(df, languages_agg_file):
    lang_mat = np.ones((len(all_languages), len(all_languages)), dtype=np.float32)
    model_based_mat = {model_name: np.ones((len(all_languages), len(all_languages)), dtype=np.float32) for model_name in df["model"].unique()}

    grouped_df = df.groupby(["task", "lang_1", "lang_2"])
    
    for key, group in grouped_df:
        task, lang_1, lang_2 = key
        mean_corr = group["cohen_kappa"].mean()

        idx_1 = all_languages.index(lang_1)
        idx_2 = all_languages.index(lang_2)

        if idx_1 > idx_2:
            idx_1, idx_2 = idx_2, idx_1

        if task == "2img":
            # above
            lang_mat[idx_1, idx_2] = mean_corr
            for _, row in group.iterrows():
                model_based_mat[row["model"]][idx_1, idx_2] = row["cohen_kappa"]
        else:
            # below
            lang_mat[idx_2, idx_1] = mean_corr

            for _, row in group.iterrows():
                model_based_mat[row["model"]][idx_2, idx_1] = row["cohen_kappa"]
    
    df = pd.DataFrame(lang_mat, index=all_languages, columns=all_languages)
    lang_mat_path = languages_agg_file.replace("languages_agreement.all", "language_corr")
    df.to_csv(lang_mat_path)

    for model_name, mat in model_based_mat.items():
        df = pd.DataFrame(mat, index=all_languages, columns=all_languages)
        model_based_mat_path = languages_agg_file.replace("languages_agreement.all", f"language_corr.{model_name}")
        df.to_csv(model_based_mat_path)


def main(model_names, languages, tasks, prompt_id, models_agg_file, languages_agg_file):
    correlations = compute_models_agreement(model_names, languages, tasks, prompt_id)
    df = pd.DataFrame(correlations)
    df.to_csv(models_agg_file, index=False)

    compute_and_export_model_agreement_2d(df, models_agg_file)

    correlations = compute_languages_agreement(model_names, languages, tasks, prompt_id)
    df = pd.DataFrame(correlations)
    df.to_csv(languages_agg_file, index=False)

    compute_and_export_language_agreement_2d(df, languages_agg_file)

if __name__ == "__main__":
    
    if "snakemake" in globals():
        from snakemake.script import snakemake
        models_agg_file = snakemake.output[0]
        languages_agg_file = snakemake.output[1]
        model_names = snakemake.params.model_names
        languages = snakemake.params.languages
        tasks = snakemake.params.tasks
        prompt_id = snakemake.params.prompt_id

    main(model_names, languages, tasks, prompt_id, models_agg_file, languages_agg_file)