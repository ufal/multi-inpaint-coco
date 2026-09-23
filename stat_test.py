import pandas as pd
import scipy.stats as st
import numpy as np
import ipdb

from gather import collect_if_available, convert_and_export_dataframe

def compute_conjunction_and_bootstrap(filenames, output_all_file):
    collection = {}
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

        strict_conj = img_conj & txt_conj

        res = st.bootstrap((strict_conj,), lambda *args: np.mean(args[0]), vectorized=False, \
                           confidence_level=0.95, n_resamples=20, random_state=101, method='BCa')
        interval = res.confidence_interval

        middle = float((interval.low + interval.high) / 2)
        deviation = float((interval.high - interval.low) / 2)
        result = "+-".join([str(middle), str(deviation)])

        model_name, lang = filename_2img.split(".")[1:3]
        collect_if_available(collection, lang, model_name, result)

    convert_and_export_dataframe(collection, output_all_file, name="")

def main(filenames, output_all_file, task):
    if task == "monolingual":
        compute_conjunction_and_bootstrap(filenames, output_all_file)


if __name__ == "__main__":
    if "snakemake" in globals():
        from snakemake.script import snakemake
        filenames = snakemake.input
        output_all_file = snakemake.output[0]
        task = snakemake.params.task
        
    main(filenames, output_all_file, task)