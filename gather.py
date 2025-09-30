import ipdb
import pandas as pd

from sklearn.metrics import accuracy_score

def compute_accuracy_and_uncertainty_by_concept(df):
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

def main(filenames, output_file):
    gathered = []
    for filename in filenames:
        df = pd.read_csv(filename)

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
        
        gathered.append(properties)

    gathered = pd.DataFrame(gathered)
    gathered.to_csv(output_file, index=False)

if __name__ == "__main__":
    if "snakemake" in globals():
        from snakemake.script import snakemake
        filenames = snakemake.input
        output_file = snakemake.output[0]
        
    main(filenames, output_file)