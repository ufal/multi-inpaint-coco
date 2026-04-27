#!/usr/bin/env python3
"""
Compute Pearson and Spearman correlations between strict accuracy and:
1. Language properties from LANGUAGE_STATS
2. Average tokenization lengths (combined_mean)
"""

import pandas as pd
from scipy.stats import pearsonr, spearmanr
import sys


def main(strict_accuracy_file, tokenization_file, output_file, language_stats):
    """
    Compute correlations for each model.
    
    Args:
        strict_accuracy_file: Path to strict_accuracy.csv
        tokenization_file: Path to tokenization_lengths.csv
        output_file: Path to save correlation results
        language_stats: Dictionary of language properties
    """
    # Load data
    strict_acc_df = pd.read_csv(strict_accuracy_file, index_col=0)
    tokenization_df = pd.read_csv(tokenization_file)
    
    # Pivot tokenization data to have model-language pairs
    tok_pivot = tokenization_df.pivot_table(
        index='language', 
        columns='model', 
        values='combined_mean'
    )
    
    # Prepare results storage
    results = []
    
    # Get all models from strict_accuracy (columns)
    models = strict_acc_df.columns
    
    for model in models:
        # Get strict accuracy for this model
        model_accuracy = strict_acc_df[model]
        
        # Filter to languages that have both accuracy and tokenization data
        if model in tok_pivot.columns:
            # Get tokenization data for this model
            model_tokenization = tok_pivot[model]
            
            # Find common languages
            common_langs = list(set(model_accuracy.index) & set(model_tokenization.index))
            
            if len(common_langs) > 2:  # Need at least 3 points for correlation
                # Align data
                acc_values = model_accuracy.loc[common_langs]
                tok_values = model_tokenization.loc[common_langs]
                
                # Remove any NaN values
                valid_mask = ~(acc_values.isna() | tok_values.isna())
                acc_values = acc_values[valid_mask]
                tok_values = tok_values[valid_mask]
                
                if len(acc_values) > 2:
                    # Compute correlation with tokenization
                    pearson_tok, p_pearson_tok = pearsonr(acc_values, tok_values)
                    spearman_tok, p_spearman_tok = spearmanr(acc_values, tok_values)
                    
                    results.append({
                        'model': model,
                        'property': 'combined_mean_tokens',
                        'pearson_r': pearson_tok,
                        'pearson_p': p_pearson_tok,
                        'spearman_r': spearman_tok,
                        'spearman_p': p_spearman_tok,
                        'n_languages': len(acc_values)
                    })
        
        # Correlate with each language property in LANGUAGE_STATS
        for prop_name, prop_values in language_stats.items():
            # Find languages that have both accuracy and this property
            common_langs = list(set(model_accuracy.index) & set(prop_values.keys()))
            
            if len(common_langs) > 2:  # Need at least 3 points for correlation
                # Get accuracy values for common languages
                acc_values = model_accuracy.loc[common_langs]
                
                # Get property values for common languages
                prop_vals = pd.Series([prop_values[lang] for lang in common_langs], 
                                     index=common_langs)
                
                # Remove any NaN values
                valid_mask = ~acc_values.isna()
                acc_values = acc_values[valid_mask]
                prop_vals = prop_vals[valid_mask]
                
                if len(acc_values) > 2:
                    # Compute correlations
                    pearson_r, p_pearson = pearsonr(acc_values, prop_vals)
                    spearman_r, p_spearman = spearmanr(acc_values, prop_vals)
                    
                    results.append({
                        'model': model,
                        'property': prop_name,
                        'pearson_r': pearson_r,
                        'pearson_p': p_pearson,
                        'spearman_r': spearman_r,
                        'spearman_p': p_spearman,
                        'n_languages': len(acc_values)
                    })
    
    # Convert to DataFrame and save
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(['model', 'property'])
    results_df.to_csv(output_file, index=False)
    
    print(f"Correlation results saved to {output_file}")
    print(f"Total correlations computed: {len(results_df)}")
    print(f"\nModels analyzed: {results_df['model'].nunique()}")
    print(f"Properties analyzed: {results_df['property'].nunique()}")


if __name__ == "__main__":
    # This will be called from Snakemake
    # The LANGUAGE_STATS will be passed via snakemake params
    pass
