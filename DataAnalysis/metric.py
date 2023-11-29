import numpy as np

def calculate_precision_at_k(answer_df, pred_df, k):
    """
    Calculates the precision at k for a given set of answer and prediction dataframes.

    Args:
        answer_df (pd.DataFrame): DataFrame containing ground truth answers.
        pred_df (pd.DataFrame): DataFrame containing model predictions.
        k (int): Number of top predictions to consider for calculating precision.

    Returns:
        float: Average precision at k for the given data.
    """

    primary_col = answer_df.columns[0]  # Obtain the first column name for primary grouping
    secondary_col = answer_df.columns[1]  # Obtain the second column name for secondary values

    # Group both answer and prediction dataframes by the primary column.
    answer_dict = answer_df.groupby(primary_col)[secondary_col].apply(list).to_dict()
    pred_dict = pred_df.groupby(primary_col)[secondary_col].apply(list).to_dict()

    # Check for duplicates in the predicted secondary_col for each primary_col
    duplicated_preds = pred_df.groupby(primary_col).apply(lambda x: x[secondary_col].duplicated().any())
    if duplicated_preds.any():
        raise ValueError(f"Predicted {secondary_col} contains duplicates for some {primary_col}.")

    total_precision = 0
    total_count = 0

    for primary_key in answer_dict:
        if primary_key in pred_dict:
            true_positives = len(set(answer_dict[primary_key]).intersection(pred_dict[primary_key][:k]))
            # Calculate precision as the ratio of true positives to k.
            precision = true_positives / k
            total_precision += precision
            total_count += 1

    if total_count == 0:
        return 0  # Avoid division by zero

    return total_precision / total_count