from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score, precision_score, recall_score, confusion_matrix
from loguru import logger as log

def calculate_classification_metrics(
    true, 
    pred, 
    model_name='Model',
    verbose=False
) -> dict:
    """
    Calculate and optionally print various classification metrics.

    Parameters:
    true (array-like): Array of true values.
    pred (array-like): Array of predicted values.
    model_name (str): Name of the model being evaluated.
    verbose (bool): If True, print the calculated metrics. Default is False.

    Returns:
    dict: A dictionary containing accuracy, MCC, precision, recall, TN, FP, FN, TP, and F1 scores.
    """
    f1 = f1_score(true, pred, average='binary')
    mcc = matthews_corrcoef(true, pred)
    accuracy = accuracy_score(true, pred)
    precision = precision_score(true, pred, average='binary')
    recall = recall_score(true, pred, average='binary')
    
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(true, pred).ravel()
    
    if verbose:
        log.info(f'--- Classification Metrics for {model_name} ---\n'
              f'Accuracy: {accuracy:.4f}\n'
              f'Matthews Correlation Coefficient (MCC): {mcc:.4f}\n'
              f'Precision (Macro): {precision:.4f}\n'
              f'Recall (Macro): {recall:.4f}\n'
              f'True Negatives (TN): {tn}\n'
              f'False Positives (FP): {fp}\n'
              f'False Negatives (FN): {fn}\n'
              f'True Positives (TP): {tp}\n'
              f'F1 Score (Macro): {f1:.4f}')
    
    response = {'model': model_name,
                'accuracy': accuracy,
                'mcc': mcc,
                'precision': precision,
                'recall': recall,
                'tn': tn,
                'fp': fp,
                'fn': fn,
                'tp': tp,
                'f1': f1
            }
    
    return response

