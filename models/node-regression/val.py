from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

def calculate_metrics(
    true, 
    pred, 
    verbose=False
) -> dict:
    """
    Calculate and optionally print various regression metrics.

    Parameters:
    true (array-like): Array of true values.
    pred (array-like): Array of predicted values.
    verbose (bool): If True, print the calculated metrics. Default is False.

    Returns:
    tuple: A tuple containing MSE, MAE, RMSE, MAPE, and R2 scores.
    """
    rmse = mean_squared_error(true, pred, squared=False)
    mse = mean_squared_error(true, pred, squared=True)
    mae = mean_absolute_error(true, pred)
    mape = mean_absolute_percentage_error(true, pred)
    r2 = r2_score(true, pred)
    
    if verbose:
        print(f'--- Regression Metrics ---\n'
              f'Mean Squared Error (MSE): {mse:.4f}\n'
              f'Mean Absolute Error (MAE): {mae:.4f}\n'
              f'Root Mean Squared Error (RMSE): {rmse:.4f}\n'
              f'Mean Absolute Percentage Error (MAPE): {mape:.4f}\n'
              f'R-squared (R2): {r2:.4f}')
        
    return {'mse': mse, 'mae': mae, 'rmse': rmse, 'mape': mape, 'r2': r2}