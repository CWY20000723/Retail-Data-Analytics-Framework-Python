import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from scipy import stats

class DataPreprocessor:

    @staticmethod
    def traditional_round(value, decimals=0):
        """Rounds a number using traditional rounding (round half up) and ensures int output if decimals=0."""
        
        # Handle NaN values
        if np.isnan(value):
            return 0  # Replace NaN with 0
        
        # Handle infinity values
        if np.isinf(value):
            return "∞" if value > 0 else "-∞"  # Return a string representation
        
        factor = 10 ** decimals
        adjusted_value = value * factor

        # Traditional rounding: Add 0.5 for positive, subtract 0.5 for negative
        if adjusted_value >= 0:
            rounded_value = np.floor(adjusted_value + 0.5)  # Round half up
        else:
            rounded_value = np.ceil(adjusted_value - 0.5)  # Round half down (away from zero)

        result = rounded_value / factor  # Scale back

        # Ensure integer output if decimals=0
        return int(result) if decimals == 0 else result
    
    @staticmethod
    def data_imputation(value, replacement=np.nan, method="Mean"):
        """
        Impute missing values using different methods: Mean, Median, or Mode.
        
        Parameters:
        - value (array-like): List or NumPy array containing numbers (with possible NaNs).
        - replacement (float, optional): Value to replace NaNs if all values are NaN.
        - method (str, optional): Imputation method: "Mean", "Median", or "Mode". Default is "Mean".
        
        Returns:
        - np.array: The array with NaNs replaced by the chosen method.
        """
        # Convert to NumPy array for easy processing
        arr = np.array(value, dtype=np.float64)

        # Handle empty array
        if arr.size == 0:
            return np.array([replacement])

        if method == "Mean":
            value = np.nanmean(arr) if np.any(~np.isnan(arr)) else replacement

        elif method == "Median":
            value = np.nanmedian(arr) if np.any(~np.isnan(arr)) else replacement

        elif method == "Mode":
            # Mode returns a tuple (mode value, count), we need only the first value
            mode_result = stats.mode(arr, nan_policy="omit")
            value = mode_result.mode[0] if mode_result.count[0] > 0 else replacement

        else:
            raise ValueError("Invalid method. Choose 'Mean', 'Median', or 'Mode'.")

        # Replace NaNs with the calculated value
        arr[np.isnan(arr)] = value

        return arr
    
    @staticmethod
    def data_normalization(value, method='max-min'):
        """
        Normalize data using 'max-min' scaling, 'z-score' standardization, or 'log' normalization.

        Parameters:
        - value (array-like): List or NumPy array containing numbers.
        - method (str, optional): Normalization method ('max-min', 'z-score', or 'log'). Default is 'max-min'.

        Returns:
        - np.array: The normalized array.
        """
        # Convert input to NumPy array
        arr = np.array(value, dtype=np.float64)

        if method == 'max-min':
            min_value = np.min(arr)
            max_value = np.max(arr)
            range_value = max_value - min_value

            # Handle division by zero (all values are the same)
            if range_value == 0:
                return np.zeros_like(arr)

            return (arr - min_value) / range_value  # Min-Max Normalization

        elif method == 'z-score':
            mean_value = np.mean(arr)
            std_dev = np.std(arr)

            # Handle zero standard deviation (all values are the same)
            if std_dev == 0:
                return np.zeros_like(arr)

            return (arr - mean_value) / std_dev  # Z-score Normalization

        elif method == 'log':
            # Handle zero and negative values by shifting
            shift_value = np.abs(np.min(arr)) + 1 if np.min(arr) <= 0 else 0
            arr_shifted = arr + shift_value  # Shift values to be strictly positive
            return np.log1p(arr_shifted)  # Log(1 + x) transformation

        else:
            raise ValueError("Invalid method. Choose 'max-min', 'z-score', or 'log'.")