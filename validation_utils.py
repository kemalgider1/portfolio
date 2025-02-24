import pandas as pd
import numpy as np
import logging


class ValidationUtils:
    @staticmethod
    def validate_columns(df, required_columns, context=""):
        """Validate required columns exist in dataframe"""
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            missing_cols_str = ", ".join(missing_cols)
            error_msg = f"{context}: Missing required columns: {missing_cols_str}"
            logging.error(error_msg)
            raise ValueError(error_msg)
        return True

    @staticmethod
    def validate_non_empty(df, context=""):
        """Validate dataframe is not empty"""
        if df.empty:
            error_msg = f"{context}: Empty dataframe"
            logging.error(error_msg)
            raise ValueError(error_msg)
        return True

    @staticmethod
    def validate_location_mapping(df, context=""):
        """Validate Location/IATA mapping consistency"""
        try:
            # Find location and IATA columns (case-insensitive)
            location_col = next((col for col in df.columns if col.upper() == 'LOCATION'), None)
            iata_col = next((col for col in df.columns if col.upper() == 'IATA'), None)

            if not (location_col and iata_col):
                error_msg = f"{context}: Missing Location or IATA columns"
                logging.error(error_msg)
                raise ValueError(error_msg)

            # Check for null values
            null_locs = df[location_col].isna().sum()
            null_iatas = df[iata_col].isna().sum()

            if null_locs > 0 or null_iatas > 0:
                error_msg = f"{context}: Found null values - Location: {null_locs}, IATA: {null_iatas}"
                logging.error(error_msg)
                raise ValueError(error_msg)

            return True

        except Exception as e:
            error_msg = f"{context}: Error validating location mapping - {str(e)}"
            logging.error(error_msg)
            raise ValueError(error_msg)

    @staticmethod
    def standardize_columns(df, column_mappings):
        """Standardize column names using mappings"""
        df = df.copy()

        # Create rename dictionary
        rename_dict = {}
        for col in df.columns:
            # Find matching standard name (case-insensitive)
            standard_name = next(
                (std for std, _ in column_mappings.items()
                 if col.upper() == std.upper()),
                None
            )
            if standard_name:
                rename_dict[col] = column_mappings[standard_name]

        # Rename columns if needed
        if rename_dict:
            df = df.rename(columns=rename_dict)

        return df

    @staticmethod
    def validate_distribution(dist, context=""):
        """Validate distribution properties"""
        try:
            if dist.empty:
                error_msg = f"{context}: Empty distribution"
                logging.error(error_msg)
                return False

            # Check for negative values
            if (dist < 0).any():
                error_msg = f"{context}: Found negative values in distribution"
                logging.error(error_msg)
                return False

            # Check sum approximately equals 1
            total = dist.sum()
            if not np.isclose(total, 1.0, rtol=1e-05):
                error_msg = f"{context}: Distribution sum ({total}) not close to 1.0"
                logging.warning(error_msg)

            return True

        except Exception as e:
            error_msg = f"{context}: Error validating distribution - {str(e)}"
            logging.error(error_msg)
            return False

    @staticmethod
    def validate_numeric_columns(df, numeric_columns, context=""):
        """Validate numeric columns contain valid values"""
        try:
            for col in numeric_columns:
                if col not in df.columns:
                    continue

                # Check for non-numeric values
                non_numeric = df[col].apply(lambda x: not pd.api.types.is_numeric_dtype(type(x)))
                if non_numeric.any():
                    error_msg = f"{context}: Non-numeric values found in {col}"
                    logging.error(error_msg)
                    return False

                # Check for negative values where inappropriate
                if col.endswith(('Volume', 'Count', 'Pax')):
                    neg_vals = (df[col] < 0).sum()
                    if neg_vals > 0:
                        error_msg = f"{context}: Found {neg_vals} negative values in {col}"
                        logging.error(error_msg)
                        return False

            return True

        except Exception as e:
            error_msg = f"{context}: Error validating numeric columns - {str(e)}"
            logging.error(error_msg)
            return False