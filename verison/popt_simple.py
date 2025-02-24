import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import logging

logging.basicConfig(
    filename='portfolio_debug.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def load_data():
    try:
        mc_per_product = pd.read_csv('MC_per_Product_SQL.csv')
        df_vols = pd.read_csv('df_vols_query.csv')
        market_pmi = pd.read_csv('Market_Summary_PMI.csv')
        market_comp = pd.read_csv('Market_Summary_Comp.csv')

        logging.info("Data loaded successfully")
        return mc_per_product, df_vols, market_pmi, market_comp

    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        return None, None, None, None


def calculate_metrics(df_vols, mc_per_product, current_year=2024):
    try:
        # Merge financial data
        df = df_vols.merge(
            mc_per_product[['DF_Market', 'Location', 'CR_BrandId',
                            f'{current_year} MC', f'{current_year} NOR']],
            how='left',
            on=['DF_Market', 'Location', 'CR_BrandId']
        ).fillna(0)

        # Calculate basic metrics
        df['Volume_Share'] = df.groupby('Location')[f'{current_year} Volume'].transform(
            lambda x: x / x.sum()
        )

        print("Sample of processed data:")
        print(df[['Location', f'{current_year} Volume', 'Volume_Share']].head())

        return df

    except Exception as e:
        logging.error(f"Error calculating metrics: {str(e)}")
        return None


def main():
    # Load data
    mc_per_product, df_vols, market_pmi, market_comp = load_data()
    if mc_per_product is None:
        return

    # Calculate metrics
    results = calculate_metrics(df_vols, mc_per_product)
    if results is not None:
        print("Processing completed successfully")


if __name__ == "__main__":
    main()