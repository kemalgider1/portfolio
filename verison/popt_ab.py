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
        pax_data = pd.read_csv('Pax_Nat.csv')
        similarity_file = pd.read_csv('similarity_file.csv')

        logging.info("Data loaded successfully")
        print("Data shapes:")
        print(f"Volume data: {df_vols.shape}")
        print(f"PMI market data: {market_pmi.shape}")
        print(f"Competitor data: {market_comp.shape}")

        return mc_per_product, df_vols, market_pmi, market_comp, pax_data, similarity_file

    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        return None, None, None, None, None, None


def calculate_category_a(df_vols, mc_per_product, current_year=2024):
    try:
        # Merge financial data
        df = df_vols.merge(
            mc_per_product[['DF_Market', 'Location', 'CR_BrandId',
                            f'{current_year} MC', f'{current_year} NOR']],
            how='left',
            on=['DF_Market', 'Location', 'CR_BrandId']
        ).fillna(0)

        # Calculate volume share
        df['Volume_Share'] = df.groupby('Location')[f'{current_year} Volume'].transform(
            lambda x: x / x.sum()
        )

        # Calculate flags
        df['Volume_Rank'] = df.groupby('Location')[f'{current_year} Volume'].rank(ascending=False)
        total_skus = df.groupby('Location')['SKU'].count()

        # Green flags (top 5%)
        df['Green_Flag'] = (df['Volume_Rank'] <= total_skus * 0.05) & (df['TMO'] == 'PMI')

        # Red flags (bottom 25%)
        df['Red_Flag'] = (df['Volume_Rank'] > total_skus * 0.75) & (df['TMO'] == 'PMI')

        # Calculate Category A scores
        scores = []
        for location in df['Location'].unique():
            loc_data = df[df['Location'] == location]
            green_count = loc_data['Green_Flag'].sum()
            red_count = loc_data['Red_Flag'].sum()
            total_count = len(loc_data)

            if total_count > 0:
                score = ((green_count - (2 * red_count)) / total_count) * 100
                scaled_score = round((score + 200) * (10 / 300), 2)
                scores.append({'Location': location, 'Cat_A': scaled_score})

        return pd.DataFrame(scores)

    except Exception as e:
        logging.error(f"Error calculating Category A: {str(e)}")
        return None


def calculate_category_b(market_pmi, market_comp):
    try:
        scores = []
        for location in market_pmi['Location'].unique():
            # Get PMI and competitor distributions
            pmi_dist = market_pmi[market_pmi['Location'] == location]['PMI_Seg_SKU']
            comp_dist = market_comp[market_comp['Location'] == location]['Comp_Seg_SKU']

            if len(pmi_dist) > 0 and len(comp_dist) > 0:
                # Calculate correlation
                corr = np.corrcoef(pmi_dist, comp_dist)[0, 1]
                r2 = corr ** 2 if not np.isnan(corr) else 0
                scores.append({'Location': location, 'Cat_B': round(r2 * 10, 2)})

        return pd.DataFrame(scores)

    except Exception as e:
        logging.error(f"Error calculating Category B: {str(e)}")
        return None


def main():
    # Load data
    mc_per_product, df_vols, market_pmi, market_comp, pax_data, similarity_file = load_data()
    if mc_per_product is None:
        return

    # Calculate Category A scores
    cat_a_scores = calculate_category_a(df_vols, mc_per_product)
    if cat_a_scores is not None:
        print("\nCategory A Scores (sample):")
        print(cat_a_scores.head())

    # Calculate Category B scores
    cat_b_scores = calculate_category_b(market_pmi, market_comp)
    if cat_b_scores is not None:
        print("\nCategory B Scores (sample):")
        print(cat_b_scores.head())


if __name__ == "__main__":
    main()