from category_a_scorer import CategoryAScorer
from category_b_scorer import CategoryBScorer
from category_c_scorer import CategoryCScorer
from category_d_scorer import CategoryDScorer
import pandas as pd
import logging
from datetime import datetime
import os

def __init__(self, config=None):
    """Initialize scorer with optional config"""
    self.config = config or {}
    self.current_year = self.config.get('current_year', 2024)

    self.cat_a_scorer = CategoryAScorer(self.current_year)
    self.cat_b_scorer = CategoryBScorer()
    self.cat_c_scorer = CategoryCScorer(self.current_year)
    self.cat_d_scorer = CategoryDScorer()

    logging.basicConfig(level=self.config.get('log_level', logging.INFO))

def validate_data_sources(self, data_sources):
    """Validate required data sources are present and not empty"""
    required_sources = [
        'df_vols',
        'mc_per_product',
        'market_pmi',
        'market_comp',
        'pax_data',
        'mrk_nat_map',
        'iata_location',
        'similarity_file'
    ]

    for source in required_sources:
        if source not in data_sources:
            raise ValueError(f"Missing required data source: {source}")
        if data_sources[source] is None or data_sources[source].empty:
            raise ValueError(f"Empty data source: {source}")

def integrate_scoring(df_vols, mc_per_product, market_pmi, market_comp,
                     pax_data, mrk_nat_map, similarity_file, iata_location):
    """
    Integrate all category scorers and combine results
    """
    try:
        # Initialize scorers
        cat_a_scorer = CategoryAScorer()
        cat_b_scorer = CategoryBScorer()
        cat_c_scorer = CategoryCScorer()
        cat_d_scorer = CategoryDScorer()

        # Calculate scores
        print("Calculating Category A scores...")
        cat_a_scores = cat_a_scorer.calculate_scores(df_vols, mc_per_product)

        print("Calculating Category B scores...")
        cat_b_scores = cat_b_scorer.calculate_scores(market_pmi, market_comp)

        print("Calculating Category C scores...")
        cat_c_scores = cat_c_scorer.calculate_scores(pax_data, df_vols, mrk_nat_map, iata_location)

        print("Calculating Category D scores...")
        # Pre-filter locations that exist in IATA mapping
        valid_locations = set(iata_location['Location'].unique())
        filtered_df_vols = df_vols[df_vols['Location'].isin(valid_locations)].copy()

        cat_d_scores = cat_d_scorer.calculate_scores(filtered_df_vols, similarity_file, iata_location)

        # Combine scores
        scores = pd.DataFrame({'Location': df_vols['Location'].unique()})
        scores = scores.merge(cat_a_scores[['Location', 'Cat_A']], how='left', on='Location')
        scores = scores.merge(cat_b_scores[['Location', 'Cat_B']], how='left', on='Location')
        scores = scores.merge(cat_c_scores[['Location', 'Cat_C']], how='left', on='Location')
        scores = scores.merge(cat_d_scores[['Location', 'Cat_D']], how='left', on='Location')

        # Fill missing values with 0
        scores = scores.fillna(0)

        # Calculate average score
        score_cols = ['Cat_A', 'Cat_B', 'Cat_C', 'Cat_D']
        scores['Average'] = scores[score_cols].mean(axis=1)

        return scores

    except Exception as e:
        logging.error(f"Error in score integration: {str(e)}")
        return None

def calculate_all_scores(self, data_sources):
    """Calculate scores for all categories"""
    try:
        scores = {}

        # Category A
        scores['cat_a'] = self.cat_a_scorer.calculate_scores(
            data_sources['df_vols'],
            data_sources['mc_per_product']
        )

        # Category B
        scores['cat_b'] = self.cat_b_scorer.calculate_scores(
            data_sources['market_pmi'],
            data_sources['market_comp']
        )

        # Category C
        scores['cat_c'] = self.cat_c_scorer.calculate_scores(
            data_sources['pax_data'],
            data_sources['df_vols'],
            data_sources['mrk_nat_map'],
            data_sources['iata_location']
        )

        # Category D
        scores['cat_d'] = self.cat_d_scorer.calculate_scores(
            data_sources['df_vols'],
            data_sources['similarity_file'],
            data_sources['iata_location']
        )

        return self.combine_scores(scores)

    except Exception as e:
        logging.error(f"Error calculating scores: {str(e)}")
        return None
def generate_scores(current_year=2024):
    """Generate all category scores and combine them into a final score dataframe"""

    # Load required data (customize these as needed for your environment)
    df_vols = pd.read_csv('df_vols_query.csv')
    mc_per_product = pd.read_csv('MC_per_Product_SQL.csv')

    # Use existing PMI and competitor market data
    market_pmi = pd.read_csv('Market_Summary_PMI.csv')
    market_comp = pd.read_csv('Market_Summary_Comp.csv')

    print("\nPMI data columns:", market_pmi.columns.tolist())
    print("Comp data columns:", market_comp.columns.tolist())
    print("\nPMI data sample:")
    print(market_pmi.head())

    # PAX data
    pax_data = pd.read_csv('Pax_Nat.csv')
    mrk_nat_map = pd.read_csv('mrk_nat_map_query.csv')
    iata_location = pd.read_csv('iata_location_query.csv')

    # Similarity file for Category D
    similarity_file = pd.read_csv('similarity_file.csv')

    # Calculate Category A scores
    print("Calculating Category A scores...")
    cat_a_scorer = CategoryAScorer(current_year=current_year)
    cat_a_scores = cat_a_scorer.calculate_scores(df_vols, mc_per_product)

    # Calculate Category B scores with 2023 methodology
    print("Calculating Category B scores...")
    cat_b_scorer = CategoryBScorer()
    cat_b_scores = cat_b_scorer.calculate_scores(market_pmi, market_comp)

    # Calculate Category C scores with 2023 methodology
    print("Calculating Category C scores...")
    cat_c_scorer = CategoryCScorer(current_year=current_year)
    cat_c_scores = cat_c_scorer.calculate_scores(pax_data, df_vols, mrk_nat_map, iata_location)

    # Calculate Category D scores with 2023 methodology
    print("Calculating Category D scores...")
    cat_d_scorer = CategoryDScorer()
    cat_d_scores = cat_d_scorer.calculate_scores(df_vols, similarity_file, iata_location)

    # Merge all scores
    print("Merging all category scores...")

    # Start with Category A scores
    all_scores = cat_a_scores.copy()

    # Add other categories
    for cat_scores, cat_col in [
        (cat_b_scores, 'Cat_B'),
        (cat_c_scores, 'Cat_C'),
        (cat_d_scores, 'Cat_D')
    ]:
        if not cat_scores.empty:
            all_scores = pd.merge(all_scores,
                                  cat_scores[['Location', cat_col]],
                                  on='Location',
                                  how='outer')

    # Calculate average score
    score_columns = [col for col in all_scores.columns if col.startswith('Cat_')]
    all_scores['Avg_Score'] = all_scores[score_columns].mean(axis=1)

    # Save the results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)

    # Save individual category scores
    cat_a_scores.to_csv(f'{results_dir}/Category A_scores_{timestamp}.csv', index=False)
    cat_b_scores.to_csv(f'{results_dir}/Category B_scores_{timestamp}.csv', index=False)
    cat_c_scores.to_csv(f'{results_dir}/Category C_scores_{timestamp}.csv', index=False)
    cat_d_scores.to_csv(f'{results_dir}/Category D_scores_{timestamp}.csv', index=False)

    # Save combined scores
    output_path = f'{results_dir}/all_scores_{timestamp}.csv'
    all_scores.to_csv(output_path, index=False)

    print(f"All scores saved to {output_path}")
    return all_scores

def combine_scores(self, scores):
    """Combine individual category scores into final results"""
    try:
        # Merge all category scores
        final = scores['cat_a']
        for cat in ['cat_b', 'cat_c', 'cat_d']:
            if scores[cat] is not None:
                final = final.merge(scores[cat], on='Location', how='outer')

        # Fill missing values
        final = final.fillna(0)

        # Calculate average score
        score_columns = [col for col in final.columns if col.startswith('Cat_')]
        final['Avg_Score'] = final[score_columns].mean(axis=1)

        return final.round(2)

    except Exception as e:
        logging.error(f"Error combining scores: {str(e)}")
        return None

if __name__ == "__main__":
    generate_scores()