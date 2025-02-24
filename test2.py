import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime

# Configure basic logging to both file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('scoring_test.log')
    ]
)


def export_results(scores_dict, validation_output, output_dir='results'):
    """
    Export all category scores and validation results to files
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get timestamp for unique filenames
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Export individual category scores
    for category, df in scores_dict.items():
        if df is not None:
            # Excel export
            excel_filename = os.path.join(output_dir, f'{category}_scores_{timestamp}.xlsx')
            df.to_excel(excel_filename, index=False)
            logging.info(f"Exported {category} scores to {excel_filename}")

            # CSV export
            csv_filename = os.path.join(output_dir, f'{category}_scores_{timestamp}.csv')
            df.to_csv(csv_filename, index=False)
            logging.info(f"Exported {category} scores to {csv_filename}")

    # Export validation report
    validation_file = os.path.join(output_dir, f'validation_report_{timestamp}.txt')
    with open(validation_file, 'w') as f:
        f.write(validation_output['report'])
    logging.info(f"Exported validation report to {validation_file}")


def standardize_data(data_dict):
    """Standardize column names in all dataframes"""
    standardized = {}

    # Column name mappings
    mappings = {
        'LOCATION': 'Location',
        'IATA': 'IATA',
        'NATIONALITY': 'Nationality',
        'CR_BRANDID': 'CR_BrandId'
    }

    for key, df in data_dict.items():
        df = df.copy()
        # Create case-insensitive rename dictionary
        rename_dict = {}
        for col in df.columns:
            for old, new in mappings.items():
                if col.upper() == old.upper():
                    rename_dict[col] = new

        if rename_dict:
            df = df.rename(columns=rename_dict)
        standardized[key] = df

    return standardized


def validate_test_data(data):
    """Validate test data before running tests"""
    required_columns = {
        'df_vols_query': ['Location', 'TMO', 'CR_BrandId', '2024 Volume'],
        'MC_per_Product_SQL': ['Location', 'CR_BrandId', '2024 MC', '2024 NOR'],
        'Market_Summary_PMI': ['Location', 'TMO', 'PMI_Seg_SKU'],
        'Market_Summary_Comp': ['Location', 'TMO', 'Comp_Seg_SKU'],
        'iata_location_query': ['Location', 'IATA']
    }

    for df_name, req_cols in required_columns.items():
        if df_name not in data:
            logging.error(f"Missing required dataset: {df_name}")
            return False

        df = data[df_name]
        missing = set(req_cols) - set(col.strip() for col in df.columns)
        if missing:
            logging.error(f"Missing columns in {df_name}: {missing}")
            return False

    return True


def test_category_scoring():
    """Test all category scoring implementations"""
    try:
        logging.info("Starting category scoring tests...")

        # Load test data
        logging.info("Loading test data...")
        data_dir = 'data'
        test_files = [
            'MC_per_Product_SQL.csv',
            'df_vols_query.csv',
            'Market_Summary_PMI.csv',
            'Market_Summary_Comp.csv',
            'Pax_Nat.csv',
            'similarity_file.csv',
            'iata_location_query.csv',
            'mrk_nat_map_query.csv'
        ]

        # Load and standardize data
        data = {}
        for file in test_files:
            filepath = os.path.join(data_dir, file)
            if not os.path.exists(filepath):
                logging.error(f"Missing required file: {file}")
                return False
            df = pd.read_csv(filepath)
            data[file.split('.')[0]] = df
            logging.info(f"Loaded {file}: {df.shape}")

        # Standardize column names
        data = standardize_data(data)

        # Validate test data
        if not validate_test_data(data):
            return False

        # Test Category A
        from category_a_scorer import CategoryAScorer
        cat_a = CategoryAScorer(current_year=2024)
        logging.info("Testing Category A scoring...")
        cat_a_scores = cat_a.calculate_scores(
            data['df_vols_query'],
            data['MC_per_Product_SQL']
        )
        if cat_a_scores is not None:
            logging.info(f"Category A scores shape: {cat_a_scores.shape}")
            print("\nCategory A sample scores:")
            print(cat_a_scores.head())

        # Test Category B
        from category_b_scorer import CategoryBScorer
        cat_b = CategoryBScorer()
        logging.info("Testing Category B scoring...")
        cat_b_scores = cat_b.calculate_scores(
            data['Market_Summary_PMI'],
            data['Market_Summary_Comp']
        )
        if cat_b_scores is not None:
            logging.info(f"Category B scores shape: {cat_b_scores.shape}")
            print("\nCategory B sample scores:")
            print(cat_b_scores.head())

        # Test Category C
        from category_c_scorer import CategoryCScorer
        cat_c = CategoryCScorer(current_year=2024)
        logging.info("Testing Category C scoring...")
        cat_c_scores = cat_c.calculate_scores(
            data['Pax_Nat'],
            data['df_vols_query'],
            data['mrk_nat_map_query'],
            data['iata_location_query']
        )
        if cat_c_scores is not None:
            logging.info(f"Category C scores shape: {cat_c_scores.shape}")
            print("\nCategory C sample scores:")
            print(cat_c_scores.head())

        # Test Category D
        from category_d_scorer import CategoryDScorer
        cat_d = CategoryDScorer()
        logging.info("Testing Category D scoring...")
        cat_d_scores = cat_d.calculate_scores(
            data['df_vols_query'],
            data['similarity_file'],
            data['iata_location_query']
        )
        if cat_d_scores is not None:
            logging.info(f"Category D scores shape: {cat_d_scores.shape}")
            print("\nCategory D sample scores:")
            print(cat_d_scores.head())

        # Store all scores in a dictionary
        scores_dict = {
            'Category A': cat_a_scores,
            'Category B': cat_b_scores,
            'Category C': cat_c_scores,
            'Category D': cat_d_scores
        }

        # Validate portfolio scores using refined validator
        from refined_score_validator import validate_portfolio_scores

        # Validate all scores
        validation_output = validate_portfolio_scores(
            cat_a_scores,
            cat_b_scores,
            cat_c_scores,
            cat_d_scores
        )

        if validation_output['valid']:
            logging.info("Score validation passed")
            print("\nValidation Report:")
            print(validation_output['report'])
        else:
            logging.error("Score validation failed")
            print("\nValidation Errors:")
            print(validation_output['report'])

        # Validate scores
        all_valid = True
        for category, score_df in scores_dict.items():
            if score_df is None:
                logging.error(f"{category} scoring failed")
                all_valid = False
                continue

            # Validate score ranges
            if not score_df['Cat_' + category[-1]].between(0, 10).all():
                logging.error(f"{category} has scores outside 0-10 range")
                all_valid = False

        if all_valid:
            # Export results if validation passed
            export_results(scores_dict, validation_output)
            logging.info("All category tests completed successfully")
            return True
        return False

    except Exception as e:
        logging.error(f"Test execution failed: {str(e)}")
        return False


if __name__ == "__main__":
    logging.info("Starting test execution...")
    success = test_category_scoring()
    if success:
        logging.info("All tests completed successfully")
    else:
        logging.error("Test execution failed")