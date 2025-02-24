import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime

# Configure console logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Log to console
        logging.FileHandler('scoring_test.log')  # Also save to file
    ]
)


class TestScoring:
    def __init__(self):
        self.current_year = 2024
        self.data_dir = 'data'
        self.output_dir = 'output'
        os.makedirs(self.output_dir, exist_ok=True)

    def load_test_data(self):
        """Load and return all required data"""
        logging.info("Loading test data...")

        data = {}
        for filename in [
            'MC_per_Product_SQL.csv',
            'df_vols_query.csv',
            'Market_Summary_PMI.csv',
            'Market_Summary_Comp.csv',
            'Pax_Nat.csv',
            'similarity_file.csv',
            'iata_location_query.csv',
            'mrk_nat_map_query.csv'
        ]:
            filepath = os.path.join(self.data_dir, filename)
            data[filename.split('.')[0]] = pd.read_csv(filepath)
            logging.info(f"Loaded {filename}")

        return data

    def test_category_a(self, data):
        """Test Category A scoring"""
        logging.info("\nTesting Category A scoring...")

        try:
            from category_a_scorer import CategoryAScorer
            scorer = CategoryAScorer(self.current_year)
            scores = scorer.calculate_scores(
                data['df_vols_query'],
                data['MC_per_Product_SQL']
            )

            if scores is not None:
                logging.info(f"Category A scores shape: {scores.shape}")
                logging.info("\nSample scores:")
                print(scores.head())
                scores.to_csv(os.path.join(self.output_dir, 'test_cat_a_scores.csv'), index=False)
                return True
            return False

        except Exception as e:
            logging.error(f"Category A scoring failed: {str(e)}")
            return False

    def test_category_b(self, data):
        """Test Category B scoring"""
        logging.info("\nTesting Category B scoring...")

        try:
            from category_b_scorer import CategoryBScorer
            scorer = CategoryBScorer()
            scores = scorer.calculate_scores(
                data['Market_Summary_PMI'],
                data['Market_Summary_Comp']
            )

            if scores is not None:
                logging.info(f"Category B scores shape: {scores.shape}")
                logging.info("\nSample scores:")
                print(scores.head())
                scores.to_csv(os.path.join(self.output_dir, 'test_cat_b_scores.csv'), index=False)
                return True
            return False

        except Exception as e:
            logging.error(f"Category B scoring failed: {str(e)}")
            return False

    def test_category_c(self, data):
        """Test Category C scoring"""
        logging.info("\nTesting Category C scoring...")

        try:
            from category_c_scorer import CategoryCScorer
            scorer = CategoryCScorer(self.current_year)
            scores = scorer.calculate_scores(
                data['Pax_Nat'],
                data['df_vols_query'],
                data['mrk_nat_map_query'],
                data['iata_location_query']
            )

            if scores is not None:
                logging.info(f"Category C scores shape: {scores.shape}")
                logging.info("\nSample scores:")
                print(scores.head())
                scores.to_csv(os.path.join(self.output_dir, 'test_cat_c_scores.csv'), index=False)
                return True
            return False

        except Exception as e:
            logging.error(f"Category C scoring failed: {str(e)}")
            return False

    def test_category_d(self, data):
        """Test Category D scoring"""
        logging.info("\nTesting Category D scoring...")

        try:
            from category_d_scorer import CategoryDScorer
            scorer = CategoryDScorer(self.current_year)
            scores = scorer.calculate_scores(
                data['df_vols_query'],
                data['similarity_file'],
                data['iata_location_query']
            )

            if scores is not None:
                logging.info(f"Category D scores shape: {scores.shape}")
                logging.info("\nSample scores:")
                print(scores.head())
                scores.to_csv(os.path.join(self.output_dir, 'test_cat_d_scores.csv'), index=False)
                return True
            return False

        except Exception as e:
            logging.error(f"Category D scoring failed: {str(e)}")
            return False

    def run_all_tests(self):
        """Run all scoring tests"""
        logging.info("Starting scoring tests...")

        # Load data
        data = self.load_test_data()
        if not data:
            logging.error("Data loading failed!")
            return

        # Test each category
        results = {
            'Category A': self.test_category_a(data),
            'Category B': self.test_category_b(data),
            'Category C': self.test_category_c(data),
            'Category D': self.test_category_d(data)
        }

        # Print summary
        logging.info("\nTest Results Summary:")
        for category, success in results.items():
            status = "✓ Passed" if success else "✗ Failed"
            logging.info(f"{category}: {status}")


def main():
    tester = TestScoring()
    tester.run_all_tests()


if __name__ == "__main__":
    main()


/Users/kemalgider/Desktop/SCENARIO_TESTING/demo/.venv/bin/python /Users/kemalgider/Desktop/PORTFOLIO OPT./popt/test.py
2025-02-21 13:44:18,558 - INFO - Starting scoring tests...
2025-02-21 13:44:18,558 - INFO - Loading test data...
2025-02-21 13:44:18,568 - INFO - Loaded MC_per_Product_SQL.csv
2025-02-21 13:44:18,597 - INFO - Loaded df_vols_query.csv
2025-02-21 13:44:18,599 - INFO - Loaded Market_Summary_PMI.csv
2025-02-21 13:44:18,603 - INFO - Loaded Market_Summary_Comp.csv
2025-02-21 13:44:18,635 - INFO - Loaded Pax_Nat.csv
2025-02-21 13:44:18,637 - INFO - Loaded similarity_file.csv
2025-02-21 13:44:18,638 - INFO - Loaded iata_location_query.csv
2025-02-21 13:44:18,639 - INFO - Loaded mrk_nat_map_query.csv
2025-02-21 13:44:18,639 - INFO -
Testing Category A scoring...
2025-02-21 13:44:19,128 - INFO - Category A scores shape: (566, 5)
2025-02-21 13:44:19,128 - INFO -
Sample scores:
2025-02-21 13:44:19,132 - INFO -
Testing Category B scoring...
                  Location  Cat_A  Green_Count  Red_Count  Total_SKUs
0            Konya Airport   6.87            5          0          81
1  Paris Charles De Gaulle   6.67            4          2          69
2             Hohhot Baita   6.71            1          0          82
3               Ngurah Rai   6.76            4          0         140
4               Manchester   6.49            0          2          76
2025-02-21 13:44:19,429 - ERROR - Error calculating Category B scores: sequence item 0: expected str instance, float found
2025-02-21 13:44:19,429 - INFO -
Testing Category C scoring...
2025-02-21 13:44:19,438 - ERROR - Error calculating Category C scores: "['Location'] not in index"
2025-02-21 13:44:19,438 - INFO -
Testing Category D scoring...
2025-02-21 13:44:19,442 - ERROR - Error calculating Category D scores: "['Location'] not in index"
2025-02-21 13:44:19,442 - INFO -
Test Results Summary:
2025-02-21 13:44:19,442 - INFO - Category A: ✓ Passed
2025-02-21 13:44:19,442 - INFO - Category B: ✗ Failed
2025-02-21 13:44:19,442 - INFO - Category C: ✗ Failed
2025-02-21 13:44:19,442 - INFO - Category D: ✗ Failed

Process finished with exit code 0