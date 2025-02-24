import pandas as pd
import numpy as np
import logging
from datetime import datetime
import os
from pathlib import Path

from data_loader import DataLoader
from category_a_scorer import CategoryAScorer
from category_b_scorer import CategoryBScorer
from category_c_scorer import CategoryCScorer
from category_d_scorer import CategoryDScorer

class PortfolioScorer:
    def __init__(self, current_year=2024, data_dir='data', output_dir='output'):
        self.current_year = current_year
        self.data_dir = data_dir
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Configure logging
        log_path = os.path.join(
            output_dir, 
            f'portfolio_scoring_{datetime.now().strftime("%Y%m%d")}.log'
        )
        logging.basicConfig(
            filename=log_path,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Initialize components
        self.data_loader = DataLoader(current_year, data_dir)
        self.cat_a_scorer = CategoryAScorer(current_year)
        self.cat_b_scorer = CategoryBScorer()
        self.cat_c_scorer = CategoryCScorer(current_year)
        self.cat_d_scorer = CategoryDScorer(current_year)
        
    def calculate_all_scores(self):
        try:
            # Load data
            data = self.data_loader.load_all_data()
            if data is None:
                return None
                
            # Calculate individual category scores
            cat_a_scores = self.cat_a_scorer.calculate_scores(
                data['df_vols'], 
                data['mc_per_product']
            )
            self._save_category_scores('Cat_A', cat_a_scores)
            
            cat_b_scores = self.cat_b_scorer.calculate_scores(
                data['market_pmi'], 
                data['market_comp']
            )
            self._save_category_scores('Cat_B', cat_b_scores)
            
            cat_c_scores = self.cat_c_scorer.calculate_scores(
                data['pax_data'],
                data['df_vols'],
                data['mrk_nat_map'],
                data['iata_location']
            )
            self._save_category_scores('Cat_C', cat_c_scores)
            
            cat_d_scores = self.cat_d_scorer.calculate_scores(
                data['df_vols'],
                data['similarity_file'],
                data['iata_location']
            )
            self._save_category_scores('Cat_D', cat_d_scores)
            
            # Combine all scores
            final_scores = self._combine_scores([
                cat_a_scores,
                cat_b_scores,
                cat_c_scores,
                cat_d_scores
            ])
            
            # Save results
            self._save_results(final_scores)
            
            return final_scores
            
        except Exception as e:
            logging.error(f"Error calculating portfolio scores: {str(e)}")
            return None
            
    def _save_category_scores(self, category, scores):
        """Save individual category scores"""
        if scores is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(
                self.output_dir,
                f'{category}_scores_{self.current_year}_{timestamp}.csv'
            )
            scores.to_csv(filename, index=False)
            print(f"\n{category} Scores (sample):")
            print(scores.head())
            
    def _combine_scores(self, score_list):
        try:
            # Start with first category scores
            final_scores = score_list[0]
            
            # Merge remaining categories
            for scores in score_list[1:]:
                final_scores = final_scores.merge(
                    scores,
                    on='Location',
                    how='outer'
                )
            
            # Calculate final score
            score_cols = ['Cat_A', 'Cat_B', 'Cat_C', 'Cat_D']
            final_scores['Final_Score'] = final_scores[score_cols].mean(axis=1)
            
            # Add metadata
            final_scores['Calculation_Date'] = datetime.now().strftime("%Y-%m-%d")
            final_scores['Data_Year'] = self.current_year
            
            return final_scores
            
        except Exception as e:
            logging.error(f"Error combining scores: {str(e)}")
            return None
            
    def _save_results(self, final_scores):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save to CSV
        csv_path = os.path.join(
            self.output_dir,
            f'portfolio_scores_{self.current_year}_{timestamp}.csv'
        )
        final_scores.to_csv(csv_path, index=False)
        
        # Generate summary report
        summary = self._generate_summary(final_scores)
        summary_path = os.path.join(
            self.output_dir,
            f'portfolio_summary_{self.current_year}_{timestamp}.txt'
        )
        with open(summary_path, 'w') as f:
            f.write(summary)
            
    def _generate_summary(self, final_scores):
        summary = f"""
Portfolio Optimization Scoring Summary
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Year: {self.current_year}
Data Directory: {self.data_dir}
Output Directory: {self.output_dir}

Overall Statistics:
------------------
Total Locations: {len(final_scores)}
Average Final Score: {final_scores['Final_Score'].mean():.2f}
Score Range: {final_scores['Final_Score'].min():.2f} - {final_scores['Final_Score'].max():.2f}

Category Statistics:
------------------
Category A (Portfolio Performance):
    Mean: {final_scores['Cat_A'].mean():.2f}
    Std Dev: {final_scores['Cat_A'].std():.2f}

Category B (Segment Distribution):
    Mean: {final_scores['Cat_B'].mean():.2f}
    Std Dev: {final_scores['Cat_B'].std():.2f}

Category C (PAX Alignment):
    Mean: {final_scores['Cat_C'].mean():.2f}
    Std Dev: {final_scores['Cat_C'].std():.2f}

Category D (Cluster Comparison):
    Mean: {final_scores['Cat_D'].mean():.2f}
    Std Dev: {final_scores['Cat_D'].std():.2f}

Top 10 Performing Locations:
--------------------------
{final_scores.nlargest(10, 'Final_Score')[['Location', 'Final_Score']].to_string()}

Bottom 10 Performing Locations:
----------------------------
{final_scores.nsmallest(10, 'Final_Score')[['Location', 'Final_Score']].to_string()}
"""
        return summary

def main():
    # Initialize scorer with data and output directories
    scorer = PortfolioScorer(
        current_year=2024,
        data_dir='data',
        output_dir='output'
    )
    
    # Calculate scores
    final_scores = scorer.calculate_all_scores()
    
    if final_scores is not None:
        print("\nScoring completed successfully.")
        print(f"Total locations scored: {len(final_scores)}")
        print(f"Average final score: {final_scores['Final_Score'].mean():.2f}")
        print(f"\nResults saved in: {scorer.output_dir}")
    else:
        print("Scoring failed. Check the log file for details.")

if __name__ == "__main__":
    main()