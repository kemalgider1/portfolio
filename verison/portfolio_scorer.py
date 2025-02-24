import pandas as pd
import numpy as np
import logging
from datetime import datetime

from data_loader import DataLoader
from category_a_scorer import CategoryAScorer
from category_b_scorer import CategoryBScorer
from category_c_scorer import CategoryCScorer
from category_d_scorer import CategoryDScorer

logging.basicConfig(
    filename=f'portfolio_scoring_{datetime.now().strftime("%Y%m%d")}.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class PortfolioScorer:
    def __init__(self, current_year=2024):
        self.current_year = current_year
        self.data_loader = DataLoader(current_year)
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
            print("\nCategory A Scores (sample):")
            print(cat_a_scores.head())
            
            cat_b_scores = self.cat_b_scorer.calculate_scores(
                data['market_pmi'], 
                data['market_comp']
            )
            print("\nCategory B Scores (sample):")
            print(cat_b_scores.head())
            
            cat_c_scores = self.cat_c_scorer.calculate_scores(
                data['pax_data'],
                data['df_vols'],
                data['mrk_nat_map'],
                data['iata_location']
            )
            print("\nCategory C Scores (sample):")
            print(cat_c_scores.head())
            
            cat_d_scores = self.cat_d_scorer.calculate_scores(
                data['df_vols'],
                data['similarity_file'],
                data['iata_location']
            )
            print("\nCategory D Scores (sample):")
            print(cat_d_scores.head())
            
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
        final_scores.to_csv(
            f'portfolio_scores_{self.current_year}_{timestamp}.csv',
            index=False
        )
        
        # Generate summary report
        summary = self._generate_summary(final_scores)
        with open(f'portfolio_summary_{self.current_year}_{timestamp}.txt', 'w') as f:
            f.write(summary)
            
    def _generate_summary(self, final_scores):
        summary = f"""
Portfolio Optimization Scoring Summary
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Year: {self.current_year}

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
    scorer = PortfolioScorer()
    final_scores = scorer.calculate_all_scores()
    
    if final_scores is not None:
        print("\nScoring completed successfully.")
        print(f"Total locations scored: {len(final_scores)}")
        print(f"Average final score: {final_scores['Final_Score'].mean():.2f}")
    else:
        print("Scoring failed. Check the log file for details.")

if __name__ == "__main__":
    main()