import pandas as pd
import numpy as np
from scipy import stats
import logging

class CategoryCScorer:
    def __init__(self, current_year=2024):
        self.current_year = current_year
        
    def calculate_scores(self, pax_data, market_data, mrk_nat_map, iata_location):
        try:
            # Process PAX data
            pax_processed = self._process_pax_data(pax_data, mrk_nat_map, iata_location)
            
            # Calculate expected vs actual distributions
            actual_dist = self._calculate_actual_distribution(market_data)
            expected_dist = self._calculate_expected_distribution(pax_processed)
            
            # Calculate scores
            scores = self._calculate_correlations(actual_dist, expected_dist)
            
            return scores
            
        except Exception as e:
            logging.error(f"Error calculating Category C scores: {str(e)}")
            return None
            
    def _process_pax_data(self, pax_data, mrk_nat_map, iata_location):
        # Filter current year
        df = pax_data[pax_data['Year'] == self.current_year].copy()
        
        # Merge mappings
        df = df.merge(mrk_nat_map, 
                     left_on='Nationality', 
                     right_on='PASSENGER_NATIONALITY', 
                     how='left')
        df = df.merge(iata_location[['IATA', 'Location']], 
                     on='IATA', 
                     how='left')
        
        # Calculate PAX shares
        df['PAX_Share'] = df.groupby('Location')['Pax'].transform(
            lambda x: x / x.sum()
        )
        
        return df
        
    def _calculate_actual_distribution(self, market_data):
        # Calculate actual market distribution
        dist = market_data.pivot_table(
            index='Location',
            columns=['Flavor', 'Taste', 'Length'],
            values=f'{self.current_year} Volume',
            aggfunc='sum',
            fill_value=0
        )
        
        # Normalize distributions
        row_sums = dist.sum(axis=1)
        dist = dist.div(row_sums, axis=0).fillna(0)
        
        return dist
        
    def _calculate_expected_distribution(self, pax_data):
        # Weight market preferences by PAX share
        dist = pd.pivot_table(
            pax_data,
            index='Location',
            columns=['Countries'],  # Using country preferences
            values='PAX_Share',
            aggfunc='sum',
            fill_value=0
        )
        
        # Normalize distributions
        row_sums = dist.sum(axis=1)
        dist = dist.div(row_sums, axis=0).fillna(0)
        
        return dist
        
    def _calculate_correlations(self, actual_dist, expected_dist):
        scores = []
        
        # Align indexes
        locations = sorted(set(actual_dist.index) & set(expected_dist.index))
        
        for location in locations:
            actual_values = actual_dist.loc[location]
            expected_values = expected_dist.loc[location]
            
            # Calculate correlation only if we have variation
            if actual_values.std() > 0 and expected_values.std() > 0:
                # Use Spearman correlation for non-normal distributions
                corr, _ = stats.spearmanr(actual_values, expected_values)
                r2 = corr ** 2 if not np.isnan(corr) else 0
            else:
                r2 = 0
                
            scores.append({
                'Location': location,
                'Cat_C': round(r2 * 10, 2)
            })
        
        return pd.DataFrame(scores)