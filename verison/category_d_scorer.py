import pandas as pd
import numpy as np
from scipy import stats
import logging

class CategoryDScorer:
    def __init__(self, current_year=2024):
        self.current_year = current_year
        self.brand_attributes = ['Taste', 'Thickness', 'Flavor', 'Length']
        
    def calculate_scores(self, market_mix, similarity_file, iata_location):
        try:
            # Process similarity data
            similarity_data = self._process_similarity_file(similarity_file, iata_location)
            
            # Calculate cluster comparisons
            scores = self._calculate_cluster_scores(market_mix, similarity_data)
            
            return scores
            
        except Exception as e:
            logging.error(f"Error calculating Category D scores: {str(e)}")
            return None
            
    def _process_similarity_file(self, similarity_file, iata_location):
        # Sort and rank similar locations
        sim_data = similarity_file.copy()
        sim_data = sim_data.sort_values(['IATA', 'SIMILARITY_SCORE'], ascending=[True, False])
        sim_data['Rank'] = sim_data.groupby('IATA').cumcount() + 1
        
        # Keep top 4 similar locations
        sim_data = sim_data[sim_data['Rank'] <= 4]
        
        # Add location information
        sim_data = sim_data.merge(
            iata_location[['IATA', 'Location']],
            how='left',
            on='IATA'
        )
        
        return sim_data
        
    def _calculate_cluster_scores(self, market_mix, similarity_data):
        scores = []
        
        for location in market_mix['Location'].unique():
            # Get cluster IATAs
            cluster_iatas = similarity_data[
                similarity_data['Location'] == location
            ]['CLUSTER_IATA'].tolist()
            
            if cluster_iatas:
                # Get location and cluster data
                location_dist = self._calculate_location_distribution(
                    market_mix, location
                )
                cluster_dist = self._calculate_cluster_distribution(
                    market_mix, cluster_iatas
                )
                
                if not (location_dist.empty or cluster_dist.empty):
                    # Calculate correlation
                    score = self._calculate_distribution_correlation(
                        location_dist, cluster_dist
                    )
                    scores.append({
                        'Location': location,
                        'Cat_D': score
                    })
        
        return pd.DataFrame(scores)
        
    def _calculate_location_distribution(self, market_mix, location):
        location_data = market_mix[market_mix['Location'] == location]
        
        dist = pd.pivot_table(
            location_data,
            index='Location',
            columns=self.brand_attributes,
            values=f'{self.current_year} Volume',
            aggfunc='sum',
            fill_value=0
        )
        
        # Normalize
        row_sums = dist.sum(axis=1)
        dist = dist.div(row_sums, axis=0).fillna(0)
        
        return dist
        
    def _calculate_cluster_distribution(self, market_mix, cluster_iatas):
        cluster_data = market_mix[market_mix['IATA'].isin(cluster_iatas)]
        
        dist = pd.pivot_table(
            cluster_data,
            index='IATA',
            columns=self.brand_attributes,
            values=f'{self.current_year} Volume',
            aggfunc='sum',
            fill_value=0
        )
        
        # Normalize
        row_sums = dist.sum(axis=1)
        dist = dist.div(row_sums, axis=0).fillna(0)
        
        return dist.mean()  # Average distribution across cluster
        
    def _calculate_distribution_correlation(self, dist1, dist2):
        # Align columns
        common_cols = dist1.columns.intersection(dist2.index)
        if len(common_cols) > 1:
            values1 = dist1[common_cols].values.flatten()
            values2 = dist2[common_cols].values
            
            # Calculate correlation if we have variation
            if np.std(values1) > 0 and np.std(values2) > 0:
                corr, _ = stats.spearmanr(values1, values2)
                r2 = corr ** 2 if not np.isnan(corr) else 0
            else:
                r2 = 0
        else:
            r2 = 0
            
        return round(r2 * 10, 2)  # Scale to 0-10