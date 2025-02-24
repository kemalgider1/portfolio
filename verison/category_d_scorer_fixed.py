import pandas as pd
import numpy as np
from scipy import stats
import logging

class CategoryDScorer:
    def __init__(self, current_year=2024):
        self.current_year = current_year
        self.brand_attributes = ['Taste', 'Thickness', 'Flavor', 'Length']
        
    def calculate_scores(self, df_vols, similarity_file, iata_location):
        """Calculate Category D scores based on location clusters"""
        try:
            # Process similarity data
            similarity_data = self._process_similarity_file(similarity_file, iata_location)
            
            scores = []
            for location in df_vols['Location'].unique():
                # Get cluster data
                cluster_info = self._get_cluster_info(location, similarity_data)
                
                if cluster_info is not None:
                    # Calculate distributions
                    loc_dist = self._calculate_location_distribution(
                        df_vols, location
                    )
                    cluster_dist = self._calculate_cluster_distribution(
                        df_vols, cluster_info['cluster_locations']
                    )
                    
                    # Calculate score
                    score = self._calculate_correlation(loc_dist, cluster_dist)
                    scores.append({
                        'Location': location,
                        'Cat_D': score
                    })
                    
            return pd.DataFrame(scores)
            
        except Exception as e:
            logging.error(f"Error in Category D scoring: {str(e)}")
            raise
            
    def _process_similarity_file(self, similarity_file, iata_location):
        """Process similarity file data"""
        sim_data = similarity_file.copy()
        
        # Sort and rank
        sim_data = sim_data.sort_values(
            ['IATA', 'SIMILARITY_SCORE'],
            ascending=[True, False]
        )
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
        
    def _get_cluster_info(self, location, similarity_data):
        """Get cluster information for a location"""
        location_data = similarity_data[similarity_data['Location'] == location]
        
        if not location_data.empty:
            return {
                'location': location,
                'cluster_locations': location_data['CLUSTER_IATA'].tolist()
            }
        return None
        
    def _calculate_location_distribution(self, df_vols, location):
        """Calculate distribution for a location"""
        location_data = df_vols[df_vols['Location'] == location]
        
        if not location_data.empty:
            # Calculate segment distribution
            location_data['Segment'] = location_data[self.brand_attributes].astype(str).agg(
                '_'.join, axis=1
            )
            dist = location_data.groupby('Segment')[f'{self.current_year} Volume'].sum()
            return dist / dist.sum() if dist.sum() > 0 else dist
            
        return pd.Series()
        
    def _calculate_cluster_distribution(self, df_vols, cluster_iatas):
        """Calculate distribution for cluster"""
        cluster_data = df_vols[df_vols['IATA'].isin(cluster_iatas)]
        
        if not cluster_data.empty:
            # Calculate segment distribution
            cluster_data['Segment'] = cluster_data[self.brand_attributes].astype(str).agg(
                '_'.join, axis=1
            )
            dist = cluster_data.groupby('Segment')[f'{self.current_year} Volume'].sum()
            return dist / dist.sum() if dist.sum() > 0 else dist
            
        return pd.Series()
        
    def _calculate_correlation(self, dist1, dist2):
        """Calculate correlation between distributions"""
        if dist1.empty or dist2.empty:
            return 0
            
        # Align distributions
        all_segments = sorted(set(dist1.index) | set(dist2.index))
        dist1 = dist1.reindex(all_segments, fill_value=0)
        dist2 = dist2.reindex(all_segments, fill_value=0)
        
        # Calculate correlation if we have variation
        if dist1.std() > 0 and dist2.std() > 0:
            corr, _ = stats.spearmanr(dist1, dist2)
            r2 = corr ** 2 if not np.isnan(corr) else 0
        else:
            r2 = 0
            
        return round(r2 * 10, 2)  # Scale to 0-10
