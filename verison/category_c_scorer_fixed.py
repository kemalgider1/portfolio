import pandas as pd
import numpy as np
from scipy import stats
import logging

class CategoryCScorer:
    def __init__(self, current_year=2024):
        self.current_year = current_year
        self.brand_attributes = ['Taste', 'Thickness', 'Flavor', 'Length']
        
    def calculate_scores(self, pax_data, df_vols, mrk_nat_map, iata_location):
        """Calculate Category C scores based on PAX alignment"""
        try:
            # Process PAX data
            pax_processed = self._process_pax_data(pax_data, mrk_nat_map, iata_location)
            
            scores = []
            for location in pax_processed['Location'].unique():
                # Get location data
                loc_pax = pax_processed[pax_processed['Location'] == location]
                loc_vols = df_vols[df_vols['Location'] == location]
                
                if not (loc_pax.empty or loc_vols.empty):
                    # Calculate distributions
                    pax_dist = self._calculate_pax_distribution(loc_pax)
                    vol_dist = self._calculate_volume_distribution(loc_vols)
                    
                    # Calculate score
                    score = self._calculate_correlation(pax_dist, vol_dist)
                    scores.append({
                        'Location': location,
                        'Cat_C': score
                    })
                    
            return pd.DataFrame(scores)
            
        except Exception as e:
            logging.error(f"Error in Category C scoring: {str(e)}")
            raise
            
    def _process_pax_data(self, pax_data, mrk_nat_map, iata_location):
        """Process and prepare PAX data"""
        # Filter current year
        df = pax_data[pax_data['Year'] == self.current_year].copy()
        
        # Add mappings
        df = df.merge(
            mrk_nat_map,
            left_on='Nationality',
            right_on='PASSENGER_NATIONALITY',
            how='left'
        )
        
        df = df.merge(
            iata_location[['IATA', 'Location']],
            on='IATA',
            how='left'
        )
        
        # Calculate shares
        df['Share'] = df.groupby('Location')['Pax'].transform(
            lambda x: x / x.sum()
        )
        
        return df
        
    def _calculate_pax_distribution(self, pax_data):
        """Calculate PAX distribution by nationality"""
        return pax_data.groupby('Nationality')['Share'].sum()
        
    def _calculate_volume_distribution(self, vol_data):
        """Calculate volume distribution by segment"""
        if vol_data.empty:
            return pd.Series()
            
        total_vol = vol_data[f'{self.current_year} Volume'].sum()
        if total_vol > 0:
            return vol_data.groupby('TMO')[f'{self.current_year} Volume'].sum() / total_vol
        return pd.Series()
        
    def _calculate_correlation(self, dist1, dist2):
        """Calculate correlation between distributions"""
        # Align distributions
        all_categories = sorted(set(dist1.index) | set(dist2.index))
        dist1 = dist1.reindex(all_categories, fill_value=0)
        dist2 = dist2.reindex(all_categories, fill_value=0)
        
        # Calculate correlation if we have variation
        if dist1.std() > 0 and dist2.std() > 0:
            corr, _ = stats.spearmanr(dist1, dist2)
            r2 = corr ** 2 if not np.isnan(corr) else 0
        else:
            r2 = 0
            
        return round(r2 * 10, 2)  # Scale to 0-10