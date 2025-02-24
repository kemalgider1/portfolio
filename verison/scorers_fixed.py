### Category A Scorer

import pandas as pd
import numpy as np
import logging

class CategoryAScorer:
    def __init__(self, current_year=2024):
        self.current_year = current_year
        self.previous_year = current_year - 1
        
    def calculate_scores(self, df_vols, mc_per_product):
        try:
            # Merge and prepare data
            df = self._prepare_data(df_vols, mc_per_product)
            
            # Calculate flags
            df = self._calculate_flags(df)
            
            # Calculate scores
            scores = self._calculate_location_scores(df)
            
            return scores
            
        except Exception as e:
            logging.error(f"Error calculating Category A scores: {str(e)}")
            return None
            
    def _prepare_data(self, df_vols, mc_per_product):
        # Merge data
        df = df_vols.merge(
            mc_per_product[['DF_Market', 'Location', 'CR_BrandId', 
                           f'{self.current_year} MC', f'{self.current_year} NOR']], 
            how='left', 
            on=['DF_Market', 'Location', 'CR_BrandId']
        ).fillna(0)
        
        # Calculate volume share
        df['Volume_Share'] = df.groupby('Location')[f'{self.current_year} Volume'].transform(
            lambda x: x / x.sum()
        )
        
        return df
        
    def _calculate_flags(self, df):
        # Calculate ranks
        df['Volume_Rank'] = df.groupby('Location')[f'{self.current_year} Volume'].rank(
            ascending=False, method='dense'
        )
        
        # Calculate thresholds per location
        thresholds = df.groupby('Location').agg({
            'SKU': 'count'
        }).reset_index()
        thresholds.columns = ['Location', 'Total_SKUs']
        thresholds['Green_Threshold'] = np.ceil(thresholds['Total_SKUs'] * 0.05)
        thresholds['Red_Threshold'] = np.ceil(thresholds['Total_SKUs'] * 0.25)
        
        # Merge thresholds back
        df = df.merge(thresholds, on='Location', how='left')
        
        # Set flags
        df['Green_Flag'] = ((df['Volume_Rank'] <= df['Green_Threshold']) & 
                           (df['TMO'] == 'PMI')).astype(int)
        df['Red_Flag'] = ((df['Volume_Rank'] > df['Total_SKUs'] - df['Red_Threshold']) & 
                         (df['TMO'] == 'PMI')).astype(int)
        
        return df
        
    def _calculate_location_scores(self, df):
        scores = []
        for location in df['Location'].unique():
            loc_data = df[df['Location'] == location]
            green_count = loc_data['Green_Flag'].sum()
            red_count = loc_data['Red_Flag'].sum()
            total_count = len(loc_data)
            
            if total_count > 0:
                # Calculate raw score (-200 to +100 range)
                raw_score = ((green_count - (2 * red_count)) / total_count) * 100
                # Scale to 0-10 range
                scaled_score = round((raw_score + 200) * (10/300), 2)
                
                scores.append({
                    'Location': location,
                    'Cat_A': scaled_score,
                    'Green_Count': green_count,
                    'Red_Count': red_count,
                    'Total_SKUs': total_count
                })
        
        return pd.DataFrame(scores)


### Category B Scorer

import pandas as pd
import numpy as np
from scipy import stats
import logging

class CategoryBScorer:
    def __init__(self):
        self.brand_attributes = ['Taste', 'Thickness', 'Flavor', 'Length']
        
    def calculate_scores(self, market_pmi, market_comp):
        """Calculate Category B scores based on segment distribution"""
        try:
            scores = []
            
            # Process each location
            for location in market_pmi['Location'].unique():
                # Get PMI data for location
                pmi_data = market_pmi[market_pmi['Location'] == location]
                comp_data = market_comp[market_comp['Location'] == location]
                
                # Calculate segment distributions
                pmi_dist = self._get_segment_distribution(pmi_data)
                comp_dist = self._get_segment_distribution(comp_data)
                
                # Calculate correlation
                if not (pmi_dist.empty or comp_dist.empty):
                    score = self._calculate_correlation(pmi_dist, comp_dist)
                    scores.append({
                        'Location': location,
                        'Cat_B': score
                    })
                
            return pd.DataFrame(scores)
            
        except Exception as e:
            logging.error(f"Error in Category B scoring: {str(e)}")
            raise
            
    def _get_segment_distribution(self, df):
        """Calculate segment distribution for a dataset"""
        if df.empty:
            return pd.Series()
            
        # Create segment identifier
        df['Segment'] = df[self.brand_attributes].astype(str).agg('_'.join, axis=1)
        
        # Calculate distribution
        dist = df.groupby('Segment')['SKU'].count()
        return dist / dist.sum() if dist.sum() > 0 else dist
        
    def _calculate_correlation(self, dist1, dist2):
        """Calculate correlation between two distributions"""
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

### Category C Scorer

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
    
### Category D Scorer


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
