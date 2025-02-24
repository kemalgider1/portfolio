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