import pandas as pd
import numpy as np
import logging

logging.basicConfig(
    filename='portfolio_scoring.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class DataLoader:
    def __init__(self, current_year=2024):
        self.current_year = current_year
        self.previous_year = current_year - 1
        
    def load_all_data(self):
        try:
            data = {
                'mc_per_product': pd.read_csv('MC_per_Product_SQL.csv'),
                'df_vols': pd.read_csv('df_vols_query.csv'),
                'market_pmi': pd.read_csv('Market_Summary_PMI.csv'),
                'market_comp': pd.read_csv('Market_Summary_Comp.csv'),
                'pax_data': pd.read_csv('Pax_Nat.csv'),
                'similarity_file': pd.read_csv('similarity_file.csv'),
                'iata_location': pd.read_csv('iata_location_query.csv'),
                'mrk_nat_map': pd.read_csv('mrk_nat_map_query.csv')
            }
            
            logging.info("Data loaded successfully")
            logging.info(f"Volume data shape: {data['df_vols'].shape}")
            logging.info(f"PMI market data shape: {data['market_pmi'].shape}")
            logging.info(f"Competitor data shape: {data['market_comp'].shape}")
            
            return data
            
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            return None

    def preprocess_data(self, data):
        try:
            # Clean and standardize columns
            data['df_vols'] = self._clean_volume_data(data['df_vols'])
            data['market_pmi'] = self._standardize_market_data(data['market_pmi'])
            data['market_comp'] = self._standardize_market_data(data['market_comp'])
            
            return data
            
        except Exception as e:
            logging.error(f"Error preprocessing data: {str(e)}")
            return None
            
    def _clean_volume_data(self, df):
        df = df.copy()
        # Handle missing values
        df = df.fillna({
            f'{self.current_year} Volume': 0,
            f'{self.current_year} Revenue': 0,
            f'{self.current_year}Month': 1  # Avoid division by zero
        })
        return df
        
    def _standardize_market_data(self, df):
        df = df.copy()
        # Standardize segment columns
        segment_cols = ['Taste', 'Thickness', 'Flavor', 'Length']
        for col in segment_cols:
            if col in df.columns:
                df[col] = df[col].fillna('Unknown')
                df[col] = df[col].str.strip()
        return df