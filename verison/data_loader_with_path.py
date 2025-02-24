import pandas as pd
import numpy as np
import logging
import os
from pathlib import Path

logging.basicConfig(
    filename='portfolio_scoring.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class DataLoader:
    def __init__(self, current_year=2024, data_dir='data'):
        self.current_year = current_year
        self.previous_year = current_year - 1
        self.data_dir = data_dir
        
        # Validate data directory exists
        if not os.path.exists(data_dir):
            raise ValueError(f"Data directory '{data_dir}' not found")
            
    def _load_csv(self, filename):
        """Load CSV file from data directory"""
        file_path = os.path.join(self.data_dir, filename)
        try:
            df = pd.read_csv(file_path)
            logging.info(f"Successfully loaded {filename}")
            return df
        except Exception as e:
            logging.error(f"Error loading {filename}: {str(e)}")
            raise
            
    def load_all_data(self):
        """Load all required data files from data directory"""
        try:
            data = {
                'mc_per_product': self._load_csv('MC_per_Product_SQL.csv'),
                'df_vols': self._load_csv('df_vols_query.csv'),
                'market_pmi': self._load_csv('Market_Summary_PMI.csv'),
                'market_comp': self._load_csv('Market_Summary_Comp.csv'),
                'pax_data': self._load_csv('Pax_Nat.csv'),
                'similarity_file': self._load_csv('similarity_file.csv'),
                'iata_location': self._load_csv('iata_location_query.csv'),
                'mrk_nat_map': self._load_csv('mrk_nat_map_query.csv')
            }
            
            logging.info("All data files loaded successfully")
            logging.info(f"Volume data shape: {data['df_vols'].shape}")
            logging.info(f"PMI market data shape: {data['market_pmi'].shape}")
            logging.info(f"Competitor data shape: {data['market_comp'].shape}")
            
            return data
            
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            return None