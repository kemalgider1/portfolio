import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CategoryCScorer:
    """Calculates Category C scores based on passenger and volume data."""

    current_year: int = 2024

    def validate_inputs(self, *dataframes: pd.DataFrame) -> bool:
        """Validates input dataframes."""
        return all(not df.empty for df in dataframes)

    def merge_data(self, pax_data: pd.DataFrame, mrk_nat_map: pd.DataFrame,
                  iata_location: pd.DataFrame) -> pd.DataFrame:
        """Merges input dataframes."""
        try:
            merged = pax_data.merge(mrk_nat_map, how='left', on='Location')
            return merged.merge(iata_location, how='left', on='Location')
        except Exception as e:
            logger.error(f"Error merging data: {e}")
            raise

    def calculate_location_score(self, pax_volume: float, df_volume: float) -> float:
        """Calculates score for a single location."""
        if pax_volume > 0 and df_volume > 0:
            return min((df_volume / pax_volume) * 10, 10)
        return 0.0

    def calculate_scores(self, pax_data: pd.DataFrame, df_vols: pd.DataFrame,
                        mrk_nat_map: pd.DataFrame, iata_location: pd.DataFrame) -> pd.DataFrame:
        """Calculates Category C scores for all locations."""
        try:
            logger.info(f"Processing data - shapes: PAX:{pax_data.shape}, "
                       f"Vols:{df_vols.shape}, Map:{mrk_nat_map.shape}, "
                       f"IATA:{iata_location.shape}")

            if not self.validate_inputs(pax_data, df_vols, mrk_nat_map, iata_location):
                raise ValueError("Invalid input data")

            merged_df = self.merge_data(pax_data, mrk_nat_map, iata_location)

            scores = []
            for location in df_vols['Location'].unique():
                try:
                    location_data = merged_df[merged_df['Location'] == location]
                    if location_data.empty:
                        continue

                    pax_volume = location_data['PAX_Volume'].sum()
                    df_volume = df_vols[df_vols['Location'] == location]['2024 Volume'].sum()

                    score = self.calculate_location_score(pax_volume, df_volume)
                    if score > 0:
                        scores.append({'Location': location, 'Cat_C': score})

                except Exception as e:
                    logger.error(f"Error processing location {location}: {e}")
                    continue

            return pd.DataFrame(scores)

        except Exception as e:
            logger.error(f"Error in Category C scoring: {e}")
            return pd.DataFrame(columns=['Location', 'Cat_C'])