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
        df = df.copy()  # Avoid SettingWithCopyWarning
        df['Segment'] = df[self.brand_attributes].astype(str).agg('-'.join, axis=1)

        # Calculate segment volume shares
        if 'PMI_Seg_SKU' in df.columns:
            vol_col = 'PMI_Seg_SKU'
        else:
            vol_col = 'Comp_Seg_SKU'

        # Calculate distribution
        dist = df.groupby('Segment')[vol_col].sum()
        return dist / dist.sum() if dist.sum() > 0 else dist

    def _calculate_correlation(self, dist1, dist2):
        """Calculate correlation between two distributions"""
        try:
            # Align distributions
            all_segments = sorted(set(dist1.index) | set(dist2.index))
            dist1 = dist1.reindex(all_segments, fill_value=0)
            dist2 = dist2.reindex(all_segments, fill_value=0)

            # Calculate correlation if we have variation
            if dist1.std() > 0 and dist2.std() > 0:
                # Use Spearman correlation for robustness
                corr, _ = stats.spearmanr(dist1, dist2)
                r2 = corr ** 2 if not np.isnan(corr) else 0
            else:
                r2 = 0

            return round(r2 * 10, 2)  # Scale to 0-10 range as per 2023

        except Exception as e:
            logging.error(f"Error calculating correlation: {str(e)}")
            return 0