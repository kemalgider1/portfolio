import pandas as pd
import numpy as np
from scipy import stats
import logging

class CategoryBScorer:
    def __init__(self):
        self.segment_cols = ['Taste', 'Thickness', 'Flavor', 'Length']
        
    def calculate_scores(self, market_pmi, market_comp):
        try:
            # Prepare segment distributions
            pmi_dist = self._prepare_distribution(market_pmi, 'PMI_Seg_SKU')
            comp_dist = self._prepare_distribution(market_comp, 'Comp_Seg_SKU')
            
            # Calculate scores
            scores = self._calculate_correlations(pmi_dist, comp_dist)
            
            return scores
            
        except Exception as e:
            logging.error(f"Error calculating Category B scores: {str(e)}")
            return None
            
    def _prepare_distribution(self, df, value_col):
        # Create segment identifier
        df = df.copy()
        df['Segment'] = df[self.segment_cols].apply(lambda x: '_'.join(x), axis=1)
        
        # Calculate distribution per location
        dist = df.pivot_table(
            index='Location',
            columns='Segment',
            values=value_col,
            aggfunc='sum',
            fill_value=0
        )
        
        # Normalize distributions
        row_sums = dist.sum(axis=1)
        dist = dist.div(row_sums, axis=0).fillna(0)
        
        return dist
        
    def _calculate_correlations(self, pmi_dist, comp_dist):
        scores = []
        
        # Align segments between PMI and competitors
        all_segments = sorted(set(pmi_dist.columns) | set(comp_dist.columns))
        pmi_dist = pmi_dist.reindex(columns=all_segments, fill_value=0)
        comp_dist = comp_dist.reindex(columns=all_segments, fill_value=0)
        
        for location in pmi_dist.index:
            if location in comp_dist.index:
                pmi_values = pmi_dist.loc[location]
                comp_values = comp_dist.loc[location]
                
                # Calculate correlation only if we have variation in both distributions
                if pmi_values.std() > 0 and comp_values.std() > 0:
                    # Use Spearman correlation to handle non-normal distributions
                    corr, _ = stats.spearmanr(pmi_values, comp_values)
                    r2 = corr ** 2 if not np.isnan(corr) else 0
                else:
                    r2 = 0
                    
                scores.append({
                    'Location': location,
                    'Cat_B': round(r2 * 10, 2)
                })
        
        return pd.DataFrame(scores)