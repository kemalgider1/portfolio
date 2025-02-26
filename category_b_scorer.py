import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import logging


class CategoryBScorer:
    def __init__(self):
        self.brand_attributes = ['Taste', 'Thickness', 'Flavor', 'Length']
        self.model = LinearRegression()

    def calculate_scores(self, market_pmi, market_comp):
            """
            Calculate Category B scores based on segment distribution

            Parameters:
            -----------
            market_pmi : DataFrame
                PMI market data with PMI Total column
            market_comp : DataFrame
                Competitor market data with Comp Total column

            Returns:
            --------
            DataFrame
                Scores for Category B with columns ['Location', 'Cat_B']
            """
            scores = []
            segments = ['Taste', 'Thickness', 'Flavor', 'Length']

            # Process each location
            for location in market_pmi['Location'].unique():
                try:
                    # Create copies of filtered data
                    pmi_data = market_pmi[market_pmi['Location'] == location].copy()
                    comp_data = market_comp[market_comp['Location'] == location].copy()

                    # Create segment combinations
                    pmi_data.loc[:, 'Segment'] = pmi_data.apply(
                        lambda x: '-'.join(str(x[col]) for col in segments), axis=1
                    )
                    comp_data.loc[:, 'Segment'] = comp_data.apply(
                        lambda x: '-'.join(str(x[col]) for col in segments), axis=1
                    )

                    # Calculate distributions using existing volume columns
                    pmi_dist = pmi_data.groupby('Segment')['PMI Total'].sum()
                    comp_dist = comp_data.groupby('Segment')['Comp Total'].sum()

                    # Normalize distributions
                    pmi_dist = pmi_dist / pmi_dist.sum() if pmi_dist.sum() > 0 else pmi_dist
                    comp_dist = comp_dist / comp_dist.sum() if comp_dist.sum() > 0 else comp_dist

                    # Calculate R-squared
                    common_segments = set(pmi_dist.index) & set(comp_dist.index)
                    if common_segments and len(common_segments) > 1:
                        pmi_values = pmi_dist[list(common_segments)]
                        comp_values = comp_dist[list(common_segments)]

                        # Calculate correlation coefficient
                        r = np.corrcoef(pmi_values, comp_values)[0, 1]
                        r_squared = r ** 2 if not np.isnan(r) else 0

                        # Scale to 0-10
                        score = r_squared * 10
                    else:
                        score = 0

                    scores.append({'Location': location, 'Cat_B': score})

                except Exception as e:
                    print(f"Error processing location {location}: {str(e)}")
                    scores.append({'Location': location, 'Cat_B': 0})

            return pd.DataFrame(scores)
