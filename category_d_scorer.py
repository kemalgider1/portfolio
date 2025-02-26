import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import logging

class CategoryDScorer:
    def __init__(self):
        self.model = LinearRegression()

    def calculate_scores(self, df_vols, similarity_file, iata_location):
        """
        Calculate Category D scores based on cluster similarity comparison

        Parameters:
        -----------
        df_vols : DataFrame
            Duty free volumes data
        similarity_file : DataFrame
            Cluster similarity mapping
        iata_location : DataFrame
            IATA code to location mapping

        Returns:
        --------
        DataFrame
            Scores for Category D with columns ['Location', 'Cat_D']
        """
        try:
            scores = []

            # Process each location
            for location in df_vols['Location'].unique():
                try:
                    # Get IATA code for location
                    iata_info = iata_location[iata_location['Location'] == location]
                    if iata_info.empty:
                        continue
                    iata = iata_info['IATA'].iloc[0]

                    # Get cluster IATAs for this location
                    clusters = similarity_file[similarity_file['IATA'] == iata]
                    if clusters.empty:
                        continue

                    # Get volume data for main location
                    main_vols = df_vols[df_vols['Location'] == location].copy()
                    if main_vols.empty:
                        continue

                    # Get cluster locations
                    cluster_locations = iata_location[iata_location['IATA'].isin(clusters['CLUSTER_IATA'])]['Location'].tolist()
                    cluster_vols = df_vols[df_vols['Location'].isin(cluster_locations)].copy()

                    if cluster_vols.empty:
                        continue

                    # Calculate distributions
                    main_total = main_vols['2024 Volume'].sum()
                    if main_total == 0:
                        continue
                    main_dist = main_vols.groupby('CR_BrandId')['2024 Volume'].sum() / main_total

                    cluster_total = cluster_vols['2024 Volume'].sum()
                    if cluster_total == 0:
                        continue
                    cluster_dist = cluster_vols.groupby('CR_BrandId')['2024 Volume'].sum() / cluster_total

                    # Find common SKUs
                    common_skus = list(set(main_dist.index) & set(cluster_dist.index))

                    if len(common_skus) >= 2:
                        # Prepare data for regression
                        X = main_dist[common_skus].values.reshape(-1, 1)
                        y = cluster_dist[common_skus].values

                        # Calculate R-squared
                        self.model.fit(X, y)
                        r_squared = max(0, self.model.score(X, y))

                        # Scale to 0-10
                        score = round(r_squared * 10, 2)

                        scores.append({
                            'Location': location,
                            'Cat_D': score
                        })

                except Exception as e:
                    logging.warning(f"Error processing location {location}: {str(e)}")
                    continue

            if scores:
                return pd.DataFrame(scores)
            else:
                return pd.DataFrame(columns=['Location', 'Cat_D'])

        except Exception as e:
            logging.error(f"Error in Category D scoring: {str(e)}")
            return pd.DataFrame(columns=['Location', 'Cat_D'])