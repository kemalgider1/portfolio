import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import logging


class CategoryDScorer:
    """
    Calculates Category D scores based on cluster similarity comparison,
    following the exact methodology from the 2023 implementation.
    """

    def __init__(self, current_year=2024):
        self.current_year = current_year
        self.model = LinearRegression()
        self.logger = logging.getLogger(__name__)

    def calculate_scores(self, df_vol_data, similarity_file, iata_location, selma_df_map):
        """
        Calculate Category D scores using the 2023 methodology

        Parameters:
        -----------
        df_vol_data : DataFrame - Duty-free volumes (df_vols_query.csv)
        similarity_file : DataFrame - Similarity mapping (similarity_file.csv)
        iata_location : DataFrame - IATA to location mapping (iata_location_query.csv)
        selma_df_map : DataFrame - Duty-free SELMA attributes (SELMA_DF_map_query.csv)

        Returns:
        --------
        DataFrame - Scores for Category D with columns ['Location', 'Cat_D']
        """
        try:
            self.logger.info("Starting Category D calculation with 2023 methodology")

            # Standardize column names
            iata_location = iata_location.rename(columns={
                'LOCATION': 'Location'
            }, errors='ignore')

            # Step 1: Filter similarity file to top 4 similar locations
            self.logger.info("Preparing cluster information")
            similarity_filtered = similarity_file[similarity_file['RANK'] <= 4].copy()

            # Step 2: Locate the volume column in df_vol_data
            volume_cols = [col for col in df_vol_data.columns if 'Volume' in col and str(self.current_year) in col]
            if not volume_cols:
                volume_cols = [col for col in df_vol_data.columns if 'Volume' in col]
            if not volume_cols:
                raise ValueError("No volume column found in df_vol_data")

            volume_col = volume_cols[0]

            # Step 3: Add SELMA attributes (segment info) to duty-free volumes
            self.logger.info("Adding segment attributes to duty-free volumes")
            df_with_segments = df_vol_data.merge(
                selma_df_map[['DF_Market', 'Location', 'CR_BrandId', 'Flavor', 'Taste', 'Thickness', 'Length']],
                on=['DF_Market', 'Location', 'CR_BrandId'],
                how='left'
            )

            # Step 4: Add IATA codes to duty-free volumes
            df_with_iata = df_with_segments.merge(iata_location, on='Location', how='left')

            # Filter to keep only rows with valid IATA codes
            df_with_iata = df_with_iata[df_with_iata['IATA'].notnull()]

            # Define segment attributes
            segment_attrs = ['Flavor', 'Taste', 'Thickness', 'Length']

            # Step 5: Calculate segment distribution for each location
            self.logger.info("Calculating segment distributions by location")

            # Calculate volume by segment
            location_segments = df_with_iata.groupby(['Location', 'IATA'] + segment_attrs)[
                volume_col].sum().reset_index()

            # Calculate total volume by location
            location_totals = location_segments.groupby(['Location', 'IATA'])[volume_col].sum().reset_index().rename(
                columns={volume_col: 'Total_Volume'})

            # Calculate segment percentage
            location_dist = location_segments.merge(location_totals, on=['Location', 'IATA'], how='left')
            location_dist['Segment_Pct'] = location_dist[volume_col] / location_dist['Total_Volume']

            # Step 6: Calculate scores by comparing with cluster locations
            self.logger.info("Calculating correlation with cluster locations")

            scores = []

            for iata in df_with_iata['IATA'].unique():
                try:
                    # Get location name
                    location_info = iata_location[iata_location['IATA'] == iata]
                    if location_info.empty:
                        continue
                    location = location_info['Location'].iloc[0]

                    # Get clusters for this location
                    clusters = similarity_filtered[similarity_filtered['IATA'] == iata]
                    if clusters.empty:
                        scores.append({'Location': location, 'Cat_D': 0})
                        continue

                    # Get cluster IATA codes
                    cluster_iatas = clusters['CLUSTER_IATA'].tolist()

                    # Get segment distribution for this location
                    location_data = location_dist[location_dist['IATA'] == iata].copy()
                    if location_data.empty:
                        scores.append({'Location': location, 'Cat_D': 0})
                        continue

                    # Get segment distribution for cluster locations
                    cluster_locations = iata_location[iata_location['IATA'].isin(cluster_iatas)]['Location'].tolist()
                    cluster_data = location_dist[location_dist['Location'].isin(cluster_locations)].copy()

                    if cluster_data.empty:
                        scores.append({'Location': location, 'Cat_D': 0})
                        continue

                    # Aggregate cluster data by segment
                    cluster_segments = cluster_data.groupby(segment_attrs)[volume_col].sum().reset_index()
                    cluster_total = cluster_segments[volume_col].sum()
                    cluster_segments['Cluster_Pct'] = cluster_segments[volume_col] / cluster_total

                    # Join location and cluster segment distributions
                    comparison = location_data.merge(
                        cluster_segments,
                        on=segment_attrs,
                        how='outer'
                    ).fillna(0)

                    # Ensure we have enough data points for correlation
                    if len(comparison) < 3:
                        scores.append({'Location': location, 'Cat_D': 0})
                        continue

                    # Calculate correlation
                    X = comparison['Segment_Pct'].values.reshape(-1, 1)
                    y = comparison['Cluster_Pct'].values

                    self.model.fit(X, y)
                    r_squared = max(0, self.model.score(X, y))

                    # Scale to 0-10
                    score = round(r_squared * 10, 2)

                    scores.append({'Location': location, 'Cat_D': score})

                except Exception as e:
                    self.logger.warning(f"Error calculating score for {iata}: {str(e)}")
                    # Make sure we still add the location with a zero score
                    try:
                        location = iata_location[iata_location['IATA'] == iata]['Location'].iloc[0]
                        scores.append({'Location': location, 'Cat_D': 0})
                    except:
                        pass

            return pd.DataFrame(scores)

        except Exception as e:
            self.logger.error(f"Error in Category D scoring: {str(e)}")
            return pd.DataFrame(columns=['Location', 'Cat_D'])