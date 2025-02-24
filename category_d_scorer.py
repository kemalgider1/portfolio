import pandas as pd
import numpy as np
from scipy import stats
import logging
import warnings


class CategoryDScorer:
    """Category D scorer for calculating location cluster-based scores"""

    def __init__(self):
        self.current_year = 2024
        self.brand_attributes = ['Taste', 'Thickness', 'Flavor', 'Length']
        self.required_columns = {
            'df_vols': ['Location', 'TMO', 'CR_BrandId', f'{self.current_year} Volume'],
            'similarity': ['IATA', 'CLUSTER_IATA', 'SIMILARITY_SCORE'],
            'iata_location': ['LOCATION', 'IATA']
        }

    def calculate_scores(self, df_vols, similarity_file, iata_location):
        """Calculate Category D scores based on location clusters"""
        try:
            # Standardize column names
            df_vols = self._standardize_columns(df_vols, 'df_vols')
            similarity_file = self._standardize_columns(similarity_file, 'similarity')
            iata_location = self._standardize_columns(iata_location, 'iata_location')

            # Rename LOCATION to Location for consistency
            iata_location = iata_location.rename(columns={'LOCATION': 'Location'})

            # Validate input data
            self._validate_input_data(df_vols, similarity_file, iata_location)

            # Process similarity data to get cluster mappings
            cluster_mappings = self._process_similarity_data(similarity_file, iata_location)

            scores = []
            # Track segment identifiers
            all_segment_cols = [col for col in df_vols.columns if col in self.brand_attributes]
            if not all_segment_cols:
                # Use CR_BrandId as fallback segment identifier
                all_segment_cols = ['CR_BrandId']

            for location in df_vols['Location'].unique():
                try:
                    # Get cluster locations for current location
                    cluster_locs = cluster_mappings.get(location, [])

                    if cluster_locs:
                        # Get data for location and clusters
                        loc_data = df_vols[df_vols['Location'] == location].copy()
                        cluster_data = df_vols[df_vols['Location'].isin(cluster_locs)].copy()

                        # Calculate distributions
                        loc_dist = self._calculate_distribution(loc_data, all_segment_cols)
                        cluster_dist = self._calculate_distribution(cluster_data, all_segment_cols)

                        # Calculate correlation score
                        score = self._calculate_correlation(loc_dist, cluster_dist)

                        # Add debug logging
                        logging.debug(f"Location: {location}")
                        logging.debug(f"Location dist sum: {loc_dist.sum():.2f}")
                        logging.debug(f"Cluster dist sum: {cluster_dist.sum():.2f}")
                        logging.debug(f"Calculated score: {score:.2f}")

                        scores.append({
                            'Location': location,
                            'Cat_D': score
                        })

                except Exception as e:
                    logging.warning(f"Error processing location {location}: {str(e)}")
                    continue

            if not scores:
                raise ValueError("No valid scores calculated")

            return pd.DataFrame(scores)

        except Exception as e:
            logging.error(f"Error in Category D scoring: {str(e)}")
            raise

    def _standardize_columns(self, df, context=""):
        """Standardize column names to be case-insensitive"""
        df = df.copy()
        column_map = {}

        # Create case-insensitive column mapping
        for col in df.columns:
            for req_col in self.required_columns.get(context, []):
                if col.upper() == req_col.upper():
                    column_map[col] = req_col

        # Rename columns
        if column_map:
            df = df.rename(columns=column_map)

        return df

    def _validate_input_data(self, df_vols, similarity_file, iata_location):
        """Validate input data structure and content"""
        # Check required columns
        for df_name, cols in self.required_columns.items():
            df = {'df_vols': df_vols,
                  'similarity': similarity_file,
                  'iata_location': iata_location}[df_name]

            # Case-insensitive column check
            df_cols_upper = set(col.upper() for col in df.columns)
            req_cols_upper = set(col.upper() for col in cols)
            missing = req_cols_upper - df_cols_upper

            if missing:
                raise ValueError(f"Missing columns in {df_name}: {missing}")

        # Validate volume data
        if df_vols[f'{self.current_year} Volume'].isnull().any():
            raise ValueError("Found null values in volume data")

    def _process_similarity_data(self, similarity_file, iata_location):
        """Process similarity file to get cluster mappings"""
        # Create location to cluster locations mapping
        cluster_mappings = {}

        # Sort similarity scores and keep top 4 similar locations
        similarity_ranked = similarity_file.sort_values(
            ['IATA', 'SIMILARITY_SCORE'],
            ascending=[True, False]
        )
        similarity_ranked['rank'] = similarity_ranked.groupby('IATA').cumcount() + 1
        similarity_top4 = similarity_ranked[similarity_ranked['rank'] <= 4]

        # Create IATA to Location mapping
        iata_to_loc = dict(zip(iata_location['IATA'], iata_location['Location']))

        # Build cluster mappings
        for iata in similarity_top4['IATA'].unique():
            if iata not in iata_to_loc:
                continue

            location = iata_to_loc[iata]
            cluster_iatas = similarity_top4[
                similarity_top4['IATA'] == iata
                ]['CLUSTER_IATA'].tolist()

            # Get cluster locations
            cluster_locations = [
                iata_to_loc[cluster_iata]
                for cluster_iata in cluster_iatas
                if cluster_iata in iata_to_loc
            ]

            if cluster_locations:
                cluster_mappings[location] = cluster_locations

        return cluster_mappings

    def _calculate_distribution(self, df, segment_cols):
        """Calculate volume distribution by segment"""
        if df.empty:
            return pd.Series()

        try:
            # Create segment identifier
            if len(segment_cols) > 1:
                df.loc[:, 'Segment'] = df[segment_cols].astype(str).agg('-'.join, axis=1)
            else:
                df.loc[:, 'Segment'] = df[segment_cols[0]].astype(str)

            # Calculate volume by segment
            vol_by_segment = df.groupby('Segment')[f'{self.current_year} Volume'].sum()

            # Calculate distribution
            total_vol = vol_by_segment.sum()
            if total_vol > 0:
                return vol_by_segment / total_vol
            return vol_by_segment

        except Exception as e:
            logging.error(f"Error calculating distribution: {str(e)}")
            return pd.Series()

    def _calculate_correlation(self, dist1, dist2):
        """Calculate correlation between distributions"""
        try:
            if dist1.empty or dist2.empty:
                return 0

            # Get all unique segments
            all_segments = sorted(set(dist1.index) | set(dist2.index))

            # Align distributions
            dist1 = dist1.reindex(all_segments, fill_value=0)
            dist2 = dist2.reindex(all_segments, fill_value=0)

            # Add small constant to avoid zero variance
            epsilon = 1e-10
            dist1 = dist1 + epsilon
            dist2 = dist2 + epsilon

            # Normalize after adding epsilon
            dist1 = dist1 / dist1.sum()
            dist2 = dist2 / dist2.sum()

            # Check for constant arrays
            if dist1.std() == 0 or dist2.std() == 0:
                return 0

            # Calculate correlation using spearman
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                corr, _ = stats.spearmanr(dist1, dist2)
                r2 = corr ** 2 if not np.isnan(corr) else 0

            return round(r2 * 10, 2)  # Scale to 0-10 range

        except Exception as e:
            logging.error(f"Error calculating correlation: {str(e)}")
            return 0