import pandas as pd
import numpy as np
from scipy import stats
import logging


class CategoryCScorer:
    def __init__(self, current_year=2024):
        self.current_year = current_year
        self.column_mappings = {
            'LOCATION': 'Location',
            'IATA': 'IATA',
            'NATIONALITY': 'Nationality',
            'PAX': 'Pax',
            'MARKET': 'Market',
            'VOLUME': 'Volume'
        }

    def _standardize_columns(self, df, columns_to_rename=None):
        """Standardize column names to match expected format"""
        if columns_to_rename is None:
            columns_to_rename = self.column_mappings

        df = df.copy()
        rename_dict = {
            col: target
            for col in df.columns
            for target in columns_to_rename.values()
            if col.upper() == target.upper()
        }
        return df.rename(columns=rename_dict)

    def calculate_scores(self, pax_data, df_vols, mrk_nat_map, iata_location):
        """Calculate Category C scores based on PAX alignment"""
        try:
            # Standardize column names
            iata_location = self._standardize_columns(iata_location)
            df_vols = self._standardize_columns(df_vols)
            pax_data = self._standardize_columns(pax_data)
            mrk_nat_map = self._standardize_columns(mrk_nat_map)

            # Validate required columns
            self._validate_data_structure(
                pax_data=pax_data,
                df_vols=df_vols,
                iata_location=iata_location,
                mrk_nat_map=mrk_nat_map
            )

            # Process PAX data
            pax_processed = self._process_pax_data(
                pax_data=pax_data,
                mrk_nat_map=mrk_nat_map,
                iata_location=iata_location
            )

            scores = []
            for location in df_vols['Location'].unique():
                try:
                    # Get location data
                    loc_pax = pax_processed[pax_processed['Location'] == location]
                    loc_vols = df_vols[df_vols['Location'] == location]

                    if not (loc_pax.empty or loc_vols.empty):
                        # Calculate distributions
                        pax_dist = self._calculate_pax_distribution(loc_pax)
                        vol_dist = self._calculate_volume_distribution(loc_vols)

                        # Calculate score
                        score = self._calculate_correlation(pax_dist, vol_dist)
                        scores.append({
                            'Location': location,
                            'Cat_C': score
                        })
                except Exception as e:
                    logging.warning(f"Error processing location {location}: {str(e)}")
                    continue

            if not scores:
                raise ValueError("No valid scores calculated")

            return pd.DataFrame(scores)

        except Exception as e:
            logging.error(f"Error in Category C scoring: {str(e)}")
            raise

    def _validate_data_structure(self, pax_data, df_vols, iata_location, mrk_nat_map):
        """Validate input data structure"""
        # Check pax_data
        required_pax_cols = {'Year', 'IATA', 'Nationality', 'Pax'}
        if not required_pax_cols.issubset(pax_data.columns):
            raise ValueError(f"Missing PAX columns: {required_pax_cols - set(pax_data.columns)}")

        # Check volumes data
        required_vol_cols = {'Location', 'TMO', f'{self.current_year} Volume'}
        if not required_vol_cols.issubset(df_vols.columns):
            raise ValueError(f"Missing volume columns: {required_vol_cols - set(df_vols.columns)}")

        # Check IATA mapping
        required_iata_cols = {'Location', 'IATA'}
        if not required_iata_cols.issubset(iata_location.columns):
            raise ValueError(f"Missing IATA columns: {required_iata_cols - set(iata_location.columns)}")

        # Check nationality mapping
        required_nat_cols = {'PASSENGER_NATIONALITY', 'PASSENGER_COUNTRY_NAME'}
        if not required_nat_cols.issubset(mrk_nat_map.columns):
            raise ValueError(f"Missing nationality columns: {required_nat_cols - set(mrk_nat_map.columns)}")

    def _process_pax_data(self, pax_data, mrk_nat_map, iata_location):
        """Process and prepare PAX data"""
        # Filter current year
        df = pax_data[pax_data['Year'] == self.current_year].copy()

        # Add mappings
        df = df.merge(
            mrk_nat_map,
            left_on='Nationality',
            right_on='PASSENGER_NATIONALITY',
            how='left'
        )

        df = df.merge(
            iata_location[['IATA', 'Location']],
            on='IATA',
            how='inner'
        )

        # Calculate shares
        df['Share'] = df.groupby('Location')['Pax'].transform(
            lambda x: x / x.sum()
        )

        return df

    def _calculate_pax_distribution(self, pax_data):
        """Calculate PAX distribution by nationality"""
        return pax_data.groupby('Nationality')['Share'].sum()

    def _calculate_volume_distribution(self, vol_data):
        """Calculate volume distribution by segment"""
        if vol_data.empty:
            return pd.Series()

        total_vol = vol_data[f'{self.current_year} Volume'].sum()
        if total_vol > 0:
            return vol_data.groupby('TMO')[f'{self.current_year} Volume'].sum() / total_vol
        return pd.Series()

    def _calculate_correlation(self, dist1, dist2):
        """Calculate correlation between distributions"""
        try:
            if dist1.empty or dist2.empty:
                return 0

            # Align distributions
            all_categories = sorted(set(dist1.index) | set(dist2.index))
            dist1 = dist1.reindex(all_categories, fill_value=0)
            dist2 = dist2.reindex(all_categories, fill_value=0)

            # Calculate correlation if we have variation
            if dist1.std() > 0 and dist2.std() > 0:
                corr, _ = stats.spearmanr(dist1, dist2)
                r2 = corr ** 2 if not np.isnan(corr) else 0
            else:
                r2 = 0

            return round(r2 * 10, 2)  # Scale to 0-10 range

        except Exception as e:
            logging.error(f"Error calculating correlation: {str(e)}")
            return 0