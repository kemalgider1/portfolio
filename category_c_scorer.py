import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import logging


class CategoryCScorer:
    """
    Calculates Category C scores based on passenger mix alignment with duty-free sales.
    Simplified implementation based directly on the 2023 code approach.
    """

    def __init__(self, current_year=2024):
        self.current_year = current_year
        self.model = LinearRegression()
        self.logger = logging.getLogger(__name__)

    def calculate_scores(self, pax_data, df_vols, mrk_nat_map, iata_location):
        """
        Calculate Category C scores based on passenger mix and volume data

        Parameters:
        -----------
        pax_data : DataFrame
            Passenger data
        df_vols : DataFrame
            Duty-free volumes data
        mrk_nat_map : DataFrame
            Mapping of Nationality to Country
        iata_location : DataFrame
            Mapping of IATA code to Location

        Returns:
        --------
        DataFrame
            Scores for Category C with columns ['Location', 'Cat_C']
        """
        try:
            # Step 1: Filter to locations with IATA codes
            locations_with_iata = iata_location['Location'].unique()
            df_vols_filtered = df_vols[df_vols['Location'].isin(locations_with_iata)].copy()

            # Step 2: Join volume data with IATA codes
            df_with_iata = df_vols_filtered.merge(iata_location, on='Location', how='left')

            # Step 3: Calculate real distribution
            # Find the volume column
            volume_columns = [c for c in df_with_iata.columns if 'Volume' in c]
            volume_col = volume_columns[0] if volume_columns else 'Volume'

            # Calculate distribution by IATA and SKU
            real_dist = df_with_iata.groupby(['IATA', 'CR_BrandId'])[volume_col].sum().reset_index()

            # Step 4: Prepare PAX distribution
            # Join nationality data
            pax_with_country = pax_data.merge(
                mrk_nat_map,
                left_on='Nationality',
                right_on='PASSENGER_NATIONALITY' if 'PASSENGER_NATIONALITY' in mrk_nat_map.columns else 'Nationality',
                how='left'
            )

            # Group by IATA and country
            country_col = 'PASSENGER_COUNTRY_NAME' if 'PASSENGER_COUNTRY_NAME' in pax_with_country.columns else 'Countries'
            if country_col not in pax_with_country.columns:
                country_col = [col for col in pax_with_country.columns if 'COUNTRY' in col.upper()
                               or 'NATION' in col.upper()]
                country_col = country_col[0] if country_col else 'Nationality'

            pax_dist = pax_with_country.groupby(['IATA', country_col])['Pax'].sum().reset_index()

            # Step 5: Calculate scores
            scores = []

            for iata in real_dist['IATA'].unique():
                try:
                    # Get location name
                    location_info = iata_location[iata_location['IATA'] == iata]
                    if location_info.empty:
                        continue
                    location = location_info['Location'].iloc[0]

                    # Get distributions - create a proper copy to avoid SettingWithCopyWarning
                    real_values = real_dist[real_dist['IATA'] == iata].copy()
                    pax_values = pax_dist[pax_dist['IATA'] == iata].copy()

                    # Skip if not enough data
                    if len(real_values) < 3 or len(pax_values) < 3:
                        scores.append({'Location': location, 'Cat_C': 0})
                        continue

                    # Calculate ranks - using .loc to avoid SettingWithCopyWarning
                    real_values.loc[:, 'Rank'] = real_values[volume_col].rank(ascending=False)
                    pax_values.loc[:, 'Rank'] = pax_values['Pax'].rank(ascending=False)

                    # Get the minimum length of both distributions
                    min_length = min(len(real_values), len(pax_values))

                    if min_length >= 3:
                        # Get the top ranked items from both distributions
                        real_ranks = real_values.sort_values('Rank').head(min_length)['Rank'].values
                        pax_ranks = pax_values.sort_values('Rank').head(min_length)['Rank'].values

                        # Use the model to calculate R-squared
                        X = np.array(real_ranks).reshape(-1, 1)
                        y = np.array(pax_ranks)

                        # Fit the model and get R-squared
                        self.model.fit(X, y)
                        r_squared = max(0, self.model.score(X, y))

                        # Scale to 0-10
                        score = round(r_squared * 10, 2)

                        scores.append({'Location': location, 'Cat_C': score})
                    else:
                        scores.append({'Location': location, 'Cat_C': 0})

                except Exception as e:
                    self.logger.warning(f"Error calculating score for {iata}: {str(e)}")
                    # Make sure we still add the location with a zero score
                    scores.append({'Location': location, 'Cat_C': 0})

            return pd.DataFrame(scores)

        except Exception as e:
            self.logger.error(f"Error in Category C scoring: {str(e)}")
            return pd.DataFrame(columns=['Location', 'Cat_C'])