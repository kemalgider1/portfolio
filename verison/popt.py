import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import warnings
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    filename=f'portfolio_optimization_{datetime.now().strftime("%Y%m%d")}.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class DataValidation:
    """Validates data structures for portfolio optimization"""

    def __init__(self):
        self.current_year = 2024
        self.required_structures = {
            'MC_per_Product_SQL.csv': {
                'required_cols': ['DF_Market', 'Location', 'SKU', 'CR_BrandId',
                                  '2024 MC', '2024 NOR'],
                'min_rows': 10000,
                'key_types': {
                    'CR_BrandId': 'float64',
                    'Location': 'object'
                }
            },
            'df_vols_query.csv': {
                'required_cols': ['DF_Market', 'Location', 'TMO', 'Brand Family',
                                  'CR_BrandId', 'SKU', '2024 Volume'],
                'min_rows': 30000,
                'key_types': {
                    'CR_BrandId': 'int64',
                    'TMO': 'object'
                }
            },
            'Market_Summary_PMI.csv': {
                'required_cols': ['DF_Market', 'Location', 'TMO', 'Flavor',
                                  'Taste', 'Thickness', 'Length', 'PMI_Seg_SKU'],
                'min_rows': 1500,
                'value_checks': {
                    'TMO': ['PMI']
                }
            },
            'Market_Summary_Comp.csv': {
                'required_cols': ['DF_Market', 'Location', 'TMO', 'Flavor',
                                  'Taste', 'Thickness', 'Length', 'Comp_Seg_SKU'],
                'min_rows': 5000,
                'value_checks': {
                    'TMO': lambda x: x != 'PMI'
                }
            },
            'Pax_Nat.csv': {
                'required_cols': ['Year', 'IATA', 'Market', 'Nationality', 'Pax'],
                'min_rows': 80000,
                'key_types': {
                    'Year': 'int64',
                    'Pax': 'float64'
                }
            }
        }

    def validate_file(self, filename, df):
        """Validate a single file against requirements.txt"""
        if filename not in self.required_structures:
            return True

        requirements = self.required_structures[filename]
        validation_results = []

        # Check required columns
        has_cols = all(col in df.columns for col in requirements['required_cols'])
        validation_results.append(('required_columns', has_cols))

        # Check minimum rows
        has_min_rows = len(df) >= requirements['min_rows']
        validation_results.append(('minimum_rows', has_min_rows))

        # Check data types
        if 'key_types' in requirements:
            type_checks = []
            for col, dtype in requirements['key_types'].items():
                if col in df.columns:
                    type_checks.append(df[col].dtype.name == dtype)
            validation_results.append(('data_types', all(type_checks)))

        # Check value constraints
        if 'value_checks' in requirements:
            value_checks = []
            for col, check in requirements['value_checks'].items():
                if col in df.columns:
                    if callable(check):
                        value_checks.append(df[check(col)].all())
                    else:
                        value_checks.append(df[col].isin(check).all())
            validation_results.append(('value_checks', all(value_checks)))

        return all(result[1] for result in validation_results)


class PortfolioOptimization:
    def __init__(self, current_year=2024):
        self.current_year = current_year
        self.previous_year = current_year - 1
        self.brand_attributes = ['Taste', 'Thickness', 'Flavor', 'Length']
        self.validator = DataValidation()

    def load_and_validate_data(self):
        """Load and validate all required data"""
        try:
            # Load core data files
            self.mc_per_product = pd.read_csv('MC_per_Product_SQL.csv')
            self.df_vols = pd.read_csv('df_vols_query.csv')
            self.market_pmi = pd.read_csv('Market_Summary_PMI.csv')
            self.market_comp = pd.read_csv('Market_Summary_Comp.csv')
            self.pax_data = pd.read_csv('Pax_Nat.csv')

            # Load mapping and reference files
            self.mrk_nat_map = pd.read_csv('mrk_nat_map_query.csv')
            self.iata_location = pd.read_csv('iata_location_query.csv')
            self.similarity_file = pd.read_csv('similarity_file.csv')
            self.selma_map = pd.read_csv('SELMA_DF_map_query.csv')

            # Validate core data
            validation_checks = {
                'MC_per_Product_SQL.csv': self.mc_per_product,
                'df_vols_query.csv': self.df_vols,
                'Market_Summary_PMI.csv': self.market_pmi,
                'Market_Summary_Comp.csv': self.market_comp,
                'Pax_Nat.csv': self.pax_data
            }

            for filename, df in validation_checks.items():
                if not self.validator.validate_file(filename, df):
                    logging.error(f"Validation failed for {filename}")
                    return False

            # Process market mix data
            self._process_market_data()

            return True

        except Exception as e:
            logging.error(f"Error loading data: {e}")
            return False

    def _process_market_data(self):
        """Process and combine market data"""
        # Prepare PMI data
        pmi_data = self.market_pmi.copy()
        pmi_data['Data_Source'] = 'PMI'
        pmi_data = pmi_data.rename(columns={
            'PMI_Seg_SKU': 'SKU_Count',
            'PMI Total': 'Volume',
            'SoM_PMI': 'Share'
        })

        # Prepare competitor data
        comp_data = self.market_comp.copy()
        comp_data['Data_Source'] = 'Competitor'
        comp_data = comp_data.rename(columns={
            'Comp_Seg_SKU': 'SKU_Count',
            'Comp Total': 'Volume',
            'SoM_Comp': 'Share'
        })

        # Combine data
        self.market_mix = pd.concat([pmi_data, comp_data], ignore_index=True)

        # Add segment information
        self.market_mix['Segment'] = self.market_mix.apply(
            lambda x: f"{x['Taste']}_{x['Thickness']}_{x['Length']}",
            axis=1
        )

        # Calculate market metrics
        self._calculate_market_metrics()

    def _calculate_market_metrics(self):
        """Calculate market level metrics"""
        self.market_mix['Volume_Share'] = self.market_mix.groupby(
            ['Location', 'Data_Source']
        )['Volume'].transform(lambda x: x / x.sum())

        self.market_mix['SKU_Share'] = self.market_mix.groupby(
            ['Location', 'Data_Source']
        )['SKU_Count'].transform(lambda x: x / x.sum())

    def calculate_category_a(self):
        """Calculate Category A scores"""
        df = self.df_vols.copy()

        # Calculate volume metrics
        df['Volume_Share'] = df.groupby('Location')[f'{self.current_year} Volume'].transform(
            lambda x: x / x.sum()
        )

        # Calculate growth metrics
        for period in ['LY', 'CY']:
            year = self.previous_year if period == 'LY' else self.current_year
            df[f'{period}RevenueAvg'] = df[f'{year} Revenue'] / df[f'{year}Month']

        df['Growth'] = (df['CYRevenueAvg'] - df['LYRevenueAvg']) / df['LYRevenueAvg']

        # Calculate margin metrics
        df['Margin'] = np.where(
            df[f'{self.current_year} NOR'] <= 0,
            0,
            df[f'{self.current_year} MC'] / df[f'{self.current_year} NOR']
        )

        # Calculate flags
        df = self._calculate_flags(df)

        return self._calculate_category_a_scores(df)

    def _calculate_flags(self, df):
        """Calculate green and red flags"""
        # Calculate thresholds
        df['Volume_Rank'] = df.groupby('Location')[f'{self.current_year} Volume'].rank(
            ascending=False
        )
        df['Growth_Rank'] = df.groupby('Location')['Growth'].rank(ascending=False)
        df['Margin_Rank'] = df.groupby('Location')['Margin'].rank(ascending=False)

        # Calculate green flags
        df['Green_Flag'] = 0
        df.loc[
            (df['Volume_Rank'] <= df.groupby('Location').size() * 0.05) |
            ((df['Growth_Rank'] <= df.groupby('Location').size() * 0.05) &
             (df['Margin'] > df.groupby('Location')['Margin'].transform('mean'))),
            'Green_Flag'
        ] = 1

        # Calculate red flags
        df['Red_Flag'] = 0
        df.loc[
            (df['Volume_Rank'] > df.groupby('Location').size() * 0.75) |
            ((df['Growth_Rank'] > df.groupby('Location').size() * 0.75) &
             (df['Margin'] < df.groupby('Location')['Margin'].transform('mean'))),
            'Red_Flag'
        ] = 1

        return df

    def _calculate_category_a_scores(self, df):
        """Calculate final Category A scores"""
        scores = []
        for location in df['Location'].unique():
            location_data = df[df['Location'] == location]

            green_count = location_data['Green_Flag'].sum()
            red_count = location_data['Red_Flag'].sum()
            total_count = len(location_data)

            if total_count > 0:
                score = ((green_count - (2 * red_count)) / total_count) * 100
                # Scale to 0-10 range
                scaled_score = round((score + 200) * (10 / 300), 2)

                scores.append({
                    'Location': location,
                    'Cat_A': scaled_score,
                    'Green_Count': green_count,
                    'Red_Count': red_count
                })

        return pd.DataFrame(scores)

    def calculate_category_b(self):
        """Calculate Category B scores based on segment distribution"""
        scores = []

        for location in self.market_mix['Location'].unique():
            location_data = self.market_mix[self.market_mix['Location'] == location]

            # Calculate PMI distribution
            pmi_dist = location_data[location_data['Data_Source'] == 'PMI'].groupby(
                'Segment'
            )['Volume_Share'].sum()

            # Calculate competitor distribution
            comp_dist = location_data[location_data['Data_Source'] == 'Competitor'].groupby(
                'Segment'
            )['Volume_Share'].sum()

            # Calculate correlation
            if len(pmi_dist) > 1 and len(comp_dist) > 1:
                corr = np.corrcoef(pmi_dist, comp_dist)[0, 1]
                r2 = corr ** 2 if not np.isnan(corr) else 0
            else:
                r2 = 0

            scores.append({
                'Location': location,
                'Cat_B': round(r2 * 10, 2)
            })

        return pd.DataFrame(scores)

    def calculate_category_c(self):
        """Calculate Category C scores based on PAX data"""
        # Process PAX data
        pax_df = self._process_pax_data()

        scores = []
        for location in pax_df['Location'].unique():
            location_data = pax_df[pax_df['Location'] == location]

            # Calculate actual vs expected distributions
            actual_dist = self._calculate_actual_distribution(location_data)
            expected_dist = self._calculate_expected_distribution(location_data)

            # Calculate correlation
            if len(actual_dist) > 1 and len(expected_dist) > 1:
                corr = np.corrcoef(actual_dist, expected_dist)[0, 1]
                r2 = corr ** 2 if not np.isnan(corr) else 0
            else:
                r2 = 0

            scores.append({
                'Location': location,
                'Cat_C': round(r2 * 10, 2)
            })

        return pd.DataFrame(scores)

    def _process_pax_data(self):
        """Process passenger data"""
        pax_df = self.pax_data.merge(
            self.mrk_nat_map,
            how='left',
            left_on='Nationality',
            right_on='PASSENGER_NATIONALITY'
        )

        pax_df = pax_df.merge(
            self.iata_location[['IATA', 'Location']],
            how='left',
            on='IATA'
        )

        return pax_df

    def calculate_category_d(self):
        """Calculate Category D scores based on location clusters"""
        similarity_data = self._process_similarity_data()

        scores = []
        for location in similarity_data['Location'].unique():
            cluster_data = self._get_cluster_data(location, similarity_data)

            if not cluster_data.empty:
                # Calculate correlation between location and cluster
                location_dist = self._calculate_location_distribution(location)
                cluster_dist = self._calculate_cluster_distribution(cluster_data)

                if len(location_dist) > 1 and len(cluster_dist) > 1:
                    corr = np.corrcoef(location_dist, cluster_dist)[0, 1]
                    r2 = corr ** 2 if not np.isnan(corr) else 0
                else:
                    r2 = 0

                scores.append({
                    'Location': location,
                    'Cat_D': round(r2 * 10, 2)
                })

        return pd.DataFrame(scores)

    def _process_similarity_data(self):
        """Process similarity file for clustering"""
        similarity_data = self.similarity_file.copy()

        # Sort and rank similar locations
        similarity_data = similarity_data.sort_values(
            ['IATA', 'SIMILARITY_SCORE'],
            ascending=[True, False]
        )
        similarity_data['Rank'] = similarity_data.groupby('IATA').cumcount() + 1

        # Keep top 4 similar locations as per 2023
        similarity_data = similarity_data[similarity_data['Rank'] <= 4]

        # Add location information
        similarity_data = similarity_data.merge(
            self.iata_location[['IATA', 'Location']],
            how='left',
            on='IATA'
        )

        return similarity_data

    def _get_cluster_data(self, location, similarity_data):
        """Get market data for location's cluster"""
        cluster_iatas = similarity_data[
            similarity_data['Location'] == location
            ]['CLUSTER_IATA'].tolist()

        return self.market_mix[self.market_mix['IATA'].isin(cluster_iatas)]

    def calculate_final_scores(self):
        """Calculate final portfolio optimization scores"""
        try:
            # Load and validate data
            if not self.load_and_validate_data():
                return None

            # Calculate category scores
            cat_a_scores = self.calculate_category_a()
            cat_b_scores = self.calculate_category_b()
            cat_c_scores = self.calculate_category_c()
            cat_d_scores = self.calculate_category_d()

            # Combine scores
            final_scores = cat_a_scores.merge(
                cat_b_scores, on='Location', how='outer'
            ).merge(
                cat_c_scores, on='Location', how='outer'
            ).merge(
                cat_d_scores, on='Location', how='outer'
            )

            # Calculate final score
            score_columns = ['Cat_A', 'Cat_B', 'Cat_C', 'Cat_D']
            final_scores['Final_Score'] = final_scores[score_columns].mean(axis=1)

            # Add metrics and validate
            final_scores = self._add_metrics(final_scores)
            self._validate_scores(final_scores)

            return final_scores

        except Exception as e:
            logging.error(f"Error calculating final scores: {e}")
            return None

    def _add_metrics(self, scores_df):
        """Add volume and performance metrics"""
        # Calculate volume metrics
        volume_metrics = self.df_vols.groupby('Location').agg({
            f'{self.current_year} Volume': 'sum',
            'SKU': 'count',
            'TMO': lambda x: (x == 'PMI').mean()
        }).reset_index()

        volume_metrics.columns = [
            'Location', 'Total_Volume', 'SKU_Count', 'PMI_Share'
        ]

        # Add growth metrics
        growth_metrics = self.df_vols.groupby('Location')['Growth'].agg([
            'mean', 'std'
        ]).reset_index()

        # Merge metrics
        scores_df = scores_df.merge(
            volume_metrics, on='Location', how='left'
        ).merge(
            growth_metrics, on='Location', how='left'
        )

        return scores_df

    def _validate_scores(self, scores_df):
        """Validate final scores"""
        # Check score ranges
        for category in ['Cat_A', 'Cat_B', 'Cat_C', 'Cat_D', 'Final_Score']:
            if not all(scores_df[category].between(0, 10)):
                logging.warning(f"Invalid scores found in {category}")

        # Check data completeness
        missing_locations = set(self.df_vols['Location']) - set(scores_df['Location'])
        if missing_locations:
            logging.warning(f"Missing scores for locations: {missing_locations}")

        # Check correlation patterns
        score_corr = scores_df[['Cat_A', 'Cat_B', 'Cat_C', 'Cat_D']].corr()
        if (score_corr > 0.9).any().any():
            logging.warning("High correlation between category scores detected")

    def generate_reports(self, final_scores):
        """Generate analysis reports and visualizations"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save scores
        final_scores.to_csv(
            f'portfolio_scores_{self.current_year}_{timestamp}.csv',
            index=False
        )

        # Generate visualizations
        self._generate_visualizations(final_scores, timestamp)

        # Generate location reports
        self._generate_location_reports(final_scores, timestamp)

    def _generate_visualizations(self, scores_df, timestamp):
        """Generate analysis visualizations"""
        plt.style.use('seaborn')

        # Score distributions
        plt.figure(figsize=(12, 6))
        categories = ['Cat_A', 'Cat_B', 'Cat_C', 'Cat_D']
        plt.boxplot(
            [scores_df[cat] for cat in categories],
            labels=categories
        )
        plt.title('Category Score Distributions')
        plt.ylabel('Score')
        plt.savefig(f'category_scores_{timestamp}.png')
        plt.close()

        # Volume vs Score
        plt.figure(figsize=(10, 6))
        plt.scatter(
            scores_df['Total_Volume'],
            scores_df['Final_Score'],
            alpha=0.6
        )
        plt.xlabel('Total Volume')
        plt.ylabel('Final Score')
        plt.title('Volume vs Final Score Relationship')
        plt.savefig(f'volume_score_{timestamp}.png')
        plt.close()

    def _generate_location_reports(self, scores_df, timestamp):
        """Generate detailed location reports"""
        for location in scores_df['Location'].unique():
            location_data = scores_df[scores_df['Location'] == location]

            report = f"""
Location Analysis Report: {location}
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Performance Scores:
- Category A (Portfolio Performance): {location_data['Cat_A'].iloc[0]:.2f}
- Category B (Segment Distribution): {location_data['Cat_B'].iloc[0]:.2f}
- Category C (PAX Alignment): {location_data['Cat_C'].iloc[0]:.2f}
- Category D (Cluster Comparison): {location_data['Cat_D'].iloc[0]:.2f}
- Final Score: {location_data['Final_Score'].iloc[0]:.2f}

Volume Metrics:
- Total Volume: {location_data['Total_Volume'].iloc[0]:,.0f}
- SKU Count: {location_data['SKU_Count'].iloc[0]}
- PMI Share: {location_data['PMI_Share'].iloc[0]:.1%}

Growth Metrics:
- Average Growth: {location_data['mean'].iloc[0]:.1%}
- Growth Std Dev: {location_data['std'].iloc[0]:.1%}
"""

            # Save location report
            with open(f'location_report_{location}_{timestamp}.txt', 'w') as f:
                f.write(report)


def main():
    """Main execution function"""
    # Initialize optimizer
    optimizer = PortfolioOptimization(current_year=2024)

    # Calculate final scores
    final_scores = optimizer.calculate_final_scores()

    if final_scores is not None:
        # Generate reports
        optimizer.generate_reports(final_scores)
        logging.info("Portfolio optimization completed successfully")
    else:
        logging.error("Portfolio optimization failed")


if __name__ == "__main__":
    main()