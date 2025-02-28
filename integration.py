from category_a_scorer import CategoryAScorer
from category_b_scorer import CategoryBScorer
from category_c_scorer import CategoryCScorer
from category_d_scorer import CategoryDScorer
import pandas as pd
import logging
from datetime import datetime
import os


class PortfolioOptimizer:
    """
    Integrates the four category scorers (A, B, C, D) to produce combined portfolio optimization scores.
    Uses the 2023 methodology for Category C (Passenger Mix) and Category D (Location Clusters).
    """

    def __init__(self, config=None):
        """Initialize optimizer with optional config"""
        self.config = config or {}
        self.current_year = self.config.get('current_year', 2024)

        # Initialize scoring components
        self.cat_a_scorer = CategoryAScorer(self.current_year)
        self.cat_b_scorer = CategoryBScorer()
        self.cat_c_scorer = CategoryCScorer(self.current_year)
        self.cat_d_scorer = CategoryDScorer(self.current_year)

        # Set up logging
        log_level = self.config.get('log_level', logging.INFO)
        logging.basicConfig(level=log_level,
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

        self.logger.info(f"Portfolio Optimizer initialized for year {self.current_year}")

    def validate_data_sources(self, data_sources):
        """Validate required data sources are present and not empty"""
        required_sources = [
            'df_vols',  # Duty-free volumes
            'mc_per_product',  # Margin contribution data
            'market_pmi',  # PMI market data
            'market_comp',  # Competitor market data
            'pax_data',  # Passenger data
            'mrk_nat_map',  # Market nationality mapping
            'iata_location',  # IATA to location mapping
            'similarity_file',  # Cluster similarity data
            'country_figures',  # Country smoking statistics
            'dom_volumes',  # Domestic volumes
            'dom_products',  # Domestic products data
            'selma_df_map'  # Duty-free SELMA attributes
        ]

        missing_sources = []
        empty_sources = []

        for source in required_sources:
            if source not in data_sources:
                missing_sources.append(source)
            elif data_sources[source] is None or len(data_sources[source]) == 0:
                empty_sources.append(source)

        if missing_sources:
            self.logger.error(f"Missing required data sources: {', '.join(missing_sources)}")
            raise ValueError(f"Missing required data sources: {', '.join(missing_sources)}")

        if empty_sources:
            self.logger.warning(f"Empty data sources: {', '.join(empty_sources)}")

        self.logger.info("All required data sources validated")
        return True

    def calculate_scores(self, data_sources):
        """Calculate scores for all categories"""
        try:
            # Validate input data
            self.validate_data_sources(data_sources)

            scores = {}

            # Category A - PMI Performance
            self.logger.info("Calculating Category A scores...")
            scores['cat_a'] = self.cat_a_scorer.calculate_scores(
                data_sources['df_vols'],
                data_sources['mc_per_product']
            )
            self.logger.info(f"Category A calculation complete for {len(scores['cat_a'])} locations")

            # Category B - Category Segments
            self.logger.info("Calculating Category B scores...")
            scores['cat_b'] = self.cat_b_scorer.calculate_scores(
                data_sources['market_pmi'],
                data_sources['market_comp']
            )
            self.logger.info(f"Category B calculation complete for {len(scores['cat_b'])} locations")

            # Category C - Passenger Mix (2023 methodology)
            self.logger.info("Calculating Category C scores with 2023 methodology...")
            scores['cat_c'] = self.cat_c_scorer.calculate_scores(
                data_sources['pax_data'],
                data_sources['df_vols'],
                data_sources['mrk_nat_map'],
                data_sources['iata_location'],
                data_sources['country_figures'],
                data_sources['dom_volumes'],
                data_sources['dom_products'],  # Correctly pass dom_products here
                data_sources['selma_df_map']
            )
            self.logger.info(f"Category C calculation complete for {len(scores['cat_c'])} locations")

            # Category D - Location Clusters (2023 methodology)
            self.logger.info("Calculating Category D scores with 2023 methodology...")
            scores['cat_d'] = self.cat_d_scorer.calculate_scores(
                data_sources['df_vols'],
                data_sources['similarity_file'],
                data_sources['iata_location'],
                data_sources['selma_df_map']
            )
            self.logger.info(f"Category D calculation complete for {len(scores['cat_d'])} locations")

            # Combine all scores
            combined_scores = self.combine_scores(scores)
            self.logger.info(f"Combined scores calculation complete for {len(combined_scores)} locations")

            return combined_scores

        except Exception as e:
            self.logger.error(f"Error calculating scores: {str(e)}")
            raise

    def combine_scores(self, scores):
        """Combine individual category scores into final results"""
        try:
            self.logger.info("Combining category scores...")

            # Get all unique locations across all category scores
            all_locations = set()
            for cat in scores.values():
                if cat is not None and 'Location' in cat.columns:
                    all_locations.update(cat['Location'].unique())

            # Create base dataframe with all locations
            final = pd.DataFrame({'Location': list(all_locations)})

            # Merge all category scores
            for cat, df in scores.items():
                if df is not None and not df.empty:
                    cat_col = f"Cat_{cat[-1].upper()}"
                    if cat_col in df.columns:
                        final = final.merge(df[['Location', cat_col]], on='Location', how='left')

            # Fill missing values with 0
            final = final.fillna(0)

            # Calculate average score
            score_columns = [col for col in final.columns if col.startswith('Cat_')]
            final['Avg_Score'] = final[score_columns].mean(axis=1)

            # Round scores to 2 decimal places
            for col in score_columns + ['Avg_Score']:
                final[col] = final[col].round(2)

            return final

        except Exception as e:
            self.logger.error(f"Error combining scores: {str(e)}")
            raise


def generate_scores(current_year=2024, output_dir='results'):
    """
    Generate all category scores and combine them into a final score dataframe.
    Uses the 2023 methodology for Categories C and D.

    Parameters:
    -----------
    current_year : int
        The year to use for calculations
    output_dir : str
        Directory to save results

    Returns:
    --------
    DataFrame
        Combined scores for all locations
    """
    # Create logger
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    try:
        logger.info(f"Starting portfolio optimization scoring for {current_year}")

        # Load all required datasets
        logger.info("Loading datasets...")

        # Core datasets for all categories
        df_vols = pd.read_csv('df_vols_query.csv')
        mc_per_product = pd.read_csv('MC_per_Product_SQL.csv')
        market_pmi = pd.read_csv('Market_Summary_PMI.csv')
        market_comp = pd.read_csv('Market_Summary_Comp.csv')
        pax_data = pd.read_csv('Pax_Nat.csv')
        mrk_nat_map = pd.read_csv('mrk_nat_map_query.csv')
        iata_location = pd.read_csv('iata_location_query.csv')
        similarity_file = pd.read_csv('similarity_file.csv')

        # Additional datasets required for 2023 methodology
        country_figures = pd.read_csv('country_figures.csv')
        dom_volumes = pd.read_csv('sql_Domest_script.csv')
        dom_products = pd.read_csv('sql_dom_script.csv')  # Added load for dom_products
        selma_df_map = pd.read_csv('SELMA_DF_map_query.csv')

        # Organize data into a dictionary
        data_sources = {
            'df_vols': df_vols,
            'mc_per_product': mc_per_product,
            'market_pmi': market_pmi,
            'market_comp': market_comp,
            'pax_data': pax_data,
            'mrk_nat_map': mrk_nat_map,
            'iata_location': iata_location,
            'similarity_file': similarity_file,
            'country_figures': country_figures,
            'dom_volumes': dom_volumes,
            'dom_products': dom_products,  # Added dom_products to data_sources
            'selma_df_map': selma_df_map
        }

        # Initialize optimizer and calculate scores
        optimizer = PortfolioOptimizer({'current_year': current_year})
        all_scores = optimizer.calculate_scores(data_sources)

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Generate timestamp for filenames
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save combined scores
        output_path = f'{output_dir}/locations_scores_{timestamp}.csv'
        all_scores.to_csv(output_path, index=False)

        # Also create a clustering version for visualization
        try:
            # Add cluster information
            from sklearn.cluster import KMeans

            # Use scores for clustering
            score_cols = [col for col in all_scores.columns if col.startswith('Cat_')]
            cluster_data = all_scores[score_cols]

            # Determine optimal number of clusters (between 2 and 5)
            n_clusters = min(4, max(2, len(all_scores) // 50))

            # Apply KMeans clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            all_scores['Cluster'] = kmeans.fit_predict(cluster_data)

            # Save clustered data
            clustered_path = f'{output_dir}/clustered_locations_{timestamp}.csv'
            all_scores.to_csv(clustered_path, index=False)

            logger.info(f"Clustered data saved to {clustered_path}")

        except Exception as e:
            logger.warning(f"Could not perform clustering: {str(e)}")

        logger.info(f"All scores saved to {output_path}")
        return all_scores

    except Exception as e:
        logger.error(f"Error generating scores: {str(e)}")
        raise


def integrate_scoring(df_vols, mc_per_product, market_pmi, market_comp,
                      pax_data, mrk_nat_map, similarity_file, iata_location,
                      country_figures=None, dom_volumes=None, dom_products=None, selma_df_map=None):
    """
    Integrate all category scorers and combine results using the 2023 methodology.

    Parameters:
    -----------
    df_vols : DataFrame
        Duty-free volumes data
    mc_per_product : DataFrame
        Margin contribution data
    market_pmi : DataFrame
        PMI market data
    market_comp : DataFrame
        Competitor market data
    pax_data : DataFrame
        Passenger data
    mrk_nat_map : DataFrame
        Market nationality mapping
    similarity_file : DataFrame
        Cluster similarity data
    iata_location : DataFrame
        IATA to location mapping
    country_figures : DataFrame, optional
        Country smoking statistics
    dom_volumes : DataFrame, optional
        Domestic volumes
    dom_products : DataFrame, optional
        Domestic product details (from sql_dom_script.csv)
    selma_df_map : DataFrame, optional
        Duty-free SELMA attributes

    Returns:
    --------
    DataFrame
        Combined scores for all locations
    """
    try:
        logger = logging.getLogger(__name__)

        # Use current year based on columns in df_vols
        current_year = 2024
        for col in df_vols.columns:
            if 'Volume' in col and col.split()[0].isdigit():
                current_year = int(col.split()[0])
                break

        logger.info(f"Using {current_year} as the current year")

        # Handle optional datasets
        if country_figures is None:
            try:
                country_figures = pd.read_csv('country_figures.csv')
            except:
                logger.warning("country_figures.csv not found. Using empty DataFrame.")
                country_figures = pd.DataFrame()

        if dom_volumes is None:
            try:
                dom_volumes = pd.read_csv('sql_Domest_script.csv')
            except:
                logger.warning("sql_Domest_script.csv not found. Using empty DataFrame.")
                dom_volumes = pd.DataFrame()

        if dom_products is None:
            try:
                dom_products = pd.read_csv('sql_dom_script.csv')
            except:
                logger.warning("sql_dom_script.csv not found. Using empty DataFrame.")
                dom_products = pd.DataFrame()

        if selma_df_map is None:
            try:
                selma_df_map = pd.read_csv('SELMA_DF_map_query.csv')
            except:
                logger.warning("SELMA_DF_map_query.csv not found. Using empty DataFrame.")
                selma_df_map = pd.DataFrame()

        # Initialize all scorers
        cat_a_scorer = CategoryAScorer(current_year)
        cat_b_scorer = CategoryBScorer()
        cat_c_scorer = CategoryCScorer(current_year)
        cat_d_scorer = CategoryDScorer(current_year)

        # Calculate Category A scores
        logger.info("Calculating Category A scores...")
        cat_a_scores = cat_a_scorer.calculate_scores(df_vols, mc_per_product)

        # Calculate Category B scores
        logger.info("Calculating Category B scores...")
        cat_b_scores = cat_b_scorer.calculate_scores(market_pmi, market_comp)

        # Calculate Category C scores using 2023 methodology
        logger.info("Calculating Category C scores using 2023 methodology...")
        cat_c_scores = cat_c_scorer.calculate_scores(
            pax_data, df_vols, mrk_nat_map, iata_location,
            country_figures, dom_volumes, dom_products, selma_df_map
        )

        # Calculate Category D scores using 2023 methodology
        logger.info("Calculating Category D scores using 2023 methodology...")
        cat_d_scores = cat_d_scorer.calculate_scores(
            df_vols, similarity_file, iata_location, selma_df_map
        )

        # Get all unique locations
        all_locations = set()
        for scores in [cat_a_scores, cat_b_scores, cat_c_scores, cat_d_scores]:
            if scores is not None and 'Location' in scores.columns:
                all_locations.update(scores['Location'].unique())

        # Create base dataframe with all locations
        scores = pd.DataFrame({'Location': list(all_locations)})

        # Merge all category scores
        for cat_scores, cat_col in [
            (cat_a_scores, 'Cat_A'),
            (cat_b_scores, 'Cat_B'),
            (cat_c_scores, 'Cat_C'),
            (cat_d_scores, 'Cat_D')
        ]:
            if cat_scores is not None and not cat_scores.empty and cat_col in cat_scores.columns:
                scores = scores.merge(cat_scores[['Location', cat_col]], how='left', on='Location')

        # Fill missing values with 0
        scores = scores.fillna(0)

        # Calculate average score
        score_cols = ['Cat_A', 'Cat_B', 'Cat_C', 'Cat_D']
        scores['Average'] = scores[score_cols].mean(axis=1)

        # Round to 2 decimal places
        for col in score_cols + ['Average']:
            scores[col] = scores[col].round(2)

        return scores

    except Exception as e:
        logger.error(f"Error in score integration: {str(e)}")
        raise


if __name__ == "__main__":
    try:
        all_scores = generate_scores()
        print(f"Successfully calculated scores for {len(all_scores)} locations")
    except Exception as e:
        print(f"Error: {str(e)}")