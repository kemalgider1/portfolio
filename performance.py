import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


class LocationAnalyzer:
    """Analyzes location performance across 2023-2024."""

    def __init__(self, locations: List[str]):
        """Initialize with target locations."""
        self.locations = locations
        self.current_year = 2024
        self.previous_year = 2023
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """Configure logging."""
        logger = logging.getLogger('LocationAnalyzer')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger


async def load_data(self) -> None:
    """Load all required datasets."""
    try:
        # Load volume data
        self.df_vols = await self._load_file('df_vols_query.csv')

        # Load margin and cost data
        self.mc_data = await self._load_file('MC_per_Product_SQL.csv')

        # Load market summaries
        self.pmi_summary = await self._load_file('Market_Summary_PMI.csv')
        self.comp_summary = await self._load_file('Market_Summary_Comp.csv')

        # Load passenger data
        self.pax_data = await self._load_file('Pax_Nat.csv')

        # Load IATA mappings
        self.iata_map = await self._load_file('iata_location_query.csv')

        # Validate data
        self._validate_data()

    except Exception as e:
        self.logger.error(f"Error loading data: {str(e)}")
        raise


async def _load_file(self, filename: str) -> pd.DataFrame:
    """Load a CSV file with appropriate error handling."""
    try:
        df = pd.read_csv(filename)
        self.logger.info(f"Successfully loaded {filename}")
        return df
    except Exception as e:
        self.logger.error(f"Error loading {filename}: {str(e)}")
        raise


def _validate_data(self) -> None:
    """Validate loaded data for completeness and consistency."""
    required_locations = set(self.locations)

    # Check location presence in each dataset
    datasets = {
        'df_vols': self.df_vols,
        'mc_data': self.mc_data,
        'pmi_summary': self.pmi_summary,
        'comp_summary': self.comp_summary
    }

    for name, df in datasets.items():
        present_locations = set(df['Location'].unique())
        missing = required_locations - present_locations
        if missing:
            self.logger.warning(f"Missing locations in {name}: {missing}")


def analyze_segment_performance(self, location: str) -> Dict:
    """Analyze segment performance for a specific location."""
    try:
        # Filter data for location
        loc_pmi = self.pmi_summary[self.pmi_summary['Location'] == location]
        loc_comp = self.comp_summary[self.comp_summary['Location'] == location]

        # Calculate segment metrics
        segment_metrics = {
            'segment_count': len(loc_pmi['Taste'].unique()),
            'pmi_sku_count': loc_pmi['PMI_Seg_SKU'].sum(),
            'comp_sku_count': loc_comp['Comp_Seg_SKU'].sum(),
            'segment_coverage': self._calculate_segment_coverage(loc_pmi, loc_comp)
        }

        return segment_metrics

    except Exception as e:
        self.logger.error(f"Error analyzing segments for {location}: {str(e)}")
        raise


def _calculate_segment_coverage(self, pmi_df: pd.DataFrame, comp_df: pd.DataFrame) -> float:
    """Calculate segment coverage ratio."""
    pmi_segments = set(pmi_df['Taste'].unique())
    comp_segments = set(comp_df['Taste'].unique())
    total_segments = len(pmi_segments | comp_segments)

    if total_segments == 0:
        return 0.0

    pmi_coverage = len(pmi_segments) / total_segments
    return round(pmi_coverage, 4)


def analyze_sku_performance(self, location: str) -> Dict:
    """Analyze SKU performance metrics."""
    try:
        # Filter data for location
        loc_vols = self.df_vols[self.df_vols['Location'] == location]
        loc_mc = self.mc_data[self.mc_data['Location'] == location]

        # Calculate performance metrics
        current_year_vols = loc_vols[f'{self.current_year} Volume'].sum()
        prev_year_vols = loc_vols[f'{self.previous_year} Volume'].sum()

        performance_metrics = {
            'current_year_volume': current_year_vols,
            'previous_year_volume': prev_year_vols,
            'volume_growth': self._calculate_growth(current_year_vols, prev_year_vols),
            'sku_metrics': self._calculate_sku_metrics(loc_vols, loc_mc)
        }

        return performance_metrics

    except Exception as e:
        self.logger.error(f"Error analyzing SKU performance for {location}: {str(e)}")
        raise


def _calculate_growth(self, current: float, previous: float) -> float:
    """Calculate year-over-year growth rate."""
    if previous == 0:
        return 0.0
    return round((current - previous) / previous, 4)


def _calculate_sku_metrics(self, vols_df: pd.DataFrame, mc_df: pd.DataFrame) -> Dict:
    """Calculate detailed SKU performance metrics."""
    # Calculate volume contribution
    vols_df['volume_share'] = vols_df[f'{self.current_year} Volume'] / vols_df[f'{self.current_year} Volume'].sum()

    # Merge with margin data
    sku_data = vols_df.merge(mc_df[['SKU', f'{self.current_year} MC']], on='SKU', how='left')

    # Calculate performance tiers
    sku_data['tier'] = pd.qcut(sku_data['volume_share'], q=4,
                               labels=['Bottom 25%', 'Lower Mid', 'Upper Mid', 'Top 25%'])

    metrics = {
        'total_skus': len(sku_data),
        'active_skus': len(sku_data[sku_data[f'{self.current_year} Volume'] > 0]),
        'tier_distribution': sku_data['tier'].value_counts().to_dict()
    }

    return metrics


def analyze_pax_alignment(self, location: str) -> Dict:
    """Analyze passenger mix alignment."""
    try:
        # Get IATA code for location
        iata = self.iata_map[self.iata_map['LOCATION'] == location]['IATA'].iloc[0]

        # Get passenger mix
        loc_pax = self.pax_data[self.pax_data['IATA'] == iata]

        # Calculate pax metrics
        pax_metrics = {
            'total_pax': loc_pax['Pax'].sum(),
            'nationality_count': len(loc_pax['Nationality'].unique()),
            'top_nationalities': self._get_top_nationalities(loc_pax),
            'pax_concentration': self._calculate_pax_concentration(loc_pax)
        }

        return pax_metrics

    except Exception as e:
        self.logger.error(f"Error analyzing PAX alignment for {location}: {str(e)}")
        raise


def _get_top_nationalities(self, pax_df: pd.DataFrame, top_n: int = 5) -> Dict:
    """Get top nationalities by passenger volume."""
    top_nat = pax_df.groupby('Nationality')['Pax'].sum().nlargest(top_n)
    return top_nat.to_dict()


def _calculate_pax_concentration(self, pax_df: pd.DataFrame) -> float:
    """Calculate passenger concentration using Herfindahl Index."""
    total_pax = pax_df['Pax'].sum()
    if total_pax == 0:
        return 0.0

    shares = pax_df.groupby('Nationality')['Pax'].sum() / total_pax
    return round((shares ** 2).sum(), 4)


def generate_location_report(self, location: str) -> Dict:
    """Generate comprehensive location performance report."""
    try:
        report = {
            'location': location,
            'segment_analysis': self.analyze_segment_performance(location),
            'sku_performance': self.analyze_sku_performance(location),
            'pax_alignment': self.analyze_pax_alignment(location)
        }

        return report

    except Exception as e:
        self.logger.error(f"Error generating report for {location}: {str(e)}")
        raise


def compare_locations(self) -> Dict:
    """Compare performance across specified locations."""
    try:
        comparison = {}
        for location in self.locations:
            comparison[location] = self.generate_location_report(location)

        # Calculate relative metrics
        self._add_relative_metrics(comparison)

        return comparison

    except Exception as e:
        self.logger.error(f"Error comparing locations: {str(e)}")
        raise


def _add_relative_metrics(self, comparison: Dict) -> None:
    """Add relative performance metrics to comparison."""
    metrics = ['segment_analysis', 'sku_performance', 'pax_alignment']

    for metric in metrics:
        values = [data[metric] for data in comparison.values()]
        best_value = max(values, key=lambda x: x.get('score', 0))

        for location in comparison:
            comparison[location][metric]['relative_to_best'] = (
                    comparison[location][metric].get('score', 0) / best_value.get('score', 1)
            )


def visualize_comparison(self, comparison: Dict) -> None:
    """Create visualizations comparing location performance."""
    try:
        # Create subplots for different metrics
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Location Performance Comparison')

        # Plot segment coverage
        self._plot_segment_coverage(comparison, axes[0, 0])

        # Plot SKU performance
        self._plot_sku_performance(comparison, axes[0, 1])

        # Plot PAX alignment
        self._plot_pax_alignment(comparison, axes[1, 0])

        # Plot overall comparison
        self._plot_overall_comparison(comparison, axes[1, 1])

        plt.tight_layout()
        plt.show()

    except Exception as e:
        self.logger.error(f"Error creating visualizations: {str(e)}")
        raise


def _plot_segment_coverage(self, comparison: Dict, ax: plt.Axes) -> None:
    """Plot segment coverage comparison."""
    locations = list(comparison.keys())
    coverage = [data['segment_analysis']['segment_coverage'] for data in comparison.values()]

    ax.bar(locations, coverage)
    ax.set_title('Segment Coverage')
    ax.set_ylabel('Coverage Ratio')
    ax.tick_params(axis='x', rotation=45)


def _plot_sku_performance(self, comparison: Dict, ax: plt.Axes) -> None:
    """Plot SKU performance comparison."""
    locations = list(comparison.keys())
    metrics = ['active_skus', 'total_skus']

    x = np.arange(len(locations))
    width = 0.35

    for i, metric in enumerate(metrics):
        values = [data['sku_performance']['sku_metrics'][metric] for data in comparison.values()]
        ax.bar(x + i * width, values, width, label=metric)

    ax.set_title('SKU Metrics')
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(locations, rotation=45)
    ax.legend()


def _plot_pax_alignment(self, comparison: Dict, ax: plt.Axes) -> None:
    """Plot PAX alignment comparison."""
    locations = list(comparison.keys())
    concentration = [data['pax_alignment']['pax_concentration'] for data in comparison.values()]

    ax.bar(locations, concentration)
    ax.set_title('PAX Concentration')
    ax.set_ylabel('Herfindahl Index')
    ax.tick_params(axis='x', rotation=45)


def _plot_overall_comparison(self, comparison: Dict, ax: plt.Axes) -> None:
    """Plot overall performance comparison."""
    metrics = ['segment_coverage', 'sku_efficiency', 'pax_alignment']
    locations = list(comparison.keys())

    data = []
    for location in locations:
        loc_data = [
            comparison[location]['segment_analysis']['segment_coverage'],
            comparison[location]['sku_performance']['sku_metrics']['active_skus'] /
            comparison[location]['sku_performance']['sku_metrics']['total_skus'],
            1 - comparison[location]['pax_alignment']['pax_concentration']  # Lower concentration is better
        ]
        data.append(loc_data)

    data = np.array(data)

    # Create radar chart
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)

    # Close the plot
    angles = np.concatenate((angles, [angles[0]]))

    ax.plot(angles, np.concatenate((data[0], [data[0][0]])), 'o-', label=locations[0])
    ax.plot(angles, np.concatenate((data[1], [data[1][0]])), 'o-', label=locations[1])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.legend(loc='upper right')
    ax.set_title('Overall Performance Comparison')


if __name__ == "__main__":
    import asyncio


    async def main():
        # Initialize analyzer with locations
        analyzer = LocationAnalyzer(['Zurich', 'Kingston'])

        # Load data
        await analyzer.load_data()

        # Generate comparison
        comparison = await analyzer.compare_locations()

        # Visualize results
        analyzer.visualize_comparison(comparison)

        # Print detailed insights
        print("\nDetailed Location Performance Analysis")
        print("=" * 50)

        for location, data in comparison.items():
            print(f"\n{location} Analysis:")
            print("-" * 30)

            # Segment Analysis
            segment_data = data['segment_analysis']
            print("\nSegment Performance:")
            print(f"- Total Segments: {segment_data['segment_count']}")
            print(f"- PMI SKU Count: {segment_data['pmi_sku_count']}")
            print(f"- Competitor SKU Count: {segment_data['comp_sku_count']}")
            print(f"- Segment Coverage: {segment_data['segment_coverage']:.2%}")

            # SKU Performance
            sku_data = data['sku_performance']
            print("\nSKU Performance:")
            print(f"- Current Year Volume: {sku_data['current_year_volume']:,.0f}")
            print(f"- Volume Growth: {sku_data['volume_growth']:.2%}")
            print(f"- Active SKUs: {sku_data['sku_metrics']['active_skus']}")
            print("Top Performers:")
            for sku in sku_data['sku_metrics']['top_performers'][:3]:
                print(f"  * {sku['SKU']}: {sku['2024 Volume']:,.0f} units")

            # PAX Alignment
            pax_data = data['pax_alignment']
            print("\nPassenger Alignment:")
            print(f"- Total Passengers: {pax_data['total_pax']:,.0f}")
            print(f"- Unique Nationalities: {pax_data['nationality_count']}")
            print(f"- PAX Concentration: {pax_data['pax_concentration']:.2f}")
            print("Top Nationalities:")
            for nat in pax_data['top_nationalities'][:3]:
                print(f"  * {nat['Nationality']}: {nat['pax_share']:.2%}")

            # Key Insights
            print("\nKey Insights:")

            # Volume insights
            vol_change = sku_data['volume_growth']
            if vol_change > 0:
                print(f"✓ Volume growth of {vol_change:.1%} indicates strong market performance")
            else:
                print(f"! Volume decline of {abs(vol_change):.1%} requires attention")

            # Segment insights
            seg_coverage = segment_data['segment_coverage']
            if seg_coverage > 0.8:
                print("✓ Strong segment coverage provides competitive advantage")
            elif seg_coverage < 0.5:
                print("! Limited segment coverage suggests opportunity for expansion")

            # PAX alignment insights
            pax_conc = pax_data['pax_concentration']
            if pax_conc < 0.3:
                print("✓ Diverse passenger mix provides stable demand")
            else:
                print("! High passenger concentration indicates potential risk")

            print("\nRecommendations:")
            # Generate specific recommendations based on metrics
            if vol_change < 0:
                print("- Focus on volume recovery through targeted promotions")
            if seg_coverage < 0.7:
                print("- Expand segment coverage to match competitor offerings")
            if pax_conc > 0.4:
                print("- Diversify portfolio to better serve varied passenger groups")

            print("-" * 50)


    # Run the async main function
    asyncio.run(main())