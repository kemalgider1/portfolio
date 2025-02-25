import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os


def load_yearly_data():
    """Load and combine 2023 and 2024 analysis results"""
    # Load location scores
    loc_2023 = pd.read_csv('results/2023/location_scores.csv')
    loc_2024 = pd.read_csv('results/2024/location_scores.csv')

    # Load cluster results
    clusters_2023 = pd.read_csv('results/2023/cluster_insights.csv')
    clusters_2024 = pd.read_csv('results/2024/cluster_insights.csv')

    # Load SKU performance
    sku_2023 = pd.read_csv('results/2023/sku_performance.csv')
    sku_2024 = pd.read_csv('results/2024/sku_performance.csv')

    return {
        'locations': (loc_2023, loc_2024),
        'clusters': (clusters_2023, clusters_2024),
        'skus': (sku_2023, sku_2024)
    }


def analyze_year_over_year_changes(data):
    """Analyze changes between 2023 and 2024"""
    loc_2023, loc_2024 = data['locations']

    # Compare location scores
    location_comparison = pd.merge(
        loc_2023[['Location', 'Total_Score']].rename(columns={'Total_Score': 'Score_2023'}),
        loc_2024[['Location', 'Total_Score']].rename(columns={'Total_Score': 'Score_2024'}),
        on='Location', how='outer'
    )

    location_comparison['Score_Change'] = location_comparison['Score_2024'] - location_comparison['Score_2023']

    # Compare SKU performance
    sku_2023, sku_2024 = data['skus']
    sku_comparison = pd.merge(
        sku_2023[['SKU', 'Volume']].rename(columns={'Volume': 'Volume_2023'}),
        sku_2024[['SKU', 'Volume']].rename(columns={'Volume': 'Volume_2024'}),
        on='SKU', how='outer'
    )

    sku_comparison['Volume_Change'] = (
            (sku_comparison['Volume_2024'] - sku_comparison['Volume_2023']) /
            sku_comparison['Volume_2023']
    )

    return {
        'location_changes': location_comparison,
        'sku_changes': sku_comparison
    }


def visualize_changes(changes):
    """Create visualizations for year-over-year changes"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs('results/yoy_analysis', exist_ok=True)

    # Location score changes
    plt.figure(figsize=(12, 6))
    sns.histplot(data=changes['location_changes'], x='Score_Change', bins=30)
    plt.title('Distribution of Location Score Changes (2023-2024)')
    plt.xlabel('Score Change')
    loc_plot_path = f'results/yoy_analysis/location_changes_{timestamp}.png'
    plt.savefig(loc_plot_path)
    plt.close()

    # SKU volume changes
    plt.figure(figsize=(12, 6))
    sns.histplot(data=changes['sku_changes'], x='Volume_Change', bins=30)
    plt.title('Distribution of SKU Volume Changes (2023-2024)')
    plt.xlabel('Volume Change %')
    sku_plot_path = f'results/yoy_analysis/sku_changes_{timestamp}.png'
    plt.savefig(sku_plot_path)
    plt.close()

    return {
        'location_plot': loc_plot_path,
        'sku_plot': sku_plot_path
    }


def generate_comparison_report(changes):
    """Generate summary report of year-over-year changes"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = f'results/yoy_analysis/comparison_report_{timestamp}.txt'

    with open(report_path, 'w') as f:
        f.write("Year-over-Year Comparison Report (2023-2024)\n")
        f.write("==========================================\n\n")

        # Location changes
        f.write("1. Location Performance Changes\n")
        f.write("--------------------------\n")
        f.write(f"Total locations analyzed: {len(changes['location_changes'])}\n")
        f.write(f"Locations with improved scores: {(changes['location_changes']['Score_Change'] > 0).sum()}\n")
        f.write(f"Locations with declined scores: {(changes['location_changes']['Score_Change'] < 0).sum()}\n\n")

        # SKU changes
        f.write("2. SKU Performance Changes\n")
        f.write("------------------------\n")
        f.write(f"Total SKUs analyzed: {len(changes['sku_changes'])}\n")
        f.write(f"SKUs with volume growth: {(changes['sku_changes']['Volume_Change'] > 0).sum()}\n")
        f.write(f"SKUs with volume decline: {(changes['sku_changes']['Volume_Change'] < 0).sum()}\n")

    return report_path


def main():
    """Run year-over-year comparison analysis"""
    print("Loading yearly data...")
    data = load_yearly_data()

    print("Analyzing year-over-year changes...")
    changes = analyze_year_over_year_changes(data)

    print("Generating visualizations...")
    plot_paths = visualize_changes(changes)

    print("Generating comparison report...")
    report_path = generate_comparison_report(changes)

    print("\nAnalysis complete. Results saved to:")
    print(f"Report: {report_path}")
    print("Visualizations:", *plot_paths.values(), sep="\n- ")


if __name__ == "__main__":
    main()