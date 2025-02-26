import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Load the loc_pick_data DataFrame from the Excel file
loc_pick_data = pd.read_excel('loc_pick.xlsx')

# Print the column names to identify the correct column name
print(loc_pick_data.columns)

def create_enhanced_comparison(location1, location2, location_data, loc_pick_data, cat_results):
    """
    Create an enhanced visualization comparing two locations with business impact metrics

    Parameters:
    -----------
    location1, location2 : str
        Names of locations to compare
    location_data : DataFrame
        Clustered scoring data
    loc_pick_data : DataFrame
        Business performance data
    cat_results : DataFrame
        Additional category details
    """
    # Extract data for the two locations
    loc1_data = location_data[location_data['Location'] == location1].iloc[0]
    loc2_data = location_data[location_data['Location'] == location2].iloc[0]

    # Get business metrics
    loc1_business = loc_pick_data[loc_pick_data['Location'] == location1].iloc[0]
    loc2_business = loc_pick_data[loc_pick_data['Location'] == location2].iloc[0]

    # Create figure
    fig = plt.figure(figsize=(18, 12), facecolor='#f8f8f8')
    gs = GridSpec(3, 4, figure=fig)

    # 1. Score comparison bar chart
    ax_scores = fig.add_subplot(gs[0, :2])
    categories = ['Cat_A', 'Cat_B', 'Cat_C', 'Cat_D', 'Avg_Score']
    loc1_scores = [loc1_data[cat] for cat in categories]
    loc2_scores = [loc2_data[cat] for cat in categories]

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax_scores.bar(x - width / 2, loc1_scores, width, label=location1, color='#1f77b4')
    bars2 = ax_scores.bar(x + width / 2, loc2_scores, width, label=location2, color='#ff7f0e')

    ax_scores.set_title('Category Score Comparison', fontsize=14, fontweight='bold')
    ax_scores.set_xticks(x)
    ax_scores.set_xticklabels(
        ['PMI\nPerformance', 'Category\nSegments', 'Passenger\nMix', 'Location\nClusters', 'Average\nScore'])
    ax_scores.set_ylim(0, 10)
    ax_scores.legend(loc='upper right')

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax_scores.annotate(f'{height:.1f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom')

    for bar in bars2:
        height = bar.get_height()
        ax_scores.annotate(f'{height:.1f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           ha='center', va='bottom')

    # 2. Business metrics comparison
    ax_business = fig.add_subplot(gs[0, 2:])
    metrics = ['2024Revenue', 'AverageProfit', 'Total_SKU_Count']

    # Normalize for comparison (convert to percentage of maximum)
    loc1_metrics = [
        loc1_business['2024Revenue'] / 1e6,  # Convert to millions
        loc1_business['AverageProfit'] * 100,  # Convert to percentage
        loc1_business['Total_SKU_Count']  # Use the correct column name
    ]

    loc2_metrics = [
        loc2_business['2024Revenue'] / 1e6,  # Convert to millions
        loc2_business['AverageProfit'] * 100,  # Convert to percentage
        loc2_business['Total_SKU_Count']  # Use the correct column name
    ]

    x = np.arange(len(metrics))

    bars1 = ax_business.bar(x - width / 2, loc1_metrics, width, label=location1, color='#1f77b4')
    bars2 = ax_business.bar(x + width / 2, loc2_metrics, width, label=location2, color='#ff7f0e')

    ax_business.set_title('Business Performance Metrics', fontsize=14, fontweight='bold')
    ax_business.set_xticks(x)
    ax_business.set_xticklabels(['Revenue\n($ Millions)', 'Profit\nMargin (%)', 'SKU\nCount'])
    ax_business.legend(loc='upper right')

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax_business.annotate(f'{height:.1f}',
                             xy=(bar.get_x() + bar.get_width() / 2, height),
                             xytext=(0, 3),
                             textcoords="offset points",
                             ha='center', va='bottom')

    for bar in bars2:
        height = bar.get_height()
        ax_business.annotate(f'{height:.1f}',
                             xy=(bar.get_x() + bar.get_width() / 2, height),
                             xytext=(0, 3),
                             textcoords="offset points",
                             ha='center', va='bottom')

    # 3. Score breakdown for Location 1
    ax_loc1 = fig.add_subplot(gs[1, :2])
    # Get component details - we'll define some example values

    # Create a lookup dictionary for component scores from cat_results
    # This would need adjustment based on your actual data structure
    component_lookup = {
        location1: {
            # Cat A components
            'PMI Performance': 0.72, 'Volume Growth': 0.15,
            'High Margin SKUs': 0.65, 'Premium Mix': 0.63,
            # Cat B components
            'Segment Coverage': 0.76, 'Competitive Position': 0.55,
            'Premium Ratio': 0.67, 'Innovation Score': 0.43,
            # Cat C components
            'PAX Alignment': 0.64, 'Nationality Mix': 0.74,
            'Traveler Type': 0.53, 'Seasonal Adjustment': 0.42,
            # Cat D components
            'Cluster Similarity': 0.58, 'Regional Alignment': 0.62,
            'Size Compatibility': 0.45, 'Format Distribution': 0.39
        },
        location2: {
            # Cat A components
            'PMI Performance': 0.68, 'Volume Growth': -0.03,
            'High Margin SKUs': 0.40, 'Premium Mix': 0.55,
            # Cat B components
            'Segment Coverage': 0.28, 'Competitive Position': 0.33,
            'Premium Ratio': 0.41, 'Innovation Score': 0.22,
            # Cat C components
            'PAX Alignment': 0.31, 'Nationality Mix': 0.38,
            'Traveler Type': 0.27, 'Seasonal Adjustment': 0.19,
            # Cat D components
            'Cluster Similarity': 0.05, 'Regional Alignment': 0.12,
            'Size Compatibility': 0.00, 'Format Distribution': 0.00
        }
    }

    # Get component scores for location 1
    component_categories = [
        'PMI Performance', 'Volume Growth', 'High Margin SKUs', 'Premium Mix',
        'Segment Coverage', 'Competitive Position', 'Premium Ratio', 'Innovation Score',
        'PAX Alignment', 'Nationality Mix', 'Traveler Type', 'Seasonal Adjustment',
        'Cluster Similarity', 'Regional Alignment', 'Size Compatibility', 'Format Distribution'
    ]

    loc1_components = [component_lookup[location1][comp] for comp in component_categories]

    # Create heatmap-style display
    component_matrix = np.array(loc1_components).reshape(4, 4)

    sns.heatmap(component_matrix, annot=True, cmap='YlGnBu', vmin=0, vmax=1,
                linewidths=.5, ax=ax_loc1, fmt='.2f', cbar_kws={'label': 'Score (0-1)'})

    # Create custom column labels
    col_labels = ['Component 1', 'Component 2', 'Component 3', 'Component 4']
    row_labels = ['Cat A', 'Cat B', 'Cat C', 'Cat D']

    ax_loc1.set_xticklabels(col_labels)
    ax_loc1.set_yticklabels(row_labels)
    ax_loc1.set_title(f'{location1} Component Breakdown', fontsize=14, fontweight='bold')

    # 4. Score breakdown for Location 2
    ax_loc2 = fig.add_subplot(gs[1, 2:])

    # Get component scores for location 2
    loc2_components = [component_lookup[location2][comp] for comp in component_categories]

    # Create heatmap-style display
    component_matrix2 = np.array(loc2_components).reshape(4, 4)

    sns.heatmap(component_matrix2, annot=True, cmap='YlGnBu', vmin=0, vmax=1,
                linewidths=.5, ax=ax_loc2, fmt='.2f', cbar_kws={'label': 'Score (0-1)'})

    ax_loc2.set_xticklabels(col_labels)
    ax_loc2.set_yticklabels(row_labels)
    ax_loc2.set_title(f'{location2} Component Breakdown', fontsize=14, fontweight='bold')

    # 5. Time series performance
    ax_trend = fig.add_subplot(gs[2, :])
    years = [2022, 2023, 2024]

    # Revenue trends
    loc1_revenue = [loc1_business['2022Revenue'], loc1_business['2023Revenue'], loc1_business['2024Revenue']]
    loc2_revenue = [loc2_business['2022Revenue'], loc2_business['2023Revenue'], loc2_business['2024Revenue']]

    # Create twin axis for volume
    ax_revenue = ax_trend
    ax_volume = ax_trend.twinx()

    # Volume trends
    loc1_volume = [loc1_business['2022Volume'], loc1_business['2023Volume'], loc1_business['2024Volume']]
    loc2_volume = [loc2_business['2022Volume'], loc2_business['2023Volume'], loc2_business['2024Volume']]

    # Plot revenue as bars
    ax_revenue.bar([x - 0.2 for x in years], loc1_revenue, width=0.35, color='#1f77b4', alpha=0.7,
                   label=f'{location1} Revenue')
    ax_revenue.bar([x + 0.2 for x in years], loc2_revenue, width=0.35, color='#ff7f0e', alpha=0.7,
                   label=f'{location2} Revenue')

    # Plot volume as lines
    ax_volume.plot(years, loc1_volume, 'o-', color='blue', linewidth=2, label=f'{location1} Volume')
    ax_volume.plot(years, loc2_volume, 's-', color='red', linewidth=2, label=f'{location2} Volume')

    ax_trend.set_title('3-Year Performance Trends', fontsize=14, fontweight='bold')
    ax_trend.set_xlabel('Year', fontsize=12)
    ax_revenue.set_ylabel('Revenue ($)', fontsize=12)
    ax_volume.set_ylabel('Volume', fontsize=12)

    # Create combined legend
    lines1, labels1 = ax_revenue.get_legend_handles_labels()
    lines2, labels2 = ax_volume.get_legend_handles_labels()
    ax_trend.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4)

    # Main title
    plt.suptitle(f'Portfolio Optimization Analysis: {location1} vs {location2}',
                 fontsize=20, y=0.98, fontweight='bold')

    # Classification banner
    loc1_class = "HIGH PERFORMER" if loc1_data['Avg_Score'] >= 3.0 else "REQUIRES OPTIMIZATION"
    loc2_class = "HIGH PERFORMER" if loc2_data['Avg_Score'] >= 3.0 else "REQUIRES OPTIMIZATION"
    classification_text = f"{location1}: {loc1_class}     |     {location2}: {loc2_class}"

    plt.figtext(0.5, 0.94, classification_text,
                fontsize=14, ha='center', color='white',
                bbox=dict(facecolor='#333333', alpha=0.9, boxstyle='round,pad=0.5'))

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(f'comparison_{location1}_vs_{location2}.png', dpi=300, bbox_inches='tight')

    print(f"Visualization comparing {location1} and {location2} saved as 'comparison_{location1}_vs_{location2}.png'")


def main():
    """
    Main function to load data and run the comparison visualization
    """
    print("Loading data files...")

    # Load clustered locations data
    location_data = pd.read_csv('results/clustered_locations_20250225_030538.csv')

    # Load location performance data from loc_pick.xlsx
    loc_pick_data = pd.read_excel('loc_pick.xlsx')

    # Load cat_results.txt - this would need preprocessing in a real scenario
    # For this example, we'll use placeholder data embedded in the function
    # In reality, you'd parse cat_results.txt here
    cat_results = None  # Placeholder

    # Define the locations to compare
    location1 = "Singapore - Changi"
    location2 = "Hanoi"

    print(f"Creating comparison visualization for {location1} vs {location2}...")
    create_enhanced_comparison(location1, location2, location_data, loc_pick_data, cat_results)

    print("Visualization complete!")


if __name__ == "__main__":
    main()