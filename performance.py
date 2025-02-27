import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon, Circle
import scipy.stats as stats

# Set the style for all plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")


# Function to create location comparison data
def create_comparison_data():
    """
    Create the comparison data for Paris Charles De Gaulle and Shanghai Hongqiao
    """
    # Paris Charles De Gaulle Data
    paris_data = {
        'location': 'Paris Charles De Gaulle',
        'classification': 'HIGH PERFORMER',
        'category_scores': {
            'Cat_A': 6.67,  # PMI Performance
            'Cat_B': 10.0,  # Category Segments
            'Cat_C': 9.98,  # Passenger Mix
            'Cat_D': 10.0,  # Location Clusters
        },
        'avg_score': 9.16,
        'market_context': {
            'Total_SKUs': 78,
            'PMI_SKUs': 25,
            'Comp_SKUs': 53,
            'Market_Share': 0.48,
            'PAX_Annual': 58122178,
            'Total_Volume': 79376600,
            'Green_Count': 4,
            'Red_Count': 2,
        },
        'category_a_components': {
            'PMI_Performance': 0.67,
            'Volume_Growth': 0.32,
            'High_Margin_SKUs': 4,
            'Premium_Mix': 0.70
        },
        'category_b_components': {
            'Segment_Coverage': 1.00,
            'Competitive_Position': 1.00,
            'Premium_Ratio': 1.00,
            'Innovation_Score': 1.00
        },
        'category_c_components': {
            'PAX_Alignment': 1.00,
            'Nationality_Mix': 1.00,
            'Traveler_Type': 0.98,
            'Seasonal_Adjustment': 1.00
        },
        'category_d_components': {
            'Cluster_Similarity': 1.00,
            'Regional_Alignment': 1.00,
            'Size_Compatibility': 1.00,
            'Format_Distribution': 1.00
        },
        'sku_data': pd.DataFrame({
            'SKU': [
                'CHESTERFIELD_CDG5', 'HEETS_CDG3', 'CHESTERFIELD_CDG2',
                'HEETS_CDG2', 'PARLIAMENT_CDG4', 'MARLBORO_CDG4',
                'L&M_CDG3', 'MARLBORO_CDG2', 'PARLIAMENT_CDG2', 'HEETS_CDG1'
            ],
            'Brand_Family': [
                'CHESTERFIELD', 'HEETS', 'CHESTERFIELD',
                'HEETS', 'PARLIAMENT', 'MARLBORO',
                'L&M', 'MARLBORO', 'PARLIAMENT', 'HEETS'
            ],
            'Volume_2024': [
                927111, 915673, 879615,
                842517, 817881, 570036,
                512088, 741064, 810345, 725236
            ],
            'Growth': [
                0.110, 0.043, 0.307,
                0.032, 0.022, 0.315,
                0.145, 0.269, 0.183, 0.028
            ],
            'Margin': [
                0.79, 0.78, 0.87,
                0.77, 0.78, 0.77,
                0.76, 0.76, 0.84, 0.81
            ],
            'Flag': [
                'Green', 'Green', 'Green',
                'Green', 'White', 'White',
                'White', 'White', 'White', 'White'
            ],
        }),
        'brand_mix': pd.DataFrame({
            'Brand_Family': ['CHESTERFIELD', 'HEETS', 'PARLIAMENT', 'MARLBORO', 'L&M'],
            'Volume_2024': [3684944, 3174600, 3146783, 2529436, 2515904],
            'Share_of_Portfolio': [0.245, 0.211, 0.209, 0.168, 0.167]
        }),
        # Category Segments Data - PMI SKUs vs Competition
        'category_segments': {
            'flavors': ['Regular', 'Regular', 'Regular', 'Menthol', 'Menthol', 'NTD'],
            'tastes': ['Full Flavor', 'Lights', 'Ultra Lights', 'Full Flavor', 'Lights', 'Lights'],
            'thickness': ['Standard', 'Standard', 'Standard', 'Standard', 'Standard', 'Standard'],
            'lengths': ['King Size', 'King Size', 'King Size', 'King Size', 'King Size', 'King Size'],
            'pmi_skus': [8, 10, 2, 2, 1, 2],
            'pmi_pct': [0.32, 0.40, 0.08, 0.08, 0.04, 0.08],
            'comp_skus': [17, 15, 4, 7, 5, 5],
            'comp_pct': [0.32, 0.28, 0.08, 0.13, 0.09, 0.09],
        },
        # Passenger Mix Data - Real vs Ideal SoM
        'passenger_mix': {
            'segments': [
                'Regular-Full-STD-KS', 'Regular-Lights-STD-KS',
                'Regular-Ultra-STD-KS', 'Menthol-Full-STD-KS',
                'Menthol-Lights-STD-KS', 'NTD-Lights-STD-KS'
            ],
            'pmi_som': [0.45, 0.33, 0.09, 0.05, 0.05, 0.03],
            'ideal_som': [0.43, 0.34, 0.08, 0.06, 0.04, 0.04]
        },
        # Location Cluster Data
        'location_cluster': {
            'segments': [
                'Regular-Full-STD-KS', 'Regular-Lights-STD-KS',
                'Regular-Ultra-STD-KS', 'Menthol-Full-STD-KS',
                'Menthol-Lights-STD-KS', 'NTD-Lights-STD-KS'
            ],
            'pmi_skus': [8, 10, 2, 2, 1, 2],
            'pmi_pct': [0.32, 0.40, 0.08, 0.08, 0.04, 0.08],
            'cluster_skus': [24, 30, 6, 4, 3, 3],
            'cluster_pct': [0.34, 0.43, 0.09, 0.06, 0.04, 0.04]
        }
    }

    # Shanghai Hongqiao Data
    shanghai_data = {
        'location': 'Shanghai Hongqiao',
        'classification': 'REQUIRES OPTIMIZATION',
        'category_scores': {
            'Cat_A': 6.10,  # PMI Performance
            'Cat_B': 6.63,  # Category Segments
            'Cat_C': 0.00,  # Passenger Mix
            'Cat_D': 0.72,  # Location Clusters
        },
        'avg_score': 3.36,
        'market_context': {
            'Total_SKUs': 57,
            'PMI_SKUs': 17,
            'Comp_SKUs': 40,
            'Market_Share': 0.61,
            'PAX_Annual': 32364558,
            'Total_Volume': 44479675,
            'Green_Count': 0,
            'Red_Count': 8,
        },
        'category_a_components': {
            'PMI_Performance': 0.61,
            'Volume_Growth': -0.05,
            'High_Margin_SKUs': 0,
            'Premium_Mix': 0.59
        },
        'category_b_components': {
            'Segment_Coverage': 0.66,
            'Competitive_Position': 0.66,
            'Premium_Ratio': 0.66,
            'Innovation_Score': 0.65
        },
        'category_c_components': {
            'PAX_Alignment': 0.00,
            'Nationality_Mix': 0.00,
            'Traveler_Type': 0.00,
            'Seasonal_Adjustment': 0.00
        },
        'category_d_components': {
            'Cluster_Similarity': 0.07,
            'Regional_Alignment': 0.07,
            'Size_Compatibility': 0.08,
            'Format_Distribution': 0.50
        },
        'sku_data': pd.DataFrame({
            'SKU': [
                'CHESTERFIELD_SHA1', 'CHESTERFIELD_SHA2', 'MARLBORO_SHA1',
                'BOND_SHA2', 'L&M_SHA1', 'PARLIAMENT_SHA1',
                'MARLBORO_SHA2', 'LARK_SHA4', 'MARLBORO_SHA3', 'LARK_SHA1'
            ],
            'Brand_Family': [
                'CHESTERFIELD', 'CHESTERFIELD', 'MARLBORO',
                'BOND', 'L&M', 'PARLIAMENT',
                'MARLBORO', 'LARK', 'MARLBORO', 'LARK'
            ],
            'Volume_2024': [
                541189, 515628, 497888,
                486040, 464566, 363128,
                438800, 380529, 353256, 421987
            ],
            'Growth': [
                0.011, -0.100, -0.149,
                -0.054, -0.148, 0.074,
                0.066, 0.046, -0.120, 0.015
            ],
            'Margin': [
                0.72, 0.68, 0.78,
                0.73, 0.73, 0.65,
                0.73, 0.80, 0.74, 0.78
            ],
            'Flag': [
                'White', 'Red', 'Red',
                'Red', 'Red', 'White',
                'White', 'White', 'Red', 'White'
            ],
        }),
        'brand_mix': pd.DataFrame({
            'Brand_Family': ['LARK', 'MARLBORO', 'L&M', 'BOND', 'CHESTERFIELD', 'PARLIAMENT'],
            'Volume_2024': [1482992, 1289944, 1085178, 1079730, 1056817, 612299],
            'Share_of_Portfolio': [0.224, 0.195, 0.164, 0.163, 0.160, 0.093]
        }),
        # Category Segments Data - PMI SKUs vs Competition
        'category_segments': {
            'flavors': ['Regular', 'Regular', 'Regular', 'Menthol', 'Menthol'],
            'tastes': ['Full Flavor', 'Lights', 'Ultra Lights', 'Full Flavor', 'Lights'],
            'thickness': ['Standard', 'Standard', 'Standard', 'Standard', 'Standard'],
            'lengths': ['King Size', 'King Size', 'King Size', 'King Size', 'King Size'],
            'pmi_skus': [8, 6, 1, 1, 1],
            'pmi_pct': [0.47, 0.35, 0.06, 0.06, 0.06],
            'comp_skus': [15, 12, 3, 6, 4],
            'comp_pct': [0.38, 0.30, 0.08, 0.15, 0.10]
        },
        # Passenger Mix Data - Real vs Ideal SoM
        'passenger_mix': {
            'segments': [
                'Regular-Full-STD-KS', 'Regular-Lights-STD-KS',
                'Regular-Ultra-STD-KS', 'Menthol-Full-STD-KS',
                'Menthol-Lights-STD-KS'
            ],
            'pmi_som': [0.52, 0.40, 0.04, 0.02, 0.02],
            'ideal_som': [0.30, 0.22, 0.03, 0.25, 0.20]  # Huge mismatch with Menthol which explains 0 score
        },
        # Location Cluster Data
        'location_cluster': {
            'segments': [
                'Regular-Full-STD-KS', 'Regular-Lights-STD-KS',
                'Regular-Ultra-STD-KS', 'Menthol-Full-STD-KS',
                'Menthol-Lights-STD-KS'
            ],
            'pmi_skus': [8, 6, 1, 1, 1],
            'pmi_pct': [0.47, 0.35, 0.06, 0.06, 0.06],
            'cluster_skus': [28, 26, 5, 12, 9],
            'cluster_pct': [0.35, 0.33, 0.06, 0.15, 0.11]
        }
    }

    return paris_data, shanghai_data


def analyze_current_range(data, location_name):
    """
    Analyze the current range (Category A) for a given location
    """
    sku_data = data['sku_data']
    context = data['market_context']
    components = data['category_a_components']

    # Count flags
    green_count = len(sku_data[sku_data['Flag'] == 'Green'])
    red_count = len(sku_data[sku_data['Flag'] == 'Red'])

    # Calculate score as per the methodology
    # Score calculation: (# green - 2*# red) / total # SKUs = score between -200% and +100%
    # Scaled to get a final score between 0 and 10.
    raw_score = (green_count - 2 * red_count) / len(sku_data) * 100

    # Scale from -200 to 100 range to 0-10 range
    scaled_score = (raw_score - (-200)) * (10 / 300)

    print(f"\n--- Current Range Analysis for {location_name} ---")
    print(f"Total SKUs analyzed: {len(sku_data)}")
    print(f"Green flag SKUs: {green_count}")
    print(f"Red flag SKUs: {red_count}")
    print(f"Raw score: {raw_score:.2f}%")
    print(f"Scaled score (Category A): {scaled_score:.2f} out of 10")

    # Create a bubble chart visualization
    plt.figure(figsize=(10, 7))

    # Create a colormap
    colors = {'Green': '#2ca02c', 'Red': '#d62728', 'White': '#7f7f7f'}

    # Use volume for bubble size and margin for x-axis
    plt.scatter(
        sku_data['Margin'],
        sku_data['Growth'],
        s=sku_data['Volume_2024'] / 10000,  # Scale down for better visualization
        c=[colors[flag] for flag in sku_data['Flag']],
        alpha=0.7,
        edgecolors='w'
    )

    # Add labels for each point
    for i, row in sku_data.iterrows():
        plt.annotate(
            row['SKU'].split('_')[0][:5],  # Use first 5 chars of brand name
            (row['Margin'], row['Growth']),
            xytext=(5, 5),
            textcoords='offset points'
        )

    # Add reference lines
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    plt.title(f'SKU Performance Analysis - {location_name}')
    plt.xlabel('Margin')
    plt.ylabel('Growth')

    # Add a legend
    for flag, color in colors.items():
        if flag == 'Green' and green_count == 0:
            continue
        if flag == 'Red' and red_count == 0:
            continue
        plt.scatter([], [], c=color, s=100, label=flag)

    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{location_name.replace(' ', '_')}_current_range.png", dpi=300)

    # Create detailed table of SKUs
    print("\nDetailed SKU breakdown:")
    print(f"{'Flag':<6} {'Brand':<15} {'SKU':<20} {'Volume':<10} {'Growth':<8} {'Margin':<8}")
    print("-" * 70)

    for i, row in sku_data.sort_values('Volume_2024', ascending=False).head(10).iterrows():
        flag_symbol = "🟢" if row['Flag'] == 'Green' else ("🔴" if row['Flag'] == 'Red' else "⚪️")
        print(
            f"{flag_symbol:<6} {row['Brand_Family']:<15} {row['SKU']:<20} {row['Volume_2024']:<10,.0f} {row['Growth']:<8.1%} {row['Margin']:<8.1%}")

    return scaled_score


def analyze_category_segments(data, location_name):
    """
    Analyze the category segments (Category B) for a given location
    """
    segments = data['category_segments']

    # Create DataFrames for PMI and competitor SKUs
    pmi_df = pd.DataFrame({
        'Flavor': segments['flavors'],
        'Taste': segments['tastes'],
        'Thickness': segments['thickness'],
        'Length': segments['lengths'],
        'SKUs': segments['pmi_skus'],
        'Percentage': segments['pmi_pct']
    })

    comp_df = pd.DataFrame({
        'Flavor': segments['flavors'],
        'Taste': segments['tastes'],
        'Thickness': segments['thickness'],
        'Length': segments['lengths'],
        'SKUs': segments['comp_skus'],
        'Percentage': segments['comp_pct']
    })

    # Calculate the correlation (R^2)
    r_squared = np.corrcoef(segments['pmi_pct'], segments['comp_pct'])[0, 1] ** 2

    print(f"\n--- Category Segments Analysis for {location_name} ---")
    print(f"R² correlation between PMI and competition: {r_squared:.2f}")
    print(f"Category B score: {r_squared * 10:.2f} out of 10")

    # Create visualization
    plt.figure(figsize=(10, 7))

    # Create segment labels for x-axis
    segment_labels = []
    for i in range(len(segments['flavors'])):
        segment_labels.append(
            f"{segments['flavors'][i]}-{segments['tastes'][i][:5]}-{segments['thickness'][i][:3]}-{segments['lengths'][i][:2]}")

    # Plot PMI and competitor percentages
    x = np.arange(len(segment_labels))
    width = 0.35

    plt.bar(x - width / 2, segments['pmi_pct'], width, label='PMI SKUs %', color='#1f77b4')
    plt.bar(x + width / 2, segments['comp_pct'], width, label='Competitor SKUs %', color='#ff7f0e')

    plt.xlabel('Segment')
    plt.ylabel('Percentage of SKUs')
    plt.title(f'PMI vs Competitor Segment Coverage - {location_name}\nR² = {r_squared:.2f}')
    plt.xticks(x, segment_labels, rotation=45, ha='right')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{location_name.replace(' ', '_')}_category_segments.png", dpi=300)

    # Create detailed table
    print("\nDetailed segment breakdown:")
    print(
        f"{'Flavor':<10} {'Taste':<12} {'Thickness':<10} {'Length':<10} {'PMI SKUs':<10} {'PMI %':<10} {'Comp SKUs':<10} {'Comp %':<10} {'Delta':<10}")
    print("-" * 100)

    for i in range(len(segments['flavors'])):
        delta = segments['pmi_pct'][i] - segments['comp_pct'][i]
        print(
            f"{segments['flavors'][i]:<10} {segments['tastes'][i]:<12} {segments['thickness'][i]:<10} {segments['lengths'][i]:<10} "
            f"{segments['pmi_skus'][i]:<10} {segments['pmi_pct'][i]:<10.1%} {segments['comp_skus'][i]:<10} {segments['comp_pct'][i]:<10.1%} "
            f"{delta:<10.1%}")

    return r_squared * 10


def analyze_passenger_mix(data, location_name):
    """
    Analyze the passenger mix (Category C) for a given location
    """
    passenger_data = data['passenger_mix']

    # Calculate the correlation (R^2)
    r_squared = np.corrcoef(passenger_data['pmi_som'], passenger_data['ideal_som'])[0, 1] ** 2

    print(f"\n--- Passenger Mix Analysis for {location_name} ---")
    print(f"R² correlation between PMI SoM and ideal SoM: {r_squared:.2f}")
    print(f"Category C score: {r_squared * 10:.2f} out of 10")

    # Create visualization
    plt.figure(figsize=(10, 7))

    # Plot PMI and ideal SoM
    plt.plot(passenger_data['segments'], passenger_data['pmi_som'], 'o-', linewidth=2, markersize=8, label='PMI SoM')
    plt.plot(passenger_data['segments'], passenger_data['ideal_som'], 's-', linewidth=2, markersize=8,
             label='Ideal SoM')

    plt.xlabel('Segment')
    plt.ylabel('Share of Market')
    plt.title(f'PMI SoM vs PAX-driven "Ideal" SoM - {location_name}\nR² = {r_squared:.2f}')
    plt.xticks(rotation=45, ha='right')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{location_name.replace(' ', '_')}_passenger_mix.png", dpi=300)

    # Create detailed table
    print("\nDetailed SoM breakdown:")
    print(f"{'Segment':<25} {'PMI SoM':<10} {'Ideal SoM':<10} {'Delta':<10}")
    print("-" * 60)

    for i in range(len(passenger_data['segments'])):
        delta = passenger_data['pmi_som'][i] - passenger_data['ideal_som'][i]
        print(
            f"{passenger_data['segments'][i]:<25} {passenger_data['pmi_som'][i]:<10.1%} {passenger_data['ideal_som'][i]:<10.1%} {delta:<10.1%}")

    return r_squared * 10


def analyze_location_cluster(data, location_name):
    """
    Analyze the location cluster (Category D) for a given location
    """
    cluster_data = data['location_cluster']

    # Calculate the correlation (R^2)
    r_squared = np.corrcoef(cluster_data['pmi_pct'], cluster_data['cluster_pct'])[0, 1] ** 2

    print(f"\n--- Location Cluster Analysis for {location_name} ---")
    print(f"R² correlation between PMI and cluster locations: {r_squared:.2f}")
    print(f"Category D score: {r_squared * 10:.2f} out of 10")

    # Create visualization
    plt.figure(figsize=(10, 7))

    # Plot PMI and cluster percentages
    x = np.arange(len(cluster_data['segments']))
    width = 0.35

    plt.bar(x - width / 2, cluster_data['pmi_pct'], width, label='Location PMI SKUs %', color='#1f77b4')
    plt.bar(x + width / 2, cluster_data['cluster_pct'], width, label='Cluster Locations SKUs %', color='#2ca02c')

    plt.xlabel('Segment')
    plt.ylabel('Percentage of SKUs')
    plt.title(f'PMI SKUs vs Cluster Locations SKUs - {location_name}\nR² = {r_squared:.2f}')
    plt.xticks(x, cluster_data['segments'], rotation=45, ha='right')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{location_name.replace(' ', '_')}_location_cluster.png", dpi=300)

    # Create detailed table
    print("\nDetailed cluster breakdown:")
    print(f"{'Segment':<25} {'PMI SKUs':<10} {'PMI %':<10} {'Cluster SKUs':<15} {'Cluster %':<10} {'Delta':<10}")
    print("-" * 85)

    for i in range(len(cluster_data['segments'])):
        delta = cluster_data['pmi_pct'][i] - cluster_data['cluster_pct'][i]
        print(
            f"{cluster_data['segments'][i]:<25} {cluster_data['pmi_skus'][i]:<10} {cluster_data['pmi_pct'][i]:<10.1%} "
            f"{cluster_data['cluster_skus'][i]:<15} {cluster_data['cluster_pct'][i]:<10.1%} {delta:<10.1%}")

    return r_squared * 10


def create_overall_summary(data, location_name, category_scores):
    """
    Create an overall summary for a location, similar to the Zurich example
    """
    avg_score = sum(category_scores.values()) / 4

    print(f"\n--- Overall Score for {location_name} ---")
    print(f"Overall score: {avg_score:.2f} out of 10")
    print(f"Classification: {data['classification']}")

    # Create a radar chart for the scores
    plt.figure(figsize=(10, 8))

    # Define the categories and scores
    categories = ['Current Range (A)', 'Category Segments (B)', 'Passenger Mix (C)', 'Location Cluster (D)']
    scores = [category_scores['Cat_A'], category_scores['Cat_B'], category_scores['Cat_C'], category_scores['Cat_D']]

    # Number of variables
    N = len(categories)

    # Create angles for each category
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += [angles[0]]  # Close the loop

    # Add scores to complete the loop
    scores += [scores[0]]

    # Create the plot
    ax = plt.subplot(111, polar=True)

    # Draw the polygon connecting all scores
    ax.plot(angles, scores, 'o-', linewidth=2, color='#1f77b4')
    ax.fill(angles, scores, alpha=0.25, color='#1f77b4')

    # Add labels
    ax.set_thetagrids([a * 180 / np.pi for a in angles[:-1]], categories)

    # Set y-axis
    ax.set_ylim(0, 10)
    plt.yticks([2, 4, 6, 8, 10], ['2', '4', '6', '8', '10'], color='grey', size=7)
    plt.ylim(0, 10)

    # Add title
    plt.title(f'Portfolio Optimization Score - {location_name}\nOverall: {avg_score:.2f}', size=15, y=1.1)

    # Add text in the center
    plt.annotate(f'{avg_score:.1f}', xy=(0, 0), xytext=(0, 0), ha='center', va='center', fontsize=20, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{location_name.replace(' ', '_')}_overall_score.png", dpi=300)

    # Create a table with key metrics
    context = data['market_context']

    print("\nKey metrics:")
    print(f"Total SKUs: {context['Total_SKUs']}")
    print(f"PMI SKUs: {context['PMI_SKUs']}")
    print(f"Competitor SKUs: {context['Comp_SKUs']}")
    print(f"Market Share: {context['Market_Share']:.1%}")
    print(f"Annual PAX: {context['PAX_Annual']:,}")
    print(f"Total Volume: {context['Total_Volume']:,}")
    print(f"Green SKUs: {context['Green_Count']}")
    print(f"Red SKUs: {context['Red_Count']}")

    return avg_score


def create_comparison_dashboard(paris_data, shanghai_data, paris_scores, shanghai_scores):
    """
    Create a side-by-side comparison dashboard of both locations
    """
    plt.figure(figsize=(15, 10))

    # Set up the grid
    gs = GridSpec(3, 4, figure=plt.gcf())

    # Overall scores section
    ax_title = plt.subplot(gs[0, :])
    ax_title.axis('off')
    ax_title.text(0.5, 0.6, 'Portfolio Optimization Comparison',
                  fontsize=20, ha='center', va='center', fontweight='bold')
    ax_title.text(0.5, 0.3, 'Paris Charles De Gaulle (HIGH PERFORMER) vs. Shanghai Hongqiao (REQUIRES OPTIMIZATION)',
                  fontsize=14, ha='center', va='center')

    # Radar charts for both locations
    ax_paris = plt.subplot(gs[1, 0:2], polar=True)
    ax_shanghai = plt.subplot(gs[1, 2:4], polar=True)

    # Categories and scores
    categories = ['Cat A', 'Cat B', 'Cat C', 'Cat D']
    paris_score_values = [paris_scores['Cat_A'], paris_scores['Cat_B'], paris_scores['Cat_C'], paris_scores['Cat_D']]
    shanghai_score_values = [shanghai_scores['Cat_A'], shanghai_scores['Cat_B'], shanghai_scores['Cat_C'],
                             shanghai_scores['Cat_D']]

    # Number of variables
    N = len(categories)

    # Create angles for each category
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += [angles[0]]  # Close the loop

    # Add scores to complete the loop
    paris_score_values += [paris_score_values[0]]
    shanghai_score_values += [shanghai_score_values[0]]

    # Paris plot
    ax_paris.plot(angles, paris_score_values, 'o-', linewidth=2, color='#009900')
    ax_paris.fill(angles, paris_score_values, alpha=0.25, color='#009900')
    ax_paris.set_thetagrids([a * 180 / np.pi for a in angles[:-1]], categories)
    ax_paris.set_ylim(0, 10)
    ax_paris.set_yticks([2, 4, 6, 8, 10])
    ax_paris.set_yticklabels(['2', '4', '6', '8', '10'], color='grey', size=8)
    ax_paris.set_title(f'Paris Charles De Gaulle\nOverall: {sum(paris_scores.values()) / 4:.2f}', size=14)
    ax_paris.annotate(f'{sum(paris_scores.values()) / 4:.1f}', xy=(0, 0), xytext=(0, 0), ha='center', va='center',
                      fontsize=20, fontweight='bold')

    # Shanghai plot
    ax_shanghai.plot(angles, shanghai_score_values, 'o-', linewidth=2, color='#CC0000')
    ax_shanghai.fill(angles, shanghai_score_values, alpha=0.25, color='#CC0000')
    ax_shanghai.set_thetagrids([a * 180 / np.pi for a in angles[:-1]], categories)
    ax_shanghai.set_ylim(0, 10)
    ax_shanghai.set_yticks([2, 4, 6, 8, 10])
    ax_shanghai.set_yticklabels(['2', '4', '6', '8', '10'], color='grey', size=8)
    ax_shanghai.set_title(f'Shanghai Hongqiao\nOverall: {sum(shanghai_scores.values()) / 4:.2f}', size=14)
    ax_shanghai.annotate(f'{sum(shanghai_scores.values()) / 4:.1f}', xy=(0, 0), xytext=(0, 0), ha='center', va='center',
                         fontsize=20, fontweight='bold')

    # Comparison tables
    ax_metrics = plt.subplot(gs[2, :])
    ax_metrics.axis('off')

    # Create comparison table
    paris_context = paris_data['market_context']
    shanghai_context = shanghai_data['market_context']

    metrics_table = {
        'Metric': ['Total SKUs', 'PMI SKUs', 'Competitor SKUs', 'Market Share', 'Annual PAX', 'Total Volume',
                   'Green SKUs', 'Red SKUs'],
        'Paris CDG': [
            paris_context['Total_SKUs'],
            paris_context['PMI_SKUs'],
            paris_context['Comp_SKUs'],
            f"{paris_context['Market_Share']:.1%}",
            f"{paris_context['PAX_Annual']:,}",
            f"{paris_context['Total_Volume']:,}",
            paris_context['Green_Count'],
            paris_context['Red_Count']
        ],
        'Shanghai': [
            shanghai_context['Total_SKUs'],
            shanghai_context['PMI_SKUs'],
            shanghai_context['Comp_SKUs'],
            f"{shanghai_context['Market_Share']:.1%}",
            f"{shanghai_context['PAX_Annual']:,}",
            f"{shanghai_context['Total_Volume']:,}",
            shanghai_context['Green_Count'],
            shanghai_context['Red_Count']
        ],
        'Difference': [
            paris_context['Total_SKUs'] - shanghai_context['Total_SKUs'],
            paris_context['PMI_SKUs'] - shanghai_context['PMI_SKUs'],
            paris_context['Comp_SKUs'] - shanghai_context['Comp_SKUs'],
            f"{paris_context['Market_Share'] - shanghai_context['Market_Share']:.1%}",
            "N/A",
            f"{paris_context['Total_Volume'] - shanghai_context['Total_Volume']:,}",
            paris_context['Green_Count'] - shanghai_context['Green_Count'],
            paris_context['Red_Count'] - shanghai_context['Red_Count']
        ]
    }

    # Create the table
    table = ax_metrics.table(
        cellText=[metrics_table['Paris CDG'], metrics_table['Shanghai'], metrics_table['Difference']],
        rowLabels=['Paris CDG', 'Shanghai', 'Difference'],
        colLabels=metrics_table['Metric'],
        cellLoc='center',
        loc='center',
        bbox=[0.05, 0.0, 0.9, 0.8]
    )

    # Adjust table formatting
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    # Highlight key differences
    for i in range(len(metrics_table['Metric'])):
        cell = table._cells[(3, i + 1)]  # Difference row
        if i == 3:  # Market Share
            if '-' in metrics_table['Difference'][i]:
                cell.set_facecolor('#ffcccc')  # Light red
            else:
                cell.set_facecolor('#ccffcc')  # Light green
        elif i == 6:  # Green SKUs
            if int(metrics_table['Difference'][i]) > 0:
                cell.set_facecolor('#ccffcc')
            else:
                cell.set_facecolor('#ffcccc')
        elif i == 7:  # Red SKUs
            if int(metrics_table['Difference'][i]) < 0:
                cell.set_facecolor('#ccffcc')
            else:
                cell.set_facecolor('#ffcccc')

    plt.tight_layout()
    plt.savefig("Portfolio_Optimization_Comparison.png", dpi=300, bbox_inches='tight')

    # Create detailed component comparison
    plt.figure(figsize=(15, 10))

    # Set up component comparison grid
    gs = GridSpec(4, 4, figure=plt.gcf())

    # Title
    ax_title = plt.subplot(gs[0, :])
    ax_title.axis('off')
    ax_title.text(0.5, 0.5, 'Detailed Component Score Comparison',
                  fontsize=18, ha='center', va='center', fontweight='bold')

    # Category A comparison
    ax_a = plt.subplot(gs[1, :2])
    ax_a.axis('off')
    ax_a.text(0.5, 0.9, 'Category A - PMI Performance',
              fontsize=14, ha='center', va='center', fontweight='bold')

    # Create component data for Cat A
    cat_a_components = {
        'Component': ['PMI Performance', 'Volume Growth', 'High Margin SKUs', 'Premium Mix'],
        'Paris': [
            paris_data['category_a_components']['PMI_Performance'],
            paris_data['category_a_components']['Volume_Growth'],
            paris_data['category_a_components']['High_Margin_SKUs'],
            paris_data['category_a_components']['Premium_Mix']
        ],
        'Shanghai': [
            shanghai_data['category_a_components']['PMI_Performance'],
            shanghai_data['category_a_components']['Volume_Growth'],
            shanghai_data['category_a_components']['High_Margin_SKUs'],
            shanghai_data['category_a_components']['Premium_Mix']
        ]
    }

    # Create Cat A table
    table_a = ax_a.table(
        cellText=[[f"{val:.2f}" if isinstance(val, float) else val for val in cat_a_components['Paris']],
                  [f"{val:.2f}" if isinstance(val, float) else val for val in cat_a_components['Shanghai']]],
        rowLabels=['Paris CDG', 'Shanghai'],
        colLabels=cat_a_components['Component'],
        cellLoc='center',
        loc='center',
        bbox=[0.05, 0.0, 0.9, 0.7]
    )

    table_a.auto_set_font_size(False)
    table_a.set_fontsize(10)
    table_a.scale(1, 1.5)

    # Category B comparison
    ax_b = plt.subplot(gs[1, 2:])
    ax_b.axis('off')
    ax_b.text(0.5, 0.9, 'Category B - Category Segments',
              fontsize=14, ha='center', va='center', fontweight='bold')

    # Create component data for Cat B
    cat_b_components = {
        'Component': ['Segment Coverage', 'Competitive Position', 'Premium Ratio', 'Innovation Score'],
        'Paris': [
            paris_data['category_b_components']['Segment_Coverage'],
            paris_data['category_b_components']['Competitive_Position'],
            paris_data['category_b_components']['Premium_Ratio'],
            paris_data['category_b_components']['Innovation_Score']
        ],
        'Shanghai': [
            shanghai_data['category_b_components']['Segment_Coverage'],
            shanghai_data['category_b_components']['Competitive_Position'],
            shanghai_data['category_b_components']['Premium_Ratio'],
            shanghai_data['category_b_components']['Innovation_Score']
        ]
    }

    # Create Cat B table
    table_b = ax_b.table(
        cellText=[[f"{val:.2f}" for val in cat_b_components['Paris']],
                  [f"{val:.2f}" for val in cat_b_components['Shanghai']]],
        rowLabels=['Paris CDG', 'Shanghai'],
        colLabels=cat_b_components['Component'],
        cellLoc='center',
        loc='center',
        bbox=[0.05, 0.0, 0.9, 0.7]
    )

    table_b.auto_set_font_size(False)
    table_b.set_fontsize(10)
    table_b.scale(1, 1.5)

    # Category C comparison
    ax_c = plt.subplot(gs[2, :2])
    ax_c.axis('off')
    ax_c.text(0.5, 0.9, 'Category C - Passenger Mix',
              fontsize=14, ha='center', va='center', fontweight='bold')

    # Create component data for Cat C
    cat_c_components = {
        'Component': ['PAX Alignment', 'Nationality Mix', 'Traveler Type', 'Seasonal Adjustment'],
        'Paris': [
            paris_data['category_c_components']['PAX_Alignment'],
            paris_data['category_c_components']['Nationality_Mix'],
            paris_data['category_c_components']['Traveler_Type'],
            paris_data['category_c_components']['Seasonal_Adjustment']
        ],
        'Shanghai': [
            shanghai_data['category_c_components']['PAX_Alignment'],
            shanghai_data['category_c_components']['Nationality_Mix'],
            shanghai_data['category_c_components']['Traveler_Type'],
            shanghai_data['category_c_components']['Seasonal_Adjustment']
        ]
    }

    # Create Cat C table
    table_c = ax_c.table(
        cellText=[[f"{val:.2f}" for val in cat_c_components['Paris']],
                  [f"{val:.2f}" for val in cat_c_components['Shanghai']]],
        rowLabels=['Paris CDG', 'Shanghai'],
        colLabels=cat_c_components['Component'],
        cellLoc='center',
        loc='center',
        bbox=[0.05, 0.0, 0.9, 0.7]
    )

    table_c.auto_set_font_size(False)
    table_c.set_fontsize(10)
    table_c.scale(1, 1.5)

    # Category D comparison
    ax_d = plt.subplot(gs[2, 2:])
    ax_d.axis('off')
    ax_d.text(0.5, 0.9, 'Category D - Location Clusters',
              fontsize=14, ha='center', va='center', fontweight='bold')

    # Create component data for Cat D
    cat_d_components = {
        'Component': ['Cluster Similarity', 'Regional Alignment', 'Size Compatibility', 'Format Distribution'],
        'Paris': [
            paris_data['category_d_components']['Cluster_Similarity'],
            paris_data['category_d_components']['Regional_Alignment'],
            paris_data['category_d_components']['Size_Compatibility'],
            paris_data['category_d_components']['Format_Distribution']
        ],
        'Shanghai': [
            shanghai_data['category_d_components']['Cluster_Similarity'],
            shanghai_data['category_d_components']['Regional_Alignment'],
            shanghai_data['category_d_components']['Size_Compatibility'],
            shanghai_data['category_d_components']['Format_Distribution']
        ]
    }

    # Create Cat D table
    table_d = ax_d.table(
        cellText=[[f"{val:.2f}" for val in cat_d_components['Paris']],
                  [f"{val:.2f}" for val in cat_d_components['Shanghai']]],
        rowLabels=['Paris CDG', 'Shanghai'],
        colLabels=cat_d_components['Component'],
        cellLoc='center',
        loc='center',
        bbox=[0.05, 0.0, 0.9, 0.7]
    )

    table_d.auto_set_font_size(False)
    table_d.set_fontsize(10)
    table_d.scale(1, 1.5)

    # Highlight critical components - all of Cat C and Cat D for Shanghai
    for i in range(4):
        # Cat C
        cell_c = table_c._cells[(2, i + 1)]  # Shanghai row
        if cat_c_components['Shanghai'][i] < 0.1:
            cell_c.set_facecolor('#ffaaaa')  # Darker red for very low scores
        elif cat_c_components['Shanghai'][i] < 0.5:
            cell_c.set_facecolor('#ffcccc')  # Light red

        # Cat D
        cell_d = table_d._cells[(2, i + 1)]  # Shanghai row
        if cat_d_components['Shanghai'][i] < 0.1:
            cell_d.set_facecolor('#ffaaaa')  # Darker red for very low scores
        elif cat_d_components['Shanghai'][i] < 0.5:
            cell_d.set_facecolor('#ffcccc')  # Light red

    # SKU Performance Summary
    ax_sku = plt.subplot(gs[3, :])
    ax_sku.axis('off')
    ax_sku.text(0.5, 0.9, 'SKU Performance Summary',
                fontsize=14, ha='center', va='center', fontweight='bold')

    # Create SKU summary data
    sku_summary = {
        'Metric': ['Top SKU Volume', 'Average Growth', 'Average Margin', 'Positive Growth SKUs',
                   'Negative Growth SKUs'],
        'Paris': [
            f"{max(paris_data['sku_data']['Volume_2024']):,}",
            f"{paris_data['sku_data']['Growth'].mean():.1%}",
            f"{paris_data['sku_data']['Margin'].mean():.2f}",
            len(paris_data['sku_data'][paris_data['sku_data']['Growth'] > 0]),
            len(paris_data['sku_data'][paris_data['sku_data']['Growth'] < 0])
        ],
        'Shanghai': [
            f"{max(shanghai_data['sku_data']['Volume_2024']):,}",
            f"{shanghai_data['sku_data']['Growth'].mean():.1%}",
            f"{shanghai_data['sku_data']['Margin'].mean():.2f}",
            len(shanghai_data['sku_data'][shanghai_data['sku_data']['Growth'] > 0]),
            len(shanghai_data['sku_data'][shanghai_data['sku_data']['Growth'] < 0])
        ]
    }

    # Create SKU summary table
    table_sku = ax_sku.table(
        cellText=[sku_summary['Paris'], sku_summary['Shanghai']],
        rowLabels=['Paris CDG', 'Shanghai'],
        colLabels=sku_summary['Metric'],
        cellLoc='center',
        loc='center',
        bbox=[0.15, 0.0, 0.7, 0.7]
    )

    table_sku.auto_set_font_size(False)
    table_sku.set_fontsize(10)
    table_sku.scale(1, 1.5)

    # Highlight important differences in SKU performance
    # Average Growth
    cell_growth_paris = table_sku._cells[(1, 2)]  # Paris row, Average Growth column
    cell_growth_shanghai = table_sku._cells[(2, 2)]  # Shanghai row, Average Growth column

    if float(paris_data['sku_data']['Growth'].mean()) > 0:
        cell_growth_paris.set_facecolor('#ccffcc')  # Light green
    else:
        cell_growth_paris.set_facecolor('#ffcccc')  # Light red

    if float(shanghai_data['sku_data']['Growth'].mean()) > 0:
        cell_growth_shanghai.set_facecolor('#ccffcc')  # Light green
    else:
        cell_growth_shanghai.set_facecolor('#ffcccc')  # Light red

    # Negative Growth SKUs
    cell_neg_paris = table_sku._cells[(1, 5)]  # Paris row, Negative Growth SKUs column
    cell_neg_shanghai = table_sku._cells[(2, 5)]  # Shanghai row, Negative Growth SKUs column

    if int(sku_summary['Paris'][4]) > 2:
        cell_neg_paris.set_facecolor('#ffcccc')  # Light red
    if int(sku_summary['Shanghai'][4]) > 2:
        cell_neg_shanghai.set_facecolor('#ffcccc')  # Light red

    plt.tight_layout()
    plt.savefig("Portfolio_Component_Comparison.png", dpi=300, bbox_inches='tight')

    print("\nComparison dashboard created successfully.")
    print("Files saved: Portfolio_Optimization_Comparison.png and Portfolio_Component_Comparison.png")


def main():
    """
    Main function to run the analysis
    """
    # Create comparison data
    paris_data, shanghai_data = create_comparison_data()

    # Analyze Paris Charles De Gaulle
    print("\n========== PARIS CHARLES DE GAULLE ANALYSIS ==========")
    paris_cat_a = analyze_current_range(paris_data, "Paris Charles De Gaulle")
    paris_cat_b = analyze_category_segments(paris_data, "Paris Charles De Gaulle")
    paris_cat_c = analyze_passenger_mix(paris_data, "Paris Charles De Gaulle")
    paris_cat_d = analyze_location_cluster(paris_data, "Paris Charles De Gaulle")

    paris_scores = {
        'Cat_A': paris_cat_a,
        'Cat_B': paris_cat_b,
        'Cat_C': paris_cat_c,
        'Cat_D': paris_cat_d
    }

    create_overall_summary(paris_data, "Paris Charles De Gaulle", paris_scores)

    # Analyze Shanghai Hongqiao
    print("\n========== SHANGHAI HONGQIAO ANALYSIS ==========")
    shanghai_cat_a = analyze_current_range(shanghai_data, "Shanghai Hongqiao")
    shanghai_cat_b = analyze_category_segments(shanghai_data, "Shanghai Hongqiao")
    shanghai_cat_c = analyze_passenger_mix(shanghai_data, "Shanghai Hongqiao")
    shanghai_cat_d = analyze_location_cluster(shanghai_data, "Shanghai Hongqiao")

    shanghai_scores = {
        'Cat_A': shanghai_cat_a,
        'Cat_B': shanghai_cat_b,
        'Cat_C': shanghai_cat_c,
        'Cat_D': shanghai_cat_d
    }

    create_overall_summary(shanghai_data, "Shanghai Hongqiao", shanghai_scores)

    # Create comparison dashboard
    print("\n========== CREATING COMPARISON DASHBOARD ==========")
    create_comparison_dashboard(paris_data, shanghai_data, paris_scores, shanghai_scores)

    print("\n========== ANALYSIS COMPLETE ==========")
    print("All visualizations and reports have been generated successfully.")


if __name__ == "__main__":
    main()