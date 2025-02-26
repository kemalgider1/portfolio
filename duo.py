import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, Rectangle, FancyBboxPatch
from matplotlib.path import Path
import matplotlib.colors as mcolors
import matplotlib.patheffects as path_effects
import seaborn as sns
from matplotlib.gridspec import GridSpec
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# Load the data
def load_data():
    # Main location data with category scores
    location_data = pd.read_csv('results/clustered_locations_20250225_030538.csv')

    # Create simulated SKU data for each location (in practice, this would be loaded from actual files)
    # For Dubai
    dubai_skus = pd.DataFrame({
        'SKU': [f'SKU_D{i}' for i in range(1, 21)],
        'Brand_Family': ['MARLBORO', 'MARLBORO', 'PARLIAMENT', 'PARLIAMENT', 'HEETS',
                         'HEETS', 'L&M', 'L&M', 'CHESTERFIELD', 'CHESTERFIELD',
                         'MARLBORO', 'PARLIAMENT', 'HEETS', 'L&M', 'CHESTERFIELD',
                         'MARLBORO', 'PARLIAMENT', 'HEETS', 'L&M', 'CHESTERFIELD'],
        'Volume_2023': np.random.randint(100000, 500000, 20),
        'Volume_2024': np.random.randint(120000, 550000, 20),
        'Growth': np.random.uniform(-0.1, 0.3, 20),
        'Margin': np.random.uniform(0.65, 0.85, 20),
        'Flavor': np.random.choice(['Regular', 'Menthol', 'Flavor Plus'], 20),
        'Strength': np.random.choice(['Full Flavor', 'Lights', 'Ultra Lights'], 20),
        'Length': np.random.choice(['KS', 'Super Slims', '100s'], 20),
        'TMO': 'PMI'
    })

    # For Prague
    prague_skus = pd.DataFrame({
        'SKU': [f'SKU_P{i}' for i in range(1, 16)],
        'Brand_Family': ['MARLBORO', 'MARLBORO', 'PARLIAMENT', 'L&M', 'L&M',
                         'CHESTERFIELD', 'CHESTERFIELD', 'LARK', 'LARK', 'BOND',
                         'MARLBORO', 'PARLIAMENT', 'L&M', 'LARK', 'BOND'],
        'Volume_2023': np.random.randint(50000, 300000, 15),
        'Volume_2024': np.random.randint(40000, 280000, 15),
        'Growth': np.random.uniform(-0.3, 0.15, 15),
        'Margin': np.random.uniform(0.6, 0.8, 15),
        'Flavor': np.random.choice(['Regular', 'Menthol'], 15),
        'Strength': np.random.choice(['Full Flavor', 'Lights'], 15),
        'Length': np.random.choice(['KS', '100s'], 15),
        'TMO': 'PMI'
    })

    # Add additional metrics for location context
    dubai_context = {
        'Total_SKUs': 253,
        'PMI_SKUs': 35,
        'Comp_SKUs': 218,
        'Total_Volume': 611738800,
        'PMI_Volume': 295194600,
        'Market_Share': 0.4825,
        'Green_Count': 11,
        'Red_Count': 9,
        'PAX_Annual': 86700000,
        'Category_A_Components': {
            'PMI_Performance': 0.72,
            'Volume_Growth': 0.15,
            'High_Margin_SKUs': 9,
            'Premium_Mix': 0.63
        },
        'Category_B_Components': {
            'Segment_Coverage': 0.78,
            'Competitive_Position': 0.55,
            'Premium_Ratio': 0.67,
            'Innovation_Score': 0.43
        },
        'Category_C_Components': {
            'PAX_Alignment': 0.64,
            'Nationality_Mix': 0.74,
            'Traveler_Type': 0.53,
            'Seasonal_Adjustment': 0.42
        },
        'Category_D_Components': {
            'Cluster_Similarity': 0.58,
            'Regional_Alignment': 0.62,
            'Size_Compatibility': 0.45,
            'Format_Distribution': 0.39
        }
    }

    prague_context = {
        'Total_SKUs': 28,
        'PMI_SKUs': 15,
        'Comp_SKUs': 13,
        'Total_Volume': 82500000,
        'PMI_Volume': 50325000,
        'Market_Share': 0.61,
        'Green_Count': 1,
        'Red_Count': 3,
        'PAX_Annual': 17800000,
        'Category_A_Components': {
            'PMI_Performance': 0.68,
            'Volume_Growth': -0.03,
            'High_Margin_SKUs': 4,
            'Premium_Mix': 0.55
        },
        'Category_B_Components': {
            'Segment_Coverage': 0.28,
            'Competitive_Position': 0.33,
            'Premium_Ratio': 0.41,
            'Innovation_Score': 0.22
        },
        'Category_C_Components': {
            'PAX_Alignment': 0.31,
            'Nationality_Mix': 0.38,
            'Traveler_Type': 0.27,
            'Seasonal_Adjustment': 0.19
        },
        'Category_D_Components': {
            'Cluster_Similarity': 0.05,
            'Regional_Alignment': 0.12,
            'Size_Compatibility': 0.0,
            'Format_Distribution': 0.0
        }
    }

    return location_data, dubai_skus, prague_skus, dubai_context, prague_context


# Extract relevant location data
def extract_location_data(location_data, location_name):
    return location_data[location_data['Location'] == location_name].iloc[0]


def create_location_visualization(location_name, location_data, sku_data, context_data):
    """
    Create an advanced hexagonal visualization for a single location.

    Parameters:
    -----------
    location_name : str
        Name of the location
    location_data : Series
        Location score data from the main dataset
    sku_data : DataFrame
        SKU-level data for the location
    context_data : dict
        Additional contextual metrics for the location
    """
    # Create figure with a specific size and style
    plt.figure(figsize=(16, 12), facecolor='#f8f8f8')
    gs = GridSpec(3, 3, figure=plt.gcf())

    # Create the main hexagonal plot
    ax_main = plt.subplot(gs[0:2, 0:2])
    ax_main.set_aspect('equal')

    # Define hexagon coordinates
    n_sides = 6
    angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)
    angles = np.roll(angles, -1)  # Rotate to start from the top

    # Create outer and inner hexagons
    radius_outer = 10
    radius_inner = radius_outer * 0.2

    hex_outer_x = radius_outer * np.cos(angles)
    hex_outer_y = radius_outer * np.sin(angles)
    hex_inner_x = radius_inner * np.cos(angles)
    hex_inner_y = radius_inner * np.sin(angles)

    # Draw the outer hexagon
    outer_hex = Polygon(np.column_stack([hex_outer_x, hex_outer_y]),
                        closed=True, fill=False, edgecolor='#333333',
                        linewidth=2, zorder=1)
    ax_main.add_patch(outer_hex)

    # Draw the inner hexagon (center point)
    inner_hex = Polygon(np.column_stack([hex_inner_x, hex_inner_y]),
                        closed=True, fill=True, facecolor='#333333',
                        edgecolor='#333333', linewidth=1.5, zorder=1)
    ax_main.add_patch(inner_hex)

    # Draw axis lines connecting inner and outer hexagons
    for i in range(n_sides):
        ax_main.plot([hex_inner_x[i], hex_outer_x[i]],
                     [hex_inner_y[i], hex_outer_y[i]],
                     color='#333333', linestyle='-', linewidth=1.5,
                     alpha=0.5, zorder=1)

    # Set up scaling for the score values (0-10 scale to radius)
    def scale_value(val):
        return radius_inner + ((radius_outer - radius_inner) * val / 10)

    # Create the category points - order: Cat_A, Cat_B, Cat_C, Cat_D, Market_Share, Growth
    categories = ['Cat_A', 'Cat_B', 'Cat_C', 'Cat_D', 'MS', 'Growth']
    score_values = [
        location_data['Cat_A'],
        location_data['Cat_B'],
        location_data['Cat_C'],
        location_data['Cat_D'],
        context_data['Market_Share'] * 10,  # Scale to 0-10
        (np.mean(sku_data['Growth']) + 0.3) * 10  # Scale from -0.3 to +0.3 to 0-10
    ]

    # Plot score points and value labels
    score_x = []
    score_y = []

    for i, (angle, score) in enumerate(zip(angles, score_values)):
        # Calculate position
        radius = scale_value(score)
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        score_x.append(x)
        score_y.append(y)

        # Plot point
        ax_main.scatter(x, y, s=120, color=f'C{i}',
                        edgecolor='white', linewidth=1.5, zorder=3)

        # Add score label
        label_radius = radius + 0.8
        label_x = label_radius * np.cos(angle)
        label_y = label_radius * np.sin(angle)

        # Add category label at the outer edge
        category_radius = radius_outer + 1.5
        cat_x = category_radius * np.cos(angle)
        cat_y = category_radius * np.sin(angle)

        cat_text = ax_main.text(cat_x, cat_y, categories[i],
                                ha='center', va='center', fontsize=14,
                                fontweight='bold', color=f'C{i}')
        cat_text.set_path_effects([path_effects.withStroke(linewidth=5, foreground='white')])

        # Add score value
        score_text = ax_main.text(label_x, label_y, f'{score:.1f}',
                                  ha='center', va='center', fontsize=12,
                                  fontweight='bold', color=f'C{i}')
        score_text.set_path_effects([path_effects.withStroke(linewidth=4, foreground='white')])

    # Connect the score points to form a polygon
    score_x.append(score_x[0])
    score_y.append(score_y[0])
    ax_main.plot(score_x, score_y, '-', color='#333333', linewidth=2.5, alpha=0.8, zorder=2)

    # Fill the polygon with a translucent color
    score_polygon = Polygon(np.column_stack([score_x, score_y]),
                            closed=True, fill=True,
                            facecolor='#333333', alpha=0.15, zorder=1)
    ax_main.add_patch(score_polygon)

    # Add a title for the main chart
    avg_score = location_data['Avg_Score']
    ax_main.text(0, 0, f'{avg_score:.1f}',
                 ha='center', va='center', fontsize=18,
                 fontweight='bold', color='white', zorder=3)

    ax_main.set_xlim(-radius_outer * 1.4, radius_outer * 1.4)
    ax_main.set_ylim(-radius_outer * 1.4, radius_outer * 1.4)
    ax_main.axis('off')

    # Add SKU Performance section (Top 5 SKUs)
    ax_skus = plt.subplot(gs[0, 2])
    top_skus = sku_data.sort_values('Volume_2024', ascending=False).head(5)

    # Create a horizontal bar chart for SKU volumes
    colors = [plt.cm.viridis(x / 10) for x in range(5)]
    top_skus['SKU_Short'] = top_skus['SKU'].str[:8] + '...'

    bars = ax_skus.barh(top_skus['SKU_Short'], top_skus['Volume_2024'], color=colors, height=0.6)

    # Add growth indicators
    for i, (idx, row) in enumerate(top_skus.iterrows()):
        growth = row['Growth']
        if growth > 0:
            ax_skus.text(row['Volume_2024'] + 5000, i, f'↑ {growth:.1%}',
                         va='center', fontsize=10, color='green', fontweight='bold')
        else:
            ax_skus.text(row['Volume_2024'] + 5000, i, f'↓ {abs(growth):.1%}',
                         va='center', fontsize=10, color='red', fontweight='bold')

    ax_skus.set_title('Top 5 SKUs by Volume', fontsize=14, fontweight='bold')
    ax_skus.spines['top'].set_visible(False)
    ax_skus.spines['right'].set_visible(False)
    ax_skus.set_xlabel('Volume (2024)', fontsize=10)

    # Add Brand Mix section
    ax_brands = plt.subplot(gs[1, 2])
    brand_mix = sku_data.groupby('Brand_Family')['Volume_2024'].sum().reset_index()
    brand_mix = brand_mix.sort_values('Volume_2024', ascending=True)

    # Create a horizontal bar chart for brand volumes
    colors = plt.cm.Oranges(np.linspace(0.4, 0.8, len(brand_mix)))
    ax_brands.barh(brand_mix['Brand_Family'], brand_mix['Volume_2024'], color=colors, height=0.6)

    ax_brands.set_title('Brand Family Mix', fontsize=14, fontweight='bold')
    ax_brands.spines['top'].set_visible(False)
    ax_brands.spines['right'].set_visible(False)
    ax_brands.set_xlabel('Volume (2024)', fontsize=10)

    # Add Key Metrics section
    ax_metrics = plt.subplot(gs[2, :])
    ax_metrics.axis('off')

    # Define metrics categories and layout
    metrics_text = [
        f"Total SKUs: {context_data['Total_SKUs']}",
        f"PMI SKUs: {context_data['PMI_SKUs']}",
        f"Competitor SKUs: {context_data['Comp_SKUs']}",
        f"Market Share: {context_data['Market_Share']:.1%}",
        f"Annual PAX: {context_data['PAX_Annual']:,}",
        f"Total Volume: {context_data['Total_Volume']:,}",
        f"Green SKUs: {context_data['Green_Count']}",
        f"Red SKUs: {context_data['Red_Count']}",
        f"Score Range: {location_data['Score_Range']:.1f}",
        f"Cluster: {int(location_data['Cluster'])}"
    ]

    # Create fancy metric boxes
    cols = 5
    rows = 2
    width = 0.17
    height = 0.08

    for i, metric in enumerate(metrics_text):
        row = i // cols
        col = i % cols

        x = 0.05 + col * (width + 0.02)
        y = 0.65 - row * (height + 0.02)

        rect = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.03",
                              facecolor='white', edgecolor='#333333', alpha=0.8)
        ax_metrics.add_patch(rect)

        ax_metrics.text(x + width / 2, y + height / 2, metric,
                        ha='center', va='center', fontsize=12)

    # Add Scoring Component breakdown
    components_box = plt.axes([0.05, 0.05, 0.9, 0.4])
    components_box.axis('off')

    # Define component titles and data
    component_titles = [
        "Category A Components (PMI Performance)",
        "Category B Components (Category Segments)",
        "Category C Components (Passenger Mix)",
        "Category D Components (Location Clusters)"
    ]

    component_data = [
        context_data['Category_A_Components'],
        context_data['Category_B_Components'],
        context_data['Category_C_Components'],
        context_data['Category_D_Components']
    ]

    # Create component boxes
    for i, (title, data) in enumerate(zip(component_titles, component_data)):
        col = i % 4

        x = 0.05 + col * 0.23
        y = 0.15

        # Draw component box
        rect = FancyBboxPatch((x, y), 0.21, 0.25, boxstyle="round,pad=0.03",
                              facecolor=f'C{i}', edgecolor='#333333', alpha=0.15)
        components_box.add_patch(rect)

        # Add title
        components_box.text(x + 0.105, y + 0.23, title.split(" (")[0],
                            ha='center', va='center', fontsize=12, fontweight='bold')
        components_box.text(x + 0.105, y + 0.2, f"({title.split(' (')[1]}",
                            ha='center', va='center', fontsize=10)

        # Add component scores
        y_pos = y + 0.15
        for key, value in data.items():
            y_pos -= 0.04
            components_box.text(x + 0.02, y_pos, key.replace('_', ' '),
                                ha='left', va='center', fontsize=9)

            # Add score bar
            bar_length = 0.1 * value
            bar_height = 0.01
            rect = Rectangle((x + 0.1, y_pos - bar_height / 2), bar_length, bar_height,
                             facecolor=f'C{i}', alpha=0.7)
            components_box.add_patch(rect)

            # Add score value
            components_box.text(x + 0.19, y_pos, f"{value:.2f}",
                                ha='right', va='center', fontsize=9, fontweight='bold')

    # Add location name and title
    plt.suptitle(f'Portfolio Optimization Analysis: {location_name}',
                 fontsize=24, y=0.98, fontweight='bold')

    # Add classification based on average score
    if location_data['Avg_Score'] >= 4.0:
        classification = "HIGH PERFORMER"
        color = "green"
    else:
        classification = "REQUIRES OPTIMIZATION"
        color = "red"

    plt.figtext(0.5, 0.94, classification,
                fontsize=16, ha='center', color=color,
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5',
                          edgecolor=color))

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(f'{location_name}_portfolio_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Visualization for {location_name} saved as '{location_name}_portfolio_analysis.png'")


def generate_location_report(location_name, location_data, sku_data, context_data):
    """
    Generate a detailed report for the location.

    Parameters:
    -----------
    location_name : str
        Name of the location
    location_data : Series
        Location score data from the main dataset
    sku_data : DataFrame
        SKU-level data for the location
    context_data : dict
        Additional contextual metrics for the location
    """
    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d")

    # Determine if location is high or low performer
    if location_data['Avg_Score'] >= 4.0:
        performance_level = "HIGH PERFORMER"
    else:
        performance_level = "REQUIRES OPTIMIZATION"

    # Calculate metrics for the report
    top_skus = sku_data.sort_values('Volume_2024', ascending=False).head(5)
    top_growth_skus = sku_data[sku_data['Volume_2024'] > sku_data['Volume_2024'].quantile(0.25)].sort_values('Growth',
                                                                                                             ascending=False).head(
        3)
    declining_skus = sku_data[sku_data['Growth'] < 0].sort_values('Growth', ascending=True).head(3)
    brand_mix = sku_data.groupby('Brand_Family')['Volume_2024'].sum().reset_index().sort_values('Volume_2024',
                                                                                                ascending=False)

    # Generate the report
    with open(f'{location_name}_portfolio_report.md', 'w') as f:
        # Header
        f.write(f"# Portfolio Optimization Analysis: {location_name}\n")
        f.write(f"**Date:** {timestamp}  \n")
        f.write(f"**Classification:** {performance_level}  \n\n")

        # Executive Summary
        f.write("## Executive Summary\n\n")

        f.write(f"{location_name} is a {performance_level.lower()} in our portfolio optimization analysis ")
        f.write(f"with an average score of {location_data['Avg_Score']:.2f} out of 10. ")

        if performance_level == "HIGH PERFORMER":
            f.write(f"The location demonstrates strong performance across multiple categories, ")
            f.write(f"particularly in {get_strongest_category(location_data)} ")
            f.write(f"(score: {get_strongest_score(location_data):.1f}). ")
            f.write(f"With a market share of {context_data['Market_Share']:.1%}, ")
            f.write(f"{location_name} represents a key strategic location in our portfolio.\n\n")
        else:
            f.write(f"The location shows uneven performance across categories, ")
            f.write(f"with particular challenges in {get_weakest_category(location_data)} ")
            f.write(f"(score: {get_weakest_score(location_data):.1f}). ")
            f.write(f"Despite a market share of {context_data['Market_Share']:.1%}, ")
            f.write(f"there are significant opportunities for portfolio optimization.\n\n")

        # Category Score Breakdown
        f.write("## Category Score Breakdown\n\n")
        f.write("| Category | Score | Description | Key Components |\n")
        f.write("|----------|-------|-------------|----------------|\n")

        # Category A
        f.write(f"| **Category A** (PMI Performance) | {location_data['Cat_A']:.2f} | ")
        if location_data['Cat_A'] >= 6.0:
            f.write("Strong PMI performance | ")
        elif location_data['Cat_A'] >= 4.0:
            f.write("Moderate PMI performance | ")
        else:
            f.write("Weak PMI performance | ")

        comp_a = context_data['Category_A_Components']
        component_text = f"PMI Performance: {comp_a['PMI_Performance']:.2f}, "
        component_text += f"Volume Growth: {comp_a['Volume_Growth']:.2f}, "
        component_text += f"High Margin SKUs: {comp_a['High_Margin_SKUs']}, "
        component_text += f"Premium Mix: {comp_a['Premium_Mix']:.2f}"
        f.write(f"{component_text} |\n")

        # Category B
        f.write(f"| **Category B** (Category Segments) | {location_data['Cat_B']:.2f} | ")
        if location_data['Cat_B'] >= 6.0:
            f.write("Strong category segmentation | ")
        elif location_data['Cat_B'] >= 4.0:
            f.write("Moderate category segmentation | ")
        else:
            f.write("Weak category segmentation | ")

        comp_b = context_data['Category_B_Components']
        component_text = f"Segment Coverage: {comp_b['Segment_Coverage']:.2f}, "
        component_text += f"Competitive Position: {comp_b['Competitive_Position']:.2f}, "
        component_text += f"Premium Ratio: {comp_b['Premium_Ratio']:.2f}, "
        component_text += f"Innovation Score: {comp_b['Innovation_Score']:.2f}"
        f.write(f"{component_text} |\n")

        # Category C
        f.write(f"| **Category C** (Passenger Mix) | {location_data['Cat_C']:.2f} | ")
        if location_data['Cat_C'] >= 6.0:
            f.write("Strong passenger alignment | ")
        elif location_data['Cat_C'] >= 4.0:
            f.write("Moderate passenger alignment | ")
        else:
            f.write("Weak passenger alignment | ")

        comp_c = context_data['Category_C_Components']
        component_text = f"PAX Alignment: {comp_c['PAX_Alignment']:.2f}, "
        component_text += f"Nationality Mix: {comp_c['Nationality_Mix']:.2f}, "
        component_text += f"Traveler Type: {comp_c['Traveler_Type']:.2f}, "
        component_text += f"Seasonal Adjustment: {comp_c['Seasonal_Adjustment']:.2f}"
        f.write(f"{component_text} |\n")

        # Category D
        f.write(f"| **Category D** (Location Clusters) | {location_data['Cat_D']:.2f} | ")
        if location_data['Cat_D'] >= 6.0:
            f.write("Strong location clustering | ")
        elif location_data['Cat_D'] >= 4.0:
            f.write("Moderate location clustering | ")
        else:
            f.write("Weak location clustering | ")

        comp_d = context_data['Category_D_Components']
        component_text = f"Cluster Similarity: {comp_d['Cluster_Similarity']:.2f}, "
        component_text += f"Regional Alignment: {comp_d['Regional_Alignment']:.2f}, "
        component_text += f"Size Compatibility: {comp_d['Size_Compatibility']:.2f}, "
        component_text += f"Format Distribution: {comp_d['Format_Distribution']:.2f}"
        f.write(f"{component_text} |\n\n")

        # SKU Performance Analysis
        f.write("## SKU Performance Analysis\n\n")

        # Top SKUs by Volume
        f.write("### Top 5 SKUs by Volume\n\n")
        f.write("| SKU | Brand Family | Volume 2024 | Growth | Margin |\n")
        f.write("|-----|--------------|-------------|--------|--------|\n")

        for _, row in top_skus.iterrows():
            f.write(f"| {row['SKU']} | {row['Brand_Family']} | {row['Volume_2024']:,} | ")
            f.write(f"{row['Growth']:.1%} | {row['Margin']:.2f} |\n")

        f.write("\n")

        # Top Growing SKUs
        f.write("### Top Growing SKUs\n\n")
        f.write("| SKU | Brand Family | Volume 2024 | Growth | Margin |\n")
        f.write("|-----|--------------|-------------|--------|--------|\n")

        for _, row in top_growth_skus.iterrows():
            f.write(f"| {row['SKU']} | {row['Brand_Family']} | {row['Volume_2024']:,} | ")
            f.write(f"{row['Growth']:.1%} | {row['Margin']:.2f} |\n")

        f.write("\n")

        # Declining SKUs
        f.write("### Declining SKUs\n\n")
        f.write("| SKU | Brand Family | Volume 2024 | Growth | Margin |\n")
        f.write("|-----|--------------|-------------|--------|--------|\n")

        for _, row in declining_skus.iterrows():
            f.write(f"| {row['SKU']} | {row['Brand_Family']} | {row['Volume_2024']:,} | ")
            f.write(f"{row['Growth']:.1%} | {row['Margin']:.2f} |\n")

        f.write("\n")

        # Brand Mix Analysis
        f.write("## Brand Mix Analysis\n\n")
        f.write("| Brand Family | Volume 2024 | Share of Portfolio |\n")
        f.write("|--------------|-------------|--------------------|\n")

        total_volume = brand_mix['Volume_2024'].sum()
        for _, row in brand_mix.iterrows():
            share = row['Volume_2024'] / total_volume
            f.write(f"| {row['Brand_Family']} | {row['Volume_2024']:,} | {share:.1%} |\n")

        f.write("\n")

        # Market Context
        f.write("## Market Context\n\n")
        f.write(f"* **Total SKUs:** {context_data['Total_SKUs']}\n")
        f.write(f"* **PMI SKUs:** {context_data['PMI_SKUs']}\n")
        f.write(f"* **Competitor SKUs:** {context_data['Comp_SKUs']}\n")
        f.write(f"* **Market Share:** {context_data['Market_Share']:.1%}\n")
        f.write(f"* **Annual PAX:** {context_data['PAX_Annual']:,}\n")
        f.write(f"* **Total Volume:** {context_data['Total_Volume']:,}\n")
        f.write(f"* **Green SKUs:** {context_data['Green_Count']}\n")
        f.write(f"* **Red SKUs:** {context_data['Red_Count']}\n\n")

        # Scoring Methodology
        f.write("## Scoring Methodology\n\n")
        f.write("The portfolio optimization scoring uses a multi-faceted approach with four key categories:\n\n")

        f.write(
            "1. **Category A (PMI Performance)**: Evaluates the core performance metrics of PMI products at the location, including:\n")
        f.write("   - Volume and value growth trends\n")
        f.write("   - Margin performance of SKUs\n")
        f.write("   - Premium product mix\n")
        f.write("   - Overall PMI competitiveness\n\n")

        f.write(
            "2. **Category B (Category Segments)**: Assesses how well PMI's portfolio covers key product segments compared to competitors:\n")
        f.write("   - Segment representation across flavor profiles\n")
        f.write("   - Format and price point coverage\n")
        f.write("   - Innovation performance\n")
        f.write("   - Competitive product positioning\n\n")

        f.write(
            "3. **Category C (Passenger Mix)**: Measures how well the portfolio aligns with traveler demographics:\n")
        f.write("   - Nationality mix alignment\n")
        f.write("   - Traveler preference matching\n")
        f.write("   - Seasonal travel pattern alignment\n")
        f.write("   - Regional consumer preference coverage\n\n")

        f.write(
            "4. **Category D (Location Clusters)**: Evaluates how well the location's portfolio matches similar locations:\n")
        f.write("   - Performance versus cluster benchmark locations\n")
        f.write("   - Regional similarity metrics\n")
        f.write("   - Size and format compatibility with similar locations\n")
        f.write("   - Best practice implementation from cluster leaders\n\n")

        # Recommendations
        f.write("## Strategic Recommendations\n\n")

        if performance_level == "HIGH PERFORMER":
            # Recommendations for high performers
            f.write(
                "Based on the portfolio analysis, the following recommendations are provided to maintain and enhance performance:\n\n")

            f.write("1. **Maintain Premium Focus**: Continue emphasis on premium brands like ")
            f.write(f"{brand_mix.iloc[0]['Brand_Family']} and {brand_mix.iloc[1]['Brand_Family']}, ")
            f.write("which drive both volume and value.\n\n")

            f.write("2. **Optimize SKU Rationalization**: Consider phasing out declining SKUs like ")
            if len(declining_skus) > 0:
                f.write(
                    f"{declining_skus.iloc[0]['SKU']} and {declining_skus.iloc[1]['SKU'] if len(declining_skus) > 1 else ''}.")
            else:
                f.write("underperforming variants while maintaining breadth of portfolio.")
            f.write("\n\n")

            f.write(
                "3. **Leverage Cluster Insights**: Implement best practices from similar high-performing locations in Cluster ")
            f.write(f"{int(location_data['Cluster'])}.\n\n")

            f.write(
                "4. **Expand Innovation**: Introduce targeted innovations based on the success of top-growing SKUs like ")
            f.write(f"{top_growth_skus.iloc[0]['SKU']} ({top_growth_skus.iloc[0]['Growth']:.1%} growth).\n\n")

            f.write(
                "5. **Enhance Premium Mix**: Further develop premium offerings to improve margins and strengthen Category A score.\n\n")
        else:
            # Recommendations for low performers
            f.write(
                "Based on the portfolio analysis, the following recommendations are provided to improve performance:\n\n")

            f.write(
                f"1. **Address {get_weakest_category(location_data)} Gap**: Focus on improving the weakest category ")
            f.write(f"({get_weakest_score(location_data):.1f}) by implementing targeted strategies:\n")

            if get_weakest_category(location_data) == "Cat_B":
                f.write("   - Expand segment coverage across flavor profiles\n")
                f.write("   - Introduce formats that address competitive gaps\n")
                f.write("   - Enhance pricing strategy to better compete with local offerings\n\n")
            elif get_weakest_category(location_data) == "Cat_C":
                f.write("   - Align portfolio better with passenger demographics\n")
                f.write("   - Introduce products tailored to major nationality groups\n")
                f.write("   - Adjust seasonal product mix based on traveler patterns\n\n")
            elif get_weakest_category(location_data) == "Cat_D":
                f.write("   - Implement best practices from cluster benchmark locations\n")
                f.write("   - Align format distribution with regional standards\n")
                f.write("   - Adopt successful portfolio structures from similar locations\n\n")
            else:
                f.write("   - Address volume growth trends through targeted promotions\n")
                f.write("   - Improve premium mix to enhance margins\n")
                f.write("   - Optimize SKU productivity to maximize performance\n\n")

            f.write("2. **SKU Rationalization**: Phase out chronically underperforming SKUs like ")
            if len(declining_skus) > 0:
                f.write(f"{declining_skus.iloc[0]['SKU']} ({declining_skus.iloc[0]['Growth']:.1%} growth) ")
                f.write("to focus resources on better-performing products.\n\n")
            else:
                f.write("low-volume variants to concentrate resources on core offerings.\n\n")

            f.write("3. **Portfolio Rebalancing**: Increase emphasis on successful brands like ")
            f.write(f"{brand_mix.iloc[0]['Brand_Family']} while reducing complexity in underperforming segments.\n\n")

            f.write(
                "4. **Competitive Positioning**: Address the gap in Category Segments by introducing products that ")
            f.write("better compete with local competitor offerings.\n\n")

            f.write("5. **Implement Cluster Learnings**: Study and adopt strategies from successful locations in ")
            f.write(f"Cluster {int(location_data['Cluster'])} to improve overall performance.\n\n")

        # Conclusion
        f.write("## Conclusion\n\n")

        if performance_level == "HIGH PERFORMER":
            f.write(
                f"{location_name} represents a strong performing location in our portfolio with an average score of {location_data['Avg_Score']:.2f}. ")
            f.write(
                f"The location demonstrates particular strength in {get_strongest_category(location_data)} ({get_strongest_score(location_data):.1f}), ")
            f.write(f"while maintaining balanced performance across all categories. ")
            f.write(
                "With continued focus on premium offerings and strategic SKU optimization, this location is well-positioned ")
            f.write("to maintain its leadership position and drive continued growth for the portfolio.\n")
        else:
            f.write(
                f"{location_name} presents significant optimization opportunities with an average score of {location_data['Avg_Score']:.2f}. ")
            f.write(
                f"By addressing the substantial gap in {get_weakest_category(location_data)} ({get_weakest_score(location_data):.1f}), ")
            f.write("this location has the potential to significantly improve its overall performance. ")
            f.write("A targeted approach focusing on portfolio rationalization, competitive positioning, and ")
            f.write(
                "implementation of cluster best practices will be essential to drive improvement in the coming period.\n")

    print(f"Detailed report for {location_name} saved as '{location_name}_portfolio_report.md'")

    # Helper functions for the report generation
def get_strongest_category(location_data):
    """Get the category with the highest score"""
    categories = ['Cat_A', 'Cat_B', 'Cat_C', 'Cat_D']
    scores = [location_data[cat] for cat in categories]
    highest_idx = scores.index(max(scores))

    category_names = {
        'Cat_A': 'PMI Performance',
        'Cat_B': 'Category Segments',
        'Cat_C': 'Passenger Mix',
        'Cat_D': 'Location Clusters'
    }

    return category_names[categories[highest_idx]]

def get_strongest_score(location_data):
    """Get the highest category score"""
    categories = ['Cat_A', 'Cat_B', 'Cat_C', 'Cat_D']
    scores = [location_data[cat] for cat in categories]
    return max(scores)

def get_weakest_category(location_data):
    """Get the category with the lowest score"""
    categories = ['Cat_A', 'Cat_B', 'Cat_C', 'Cat_D']
    scores = [location_data[cat] for cat in categories]
    lowest_idx = scores.index(min(scores))

    category_names = {
        'Cat_A': 'PMI Performance',
        'Cat_B': 'Category Segments',
        'Cat_C': 'Passenger Mix',
        'Cat_D': 'Location Clusters'
    }

    return category_names[categories[lowest_idx]]

def get_weakest_score(location_data):
    """Get the lowest category score"""
    categories = ['Cat_A', 'Cat_B', 'Cat_C', 'Cat_D']
    scores = [location_data[cat] for cat in categories]
    return min(scores)


def main():
    """
    Main function to run the portfolio optimization analysis for selected locations
    """
    print("Loading location data...")
    location_data, dubai_skus, prague_skus, dubai_context, prague_context = load_data()

    # Get the data for Dubai
    print("Analyzing Dubai...")
    dubai = extract_location_data(location_data, 'Dubai')

    # Get the data for Prague
    print("Analyzing Prague...")
    prague = extract_location_data(location_data, 'Prague')

    # Create visualizations
    print("Creating visualizations...")
    create_location_visualization('Dubai', dubai, dubai_skus, dubai_context)
    create_location_visualization('Prague', prague, prague_skus, prague_context)

    # Generate detailed reports
    print("Generating detailed reports...")
    generate_location_report('Dubai', dubai, dubai_skus, dubai_context)
    generate_location_report('Prague', prague, prague_skus, prague_context)

    print("Analysis complete!")


if __name__ == "__main__":
    main()