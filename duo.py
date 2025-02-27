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
    location_data = pd.read_csv('results/clustered_locations_20250227_022315.csv')

    # Create simulated SKU data for each location (in practice, this would be loaded from actual files)
    # For Paris Charles De Gaulle

    # Updated code for Paris Charles De Gaulle
    brand_families_paris = ['MARLBORO', 'MARLBORO', 'PARLIAMENT', 'PARLIAMENT', 'HEETS',
                            'HEETS', 'L&M', 'L&M', 'CHESTERFIELD', 'CHESTERFIELD',
                            'MARLBORO', 'PARLIAMENT', 'HEETS', 'L&M', 'CHESTERFIELD',
                            'MARLBORO', 'PARLIAMENT', 'HEETS', 'L&M', 'CHESTERFIELD',
                            'MARLBORO', 'PARLIAMENT', 'HEETS', 'L&M', 'CHESTERFIELD']

    # Create more descriptive SKU names based on brand family
    paris_sku_names = []
    brand_counters = {}

    for brand in brand_families_paris:
        if brand not in brand_counters:
            brand_counters[brand] = 1
        else:
            brand_counters[brand] += 1

        paris_sku_names.append(f"{brand}_CDG{brand_counters[brand]}")

    paris_skus = pd.DataFrame({
        'SKU': paris_sku_names,
        'Brand_Family': brand_families_paris,
        # Copy the rest of the attributes as they were in the original
        'Volume_2023': np.random.randint(300000, 900000, 25),
        'Volume_2024': np.random.randint(320000, 950000, 25),
        'Growth': np.random.uniform(0.02, 0.35, 25),
        'Margin': np.random.uniform(0.70, 0.90, 25),
        'Flavor': np.random.choice(['Regular', 'Menthol', 'Flavor Plus'], 25),
        'Strength': np.random.choice(['Full Flavor', 'Lights', 'Ultra Lights'], 25),
        'Length': np.random.choice(['KS', 'Super Slims', '100s'], 25),
        'TMO': 'PMI'
    })

    # Updated code for Shanghai Hongqiao
    brand_families_shanghai = ['MARLBORO', 'MARLBORO', 'PARLIAMENT', 'L&M', 'L&M',
                               'CHESTERFIELD', 'CHESTERFIELD', 'LARK', 'LARK', 'BOND',
                               'MARLBORO', 'PARLIAMENT', 'L&M', 'LARK', 'BOND',
                               'LARK', 'BOND']

    # Create more descriptive SKU names based on brand family
    shanghai_sku_names = []
    brand_counters = {}

    for brand in brand_families_shanghai:
        if brand not in brand_counters:
            brand_counters[brand] = 1
        else:
            brand_counters[brand] += 1

        shanghai_sku_names.append(f"{brand}_SHA{brand_counters[brand]}")

    shanghai_skus = pd.DataFrame({
        'SKU': shanghai_sku_names,
        'Brand_Family': brand_families_shanghai,
        # Copy the rest of the attributes as they were in the original
        'Volume_2023': np.random.randint(200000, 600000, 17),
        'Volume_2024': np.random.randint(180000, 550000, 17),
        'Growth': np.random.uniform(-0.15, 0.10, 17),
        'Margin': np.random.uniform(0.65, 0.82, 17),
        'Flavor': np.random.choice(['Regular', 'Menthol'], 17),
        'Strength': np.random.choice(['Full Flavor', 'Lights'], 17),
        'Length': np.random.choice(['KS', '100s'], 17),
        'TMO': 'PMI'
    })

    # Add additional metrics for location context
    paris_context = {
        'Total_SKUs': 78,
        'PMI_SKUs': 25,
        'Comp_SKUs': 53,
        'Total_Volume': 79376600,
        'PMI_Volume': 38100768,
        'Market_Share': 0.48,
        'Green_Count': 4,
        'Red_Count': 2,
        'PAX_Annual': 58122178,
        'Category_A_Components': {
            'PMI_Performance': 0.67,
            'Volume_Growth': 0.32,
            'High_Margin_SKUs': 4,
            'Premium_Mix': 0.70
        },
        'Category_B_Components': {
            'Segment_Coverage': 1.00,
            'Competitive_Position': 1.00,
            'Premium_Ratio': 1.00,
            'Innovation_Score': 1.00
        },
        'Category_C_Components': {
            'PAX_Alignment': 1.00,
            'Nationality_Mix': 1.00,
            'Traveler_Type': 0.98,
            'Seasonal_Adjustment': 1.00
        },
        'Category_D_Components': {
            'Cluster_Similarity': 1.00,
            'Regional_Alignment': 1.00,
            'Size_Compatibility': 1.00,
            'Format_Distribution': 1.00
        }
    }

    shanghai_context = {
        'Total_SKUs': 57,
        'PMI_SKUs': 17,
        'Comp_SKUs': 40,
        'Total_Volume': 44479675,
        'PMI_Volume': 27117301,
        'Market_Share': 0.61,
        'Green_Count': 0,
        'Red_Count': 8,
        'PAX_Annual': 32364558,
        'Category_A_Components': {
            'PMI_Performance': 0.61,
            'Volume_Growth': -0.05,
            'High_Margin_SKUs': 0,
            'Premium_Mix': 0.59
        },
        'Category_B_Components': {
            'Segment_Coverage': 0.66,
            'Competitive_Position': 0.66,
            'Premium_Ratio': 0.66,
            'Innovation_Score': 0.65
        },
        'Category_C_Components': {
            'PAX_Alignment': 0.00,
            'Nationality_Mix': 0.00,
            'Traveler_Type': 0.00,
            'Seasonal_Adjustment': 0.00
        },
        'Category_D_Components': {
            'Cluster_Similarity': 0.07,
            'Regional_Alignment': 0.07,
            'Size_Compatibility': 0.08,
            'Format_Distribution': 0.50
        }
    }

    return location_data, paris_skus, shanghai_skus, paris_context, shanghai_context

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
    # Determine performance category based on actual scores
    if location_name == "Paris Charles De Gaulle":
        performance_level = "HIGH PERFORMER"
        title_color = "#009900"  # Green
    else:  # Shanghai Hongqiao
        performance_level = "REQUIRES OPTIMIZATION"
        title_color = "#CC0000"  # Red

    # Check if actual data is available from location_data - only used for validation
    # In normal production use, the default scores are used above

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

    # Define category colors
    category_colors = {
        'Cat_A': '#1f77b4',  # Blue
        'Cat_B': '#ff7f0e',  # Orange
        'Cat_C': '#2ca02c',  # Green
        'Cat_D': '#d62728',  # Red
        'MS': '#9467bd',  # Purple
        'Growth': '#8c564b'  # Brown
    }

    # Define score values based on actual data from Project Knowledge
    if location_name == "Paris Charles De Gaulle":
        # Category scores from Project Knowledge
        score_values = [
            6.67,  # Cat_A
            10.0,  # Cat_B
            9.98,  # Cat_C
            10.0,  # Cat_D
            context_data['Market_Share'] * 10,  # Convert to 0-10 scale
            0.32 * 10  # Growth converted to 0-10 scale
        ]
        avg_score = 9.16  # Average of the four category scores
    else:  # Shanghai Hongqiao
        # Category scores from Project Knowledge
        score_values = [
            6.10,  # Cat_A
            6.63,  # Cat_B
            0.0,  # Cat_C
            0.72,  # Cat_D
            context_data['Market_Share'] * 10,  # Convert to 0-10 scale
            -0.05 * 10  # Growth converted to 0-10 scale (negative)
        ]
        avg_score = 3.36  # Average of the four category scores

    # Create the category points - order: Cat_A, Cat_B, Cat_C, Cat_D, Market_Share, Growth
    categories = ['Cat_A', 'Cat_B', 'Cat_C', 'Cat_D', 'MS', 'Growth']

    # Plot score points and value labels
    score_x = []
    score_y = []

    for i, (angle, score, category) in enumerate(zip(angles, score_values, categories)):
        # Calculate position
        radius = scale_value(score)
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        score_x.append(x)
        score_y.append(y)

        # Plot point
        ax_main.scatter(x, y, s=120, color=category_colors[category],
                        edgecolor='white', linewidth=1.5, zorder=3)

        # Add score label
        label_radius = radius + 0.8
        label_x = label_radius * np.cos(angle)
        label_y = label_radius * np.sin(angle)

        # Add category label at the outer edge
        category_radius = radius_outer + 1.5
        cat_x = category_radius * np.cos(angle)
        cat_y = category_radius * np.sin(angle)

        cat_text = ax_main.text(cat_x, cat_y, category,
                                ha='center', va='center', fontsize=14,
                                fontweight='bold', color=category_colors[category])
        cat_text.set_path_effects([path_effects.withStroke(linewidth=5, foreground='white')])

        # Add score value
        score_text = ax_main.text(label_x, label_y, f'{score:.1f}',
                                  ha='center', va='center', fontsize=12,
                                  fontweight='bold', color=category_colors[category])
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
    ax_main.text(0, 0, f'{avg_score:.1f}',
                 ha='center', va='center', fontsize=18,
                 fontweight='bold', color='white', zorder=3)

    ax_main.set_xlim(-radius_outer * 1.4, radius_outer * 1.4)
    ax_main.set_ylim(-radius_outer * 1.4, radius_outer * 1.4)
    ax_main.axis('off')

    # Add SKU Performance section (Top 5 SKUs)
    ax_skus = plt.subplot(gs[0, 2])

    # Sort SKUs by volume and get top 5
    top_skus = sku_data.sort_values('Volume_2024', ascending=False).head(5)

    # Shorten SKU names for display
    top_skus = top_skus.copy()
    top_skus['SKU_Display'] = top_skus['SKU'].apply(lambda x: x[:8] + '...')

    # Create a horizontal bar chart for SKU volumes
    colors = [plt.cm.viridis(x / 10) for x in range(5)]
    bars = ax_skus.barh(top_skus['SKU_Display'], top_skus['Volume_2024'], color=colors, height=0.6)

    # Add growth indicators
    for i, (_, row) in enumerate(top_skus.iterrows()):
        growth = row['Growth']
        max_volume = top_skus['Volume_2024'].max()
        # Position the growth indicator at the end of the bar plus a small margin
        if growth > 0:
            ax_skus.text(row['Volume_2024'] + (max_volume * 0.05), i, f'↑ {abs(growth):.1%}',
                         va='center', fontsize=10, color='green', fontweight='bold')
        else:
            ax_skus.text(row['Volume_2024'] + (max_volume * 0.05), i, f'↓ {abs(growth):.1%}',
                         va='center', fontsize=10, color='red', fontweight='bold')

    ax_skus.set_title('Top 5 SKUs by Volume', fontsize=14, fontweight='bold')
    ax_skus.spines['top'].set_visible(False)
    ax_skus.spines['right'].set_visible(False)
    ax_skus.set_xlabel('Volume (2024)', fontsize=10)

    # Add Brand Mix section
    ax_brands = plt.subplot(gs[1, 2])

    # Calculate brand mix from SKU data
    brand_mix = sku_data.groupby('Brand_Family')['Volume_2024'].sum().reset_index()
    brand_mix = brand_mix.sort_values('Volume_2024', ascending=True).tail(5)  # Get top 5 brands

    # Create a horizontal bar chart for brand volumes
    colors = plt.cm.Oranges(np.linspace(0.4, 0.8, len(brand_mix)))
    ax_brands.barh(brand_mix['Brand_Family'], brand_mix['Volume_2024'], color=colors, height=0.6)

    ax_brands.set_title('Brand Family Mix', fontsize=14, fontweight='bold')
    ax_brands.spines['top'].set_visible(False)
    ax_brands.spines['right'].set_visible(False)
    ax_brands.set_xlabel('Volume (2024)', fontsize=10)

    # Add Key Metrics section with 4 category component boxes
    component_box = plt.subplot(gs[2, :])
    component_box.axis('off')

    # Define category titles and bgcolors
    categories = [
        {"title": "Category A Components", "subtitle": "(PMI Performance)", "color": "#E6F0FF"},
        {"title": "Category B Components", "subtitle": "(Category Segments)", "color": "#FFF3E6"},
        {"title": "Category C Components", "subtitle": "(Passenger Mix)", "color": "#E6FFE6"},
        {"title": "Category D Components", "subtitle": "(Location Clusters)", "color": "#FFE6E6"}
    ]

    # Create the header row with metrics
    metrics = [
        [f"Total SKUs: {context_data['Total_SKUs']}",
         f"PMI SKUs: {context_data['PMI_SKUs']}",
         f"Competitor SKUs: {context_data['Comp_SKUs']}",
         f"Market Share: {context_data['Market_Share'] * 100:.1f}%"],
        [f"Total Volume: {context_data['Total_Volume']:,}",
         f"Green SKUs: {context_data['Green_Count']}",
         f"Red SKUs: {context_data['Red_Count']}",
         f"Annual PAX: {context_data['PAX_Annual']:,}"]
    ]

    # Draw metric section
    # First row
    for i, metric in enumerate(metrics[0]):
        x = 0.05 + i * 0.23
        y = 0.80
        rect = FancyBboxPatch((x, y), 0.21, 0.05, boxstyle="round,pad=0.03",
                              facecolor='white', edgecolor='#333333', alpha=0.9)
        component_box.add_patch(rect)
        component_box.text(x + 0.105, y + 0.025, metric,
                           ha='center', va='center', fontsize=11)

    # Second row
    for i, metric in enumerate(metrics[1]):
        x = 0.05 + i * 0.23
        y = 0.74
        rect = FancyBboxPatch((x, y), 0.21, 0.05, boxstyle="round,pad=0.03",
                              facecolor='white', edgecolor='#333333', alpha=0.9)
        component_box.add_patch(rect)
        component_box.text(x + 0.105, y + 0.025, metric,
                           ha='center', va='center', fontsize=11)

    # Component data from context
    component_data = [
        context_data['Category_A_Components'],
        context_data['Category_B_Components'],
        context_data['Category_C_Components'],
        context_data['Category_D_Components']
    ]

    # Draw component boxes
    for i, (category, data) in enumerate(zip(categories, component_data)):
        # Position boxes in a row
        x = 0.05 + i * 0.23
        y = 0.10
        box_width = 0.21
        box_height = 0.60

        # Create box with colored background
        rect = FancyBboxPatch((x, y), box_width, box_height,
                              boxstyle="round,pad=0.03",
                              facecolor=category["color"],
                              edgecolor='#666666', alpha=0.8)
        component_box.add_patch(rect)

        # Add title
        component_box.text(x + box_width / 2, y + box_height - 0.05,
                           category["title"],
                           ha='center', va='center',
                           fontsize=12, fontweight='bold')

        # Add subtitle
        component_box.text(x + box_width / 2, y + box_height - 0.10,
                           category["subtitle"],
                           ha='center', va='center', fontsize=10)

        # Add component metrics
        y_pos = y + box_height - 0.18
        for key, value in data.items():
            y_pos -= 0.07
            # Format key name for display
            key_display = key.replace('_', ' ')

            # Add key name
            component_box.text(x + 0.02, y_pos, key_display,
                               ha='left', va='center', fontsize=9)

            # Add score bar
            bar_length = box_width * 0.5 * value
            bar_height = 0.01
            rect = Rectangle((x + 0.08, y_pos - bar_height / 2),
                             bar_length, bar_height,
                             facecolor=plt.cm.RdYlGn(value), alpha=0.8)
            component_box.add_patch(rect)

            # Add value
            component_box.text(x + box_width - 0.02, y_pos, f"{value:.2f}",
                               ha='right', va='center', fontsize=9, fontweight='bold')

    # Add location name and title
    plt.suptitle(f'Portfolio Optimization Analysis: {location_name}',
                 fontsize=24, y=0.98, fontweight='bold')

    # Add classification banner
    plt.figtext(0.5, 0.94, performance_level,
                fontsize=16, ha='center', color='white',
                bbox=dict(facecolor=title_color, alpha=0.9, boxstyle='round,pad=0.5'))

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
    if location_name == "Paris Charles De Gaulle":
        performance_level = "HIGH PERFORMER"
        avg_score = 9.16  # Average of the four category scores
        cat_a = 6.67
        cat_b = 10.0
        cat_c = 9.98
        cat_d = 10.0
    else:  # Shanghai Hongqiao
        performance_level = "REQUIRES OPTIMIZATION"
        avg_score = 3.36  # Average of the four category scores
        cat_a = 6.10
        cat_b = 6.63
        cat_c = 0.0
        cat_d = 0.72

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
    # Load data
    location_data, paris_skus, shanghai_skus, paris_context, shanghai_context = load_data()

    try:
        # Extract location data if available
        paris_data = extract_location_data(location_data, "Paris Charles De Gaulle")
        shanghai_data = extract_location_data(location_data, "Shanghai Hongqiao")

        # Create visualizations
        create_location_visualization("Paris Charles De Gaulle", paris_data, paris_skus, paris_context)
        create_location_visualization("Shanghai Hongqiao", shanghai_data, shanghai_skus, shanghai_context)

        # Generate reports
        generate_location_report("Paris Charles De Gaulle", paris_data, paris_skus, paris_context)
        generate_location_report("Shanghai Hongqiao", shanghai_data, shanghai_skus, shanghai_context)

    except Exception as e:
        print(f"Error: {e}")
        # Use fallback for location data if not found in dataset
        print("Using fallback location data...")
        create_location_visualization("Paris Charles De Gaulle", None, paris_skus, paris_context)
        create_location_visualization("Shanghai Hongqiao", None, shanghai_skus, shanghai_context)
        generate_location_report("Paris Charles De Gaulle", None, paris_skus, paris_context)
        generate_location_report("Shanghai Hongqiao", None, shanghai_skus, shanghai_context)


if __name__ == "__main__":
    main()
