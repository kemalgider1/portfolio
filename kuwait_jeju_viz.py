"""
Kuwait and Jeju Portfolio Visualization Script
This script creates hexagonal visualizations for Kuwait and Jeju portfolios,
similar to the duo.py approach but with specific focus on these two locations.
"""

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

def load_data():
    """Load location data and create simulated SKU data for Kuwait and Jeju"""
    # Main location data with category scores - use the most recent scores
    location_data = pd.read_csv('results/all_scores_20250228_084055.csv')

    # Get Kuwait and Jeju data
    kuwait_data = location_data[location_data['Location'] == 'Kuwait'].iloc[0]
    jeju_data = location_data[location_data['Location'] == 'Jeju'].iloc[0]
    
    print(f"Kuwait score: {kuwait_data['Avg_Score']:.2f}")
    print(f"Jeju score: {jeju_data['Avg_Score']:.2f}")

    # Create simulated SKU data for Kuwait (high performer)
    brand_families_kuwait = ['MARLBORO', 'MARLBORO', 'PARLIAMENT', 'PARLIAMENT', 'HEETS',
                          'HEETS', 'L&M', 'L&M', 'CHESTERFIELD', 'CHESTERFIELD',
                          'MARLBORO', 'PARLIAMENT', 'HEETS', 'L&M', 'CHESTERFIELD',
                          'MARLBORO', 'PARLIAMENT', 'HEETS', 'L&M', 'CHESTERFIELD']
    
    # Create more descriptive SKU names based on brand family
    kuwait_sku_names = []
    brand_counters = {}
    
    for brand in brand_families_kuwait:
        if brand not in brand_counters:
            brand_counters[brand] = 1
        else:
            brand_counters[brand] += 1
    
        kuwait_sku_names.append(f"{brand}_KUW{brand_counters[brand]}")
    
    kuwait_skus = pd.DataFrame({
        'SKU': kuwait_sku_names,
        'Brand_Family': brand_families_kuwait,
        'Volume_2023': np.random.randint(300000, 900000, 20),
        'Volume_2024': np.random.randint(320000, 950000, 20),
        'Growth': np.random.uniform(0.02, 0.35, 20),
        'Margin': np.random.uniform(0.70, 0.90, 20),
        'Flavor': np.random.choice(['Regular', 'Menthol', 'Flavor Plus'], 20),
        'Strength': np.random.choice(['Full Flavor', 'Lights', 'Ultra Lights'], 20),
        'Length': np.random.choice(['KS', 'Super Slims', '100s'], 20),
        'TMO': 'PMI'
    })
    
    # Create simulated SKU data for Jeju (low performer)
    brand_families_jeju = ['MARLBORO', 'MARLBORO', 'PARLIAMENT', 'L&M', 'L&M',
                           'CHESTERFIELD', 'CHESTERFIELD', 'LARK', 'LARK', 'BOND',
                           'MARLBORO', 'PARLIAMENT', 'L&M', 'LARK', 'BOND']
    
    # Create more descriptive SKU names based on brand family
    jeju_sku_names = []
    brand_counters = {}
    
    for brand in brand_families_jeju:
        if brand not in brand_counters:
            brand_counters[brand] = 1
        else:
            brand_counters[brand] += 1
    
        jeju_sku_names.append(f"{brand}_JEJ{brand_counters[brand]}")
    
    jeju_skus = pd.DataFrame({
        'SKU': jeju_sku_names,
        'Brand_Family': brand_families_jeju,
        'Volume_2023': np.random.randint(200000, 600000, 15),
        'Volume_2024': np.random.randint(180000, 550000, 15),
        'Growth': np.random.uniform(-0.15, 0.10, 15),
        'Margin': np.random.uniform(0.65, 0.82, 15),
        'Flavor': np.random.choice(['Regular', 'Menthol'], 15),
        'Strength': np.random.choice(['Full Flavor', 'Lights'], 15),
        'Length': np.random.choice(['KS', '100s'], 15),
        'TMO': 'PMI'
    })
    
    # Add additional metrics for Kuwait context
    kuwait_context = {
        'Total_SKUs': 78,
        'PMI_SKUs': 20,
        'Comp_SKUs': 58,
        'Total_Volume': 79376600,
        'PMI_Volume': 38100768,
        'Market_Share': 0.48,
        'Green_Count': 6,
        'Red_Count': 1,
        'PAX_Annual': 58122178,
        'Category_A_Components': {
            'PMI_Performance': 0.80,
            'Volume_Growth': 0.75,
            'High_Margin_SKUs': 7,
            'Premium_Mix': 0.83
        },
        'Category_B_Components': {
            'Segment_Coverage': 0.95,
            'Competitive_Position': 0.97,
            'Premium_Ratio': 0.98,
            'Innovation_Score': 0.93
        },
        'Category_C_Components': {
            'PAX_Alignment': 0.92,
            'Nationality_Mix': 0.92,
            'Traveler_Type': 0.90,
            'Seasonal_Adjustment': 0.95
        },
        'Category_D_Components': {
            'Cluster_Similarity': 0.95,
            'Regional_Alignment': 0.98,
            'Size_Compatibility': 0.92,
            'Format_Distribution': 0.97
        }
    }
    
    # Add additional metrics for Jeju context
    jeju_context = {
        'Total_SKUs': 57,
        'PMI_SKUs': 15,
        'Comp_SKUs': 42,
        'Total_Volume': 44479675,
        'PMI_Volume': 27117301,
        'Market_Share': 0.61,
        'Green_Count': 0,
        'Red_Count': 8,
        'PAX_Annual': 32364558,
        'Category_A_Components': {
            'PMI_Performance': 0.30,
            'Volume_Growth': -0.05,
            'High_Margin_SKUs': 0,
            'Premium_Mix': 0.25
        },
        'Category_B_Components': {
            'Segment_Coverage': 0.10,
            'Competitive_Position': 0.12,
            'Premium_Ratio': 0.07,
            'Innovation_Score': 0.05
        },
        'Category_C_Components': {
            'PAX_Alignment': 0.06,
            'Nationality_Mix': 0.05,
            'Traveler_Type': 0.07,
            'Seasonal_Adjustment': 0.08
        },
        'Category_D_Components': {
            'Cluster_Similarity': 0.03,
            'Regional_Alignment': 0.04,
            'Size_Compatibility': 0.05,
            'Format_Distribution': 0.06
        }
    }
    
    return location_data, kuwait_data, jeju_data, kuwait_skus, jeju_skus, kuwait_context, jeju_context

def create_hexagon_visualization(location_name, location_data, sku_data, context_data):
    """
    Create a hexagonal visualization for a single location with improved styling
    """
    # Determine styling based on performance level
    avg_score = location_data['Avg_Score']
    if avg_score >= 7.0:
        performance_level = "HIGH PERFORMER"
        title_color = "#009900"  # Green
    elif avg_score >= 4.0:
        performance_level = "MODERATE PERFORMER"
        title_color = "#FFA500"  # Orange
    else:
        performance_level = "REQUIRES OPTIMIZATION"
        title_color = "#CC0000"  # Red

    # Create figure with dark background
    plt.figure(figsize=(16, 12), facecolor='#0a0a0a')
    plt.style.use('dark_background')
    gs = GridSpec(3, 3, figure=plt.gcf())

    # Create the main hexagonal plot
    ax_main = plt.subplot(gs[0:2, 0:2])
    ax_main.set_aspect('equal')
    ax_main.set_facecolor('#0a0a0a')

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
                        closed=True, fill=False, edgecolor='#cccccc',
                        linewidth=2, zorder=1)
    ax_main.add_patch(outer_hex)

    # Draw the inner hexagon (center point)
    inner_hex = Polygon(np.column_stack([hex_inner_x, hex_inner_y]),
                        closed=True, fill=True, facecolor='#444444',
                        edgecolor='#cccccc', linewidth=1.5, zorder=1)
    ax_main.add_patch(inner_hex)

    # Draw axis lines connecting inner and outer hexagons
    for i in range(n_sides):
        ax_main.plot([hex_inner_x[i], hex_outer_x[i]],
                     [hex_inner_y[i], hex_outer_y[i]],
                     color='#cccccc', linestyle='-', linewidth=1.5,
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

    # Define score values based on actual data
    score_values = [
        location_data['Cat_A'],  # Cat_A
        location_data['Cat_B'],  # Cat_B
        location_data['Cat_C'],  # Cat_C
        location_data['Cat_D'],  # Cat_D
        context_data['Market_Share'] * 10,  # Convert to 0-10 scale
        max(0, (context_data['Category_A_Components']['Volume_Growth'] + 0.15) * 10)  # Growth to 0-10 scale
    ]
    
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

        # Plot point with glow effect
        ax_main.scatter(x, y, s=150, color=category_colors[category],
                        edgecolor='white', linewidth=1.5, zorder=3)

        # Add score label with improved visibility
        label_radius = radius + 1.0
        label_x = label_radius * np.cos(angle)
        label_y = label_radius * np.sin(angle)

        # Add category label at the outer edge
        category_radius = radius_outer + 1.8
        cat_x = category_radius * np.cos(angle)
        cat_y = category_radius * np.sin(angle)

        # Full category name with better visibility
        category_labels = {
            'Cat_A': 'Category A',
            'Cat_B': 'Category B',
            'Cat_C': 'Category C',
            'Cat_D': 'Category D',
            'MS': 'Market Share',
            'Growth': 'Volume Growth'
        }
        
        cat_text = ax_main.text(cat_x, cat_y, category_labels[category],
                                ha='center', va='center', fontsize=14,
                                fontweight='bold', color=category_colors[category])
        cat_text.set_path_effects([path_effects.withStroke(linewidth=5, foreground='black')])

        # Add score value with glow
        score_text = ax_main.text(label_x, label_y, f'{score:.1f}',
                                  ha='center', va='center', fontsize=12,
                                  fontweight='bold', color=category_colors[category])
        score_text.set_path_effects([path_effects.withStroke(linewidth=4, foreground='black')])

    # Connect the score points to form a polygon
    score_x.append(score_x[0])
    score_y.append(score_y[0])
    ax_main.plot(score_x, score_y, '-', color='white', linewidth=2.5, alpha=0.8, zorder=2)

    # Fill the polygon with a translucent color
    score_polygon = Polygon(np.column_stack([score_x, score_y]),
                            closed=True, fill=True,
                            facecolor='white', alpha=0.15, zorder=1)
    ax_main.add_patch(score_polygon)

    # Add the average score in the center
    ax_main.text(0, 0, f'{avg_score:.1f}',
                 ha='center', va='center', fontsize=24,
                 fontweight='bold', color='white', zorder=3)

    # Add reference circles
    for r in [2.5, 5, 7.5]:
        circle = plt.Circle((0, 0), scale_value(r), fill=False, 
                           edgecolor='white', alpha=0.15, linewidth=0.5)
        ax_main.add_patch(circle)
        
        # Add subtle reference labels
        label_x = scale_value(r) * np.cos(np.pi/4)
        label_y = scale_value(r) * np.sin(np.pi/4)
        ax_main.text(label_x, label_y, f"{r}", fontsize=8, 
                    color='#999999', ha='center', va='center', alpha=0.7)

    ax_main.set_xlim(-radius_outer * 1.5, radius_outer * 1.5)
    ax_main.set_ylim(-radius_outer * 1.5, radius_outer * 1.5)
    ax_main.axis('off')

    # Add SKU Performance section (Top 5 SKUs)
    ax_skus = plt.subplot(gs[0, 2])
    ax_skus.set_facecolor('#0a0a0a')

    # Sort SKUs by volume and get top 5
    top_skus = sku_data.sort_values('Volume_2024', ascending=False).head(5)

    # Shorten SKU names for display
    top_skus = top_skus.copy()
    top_skus['SKU_Display'] = top_skus['SKU'].apply(lambda x: x.split('_')[0] + '...')

    # Create a horizontal bar chart for SKU volumes
    colors = [plt.cm.viridis(x / 5) for x in range(5)]
    bars = ax_skus.barh(top_skus['SKU_Display'], top_skus['Volume_2024'], color=colors, height=0.6)

    # Add growth indicators with better visibility
    for i, (_, row) in enumerate(top_skus.iterrows()):
        growth = row['Growth']
        max_volume = top_skus['Volume_2024'].max()
        # Position the growth indicator at the end of the bar plus a small margin
        if growth > 0:
            growth_text = ax_skus.text(row['Volume_2024'] + (max_volume * 0.05), i, f'↑ {abs(growth):.1%}',
                         va='center', fontsize=10, color='#00cc00', fontweight='bold')
            growth_text.set_path_effects([path_effects.withStroke(linewidth=2, foreground='black')])
        else:
            growth_text = ax_skus.text(row['Volume_2024'] + (max_volume * 0.05), i, f'↓ {abs(growth):.1%}',
                         va='center', fontsize=10, color='#cc0000', fontweight='bold')
            growth_text.set_path_effects([path_effects.withStroke(linewidth=2, foreground='black')])

    ax_skus.set_title('Top 5 SKUs by Volume', fontsize=16, fontweight='bold', color='white')
    ax_skus.spines['top'].set_visible(False)
    ax_skus.spines['right'].set_visible(False)
    ax_skus.set_xlabel('Volume (2024)', fontsize=12, color='white')
    ax_skus.tick_params(colors='white')

    # Add Brand Mix section
    ax_brands = plt.subplot(gs[1, 2])
    ax_brands.set_facecolor('#0a0a0a')

    # Calculate brand mix from SKU data
    brand_mix = sku_data.groupby('Brand_Family')['Volume_2024'].sum().reset_index()
    brand_mix = brand_mix.sort_values('Volume_2024', ascending=True).tail(5)  # Get top 5 brands

    # Create a horizontal bar chart for brand volumes with improved colors
    colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(brand_mix)))
    ax_brands.barh(brand_mix['Brand_Family'], brand_mix['Volume_2024'], color=colors, height=0.6)

    ax_brands.set_title('Brand Family Mix', fontsize=16, fontweight='bold', color='white')
    ax_brands.spines['top'].set_visible(False)
    ax_brands.spines['right'].set_visible(False)
    ax_brands.set_xlabel('Volume (2024)', fontsize=12, color='white')
    ax_brands.tick_params(colors='white')

    # Add Key Metrics section with component boxes
    component_box = plt.subplot(gs[2, :])
    component_box.set_facecolor('#0a0a0a')
    component_box.axis('off')

    # Define category titles and bgcolors
    categories = [
        {"title": "Category A Components", "subtitle": "(PMI Performance)", "color": "#1f77b422"},
        {"title": "Category B Components", "subtitle": "(Category Segments)", "color": "#ff7f0e22"},
        {"title": "Category C Components", "subtitle": "(Passenger Mix)", "color": "#2ca02c22"},
        {"title": "Category D Components", "subtitle": "(Location Clusters)", "color": "#d6272822"}
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

    # Draw metric section with improved visibility
    # First row
    for i, metric in enumerate(metrics[0]):
        x = 0.05 + i * 0.23
        y = 0.80
        rect = FancyBboxPatch((x, y), 0.21, 0.06, boxstyle="round,pad=0.03",
                              facecolor='#222222', edgecolor='#666666', alpha=0.9)
        component_box.add_patch(rect)
        component_box.text(x + 0.105, y + 0.03, metric,
                           ha='center', va='center', fontsize=11, color='white')

    # Second row
    for i, metric in enumerate(metrics[1]):
        x = 0.05 + i * 0.23
        y = 0.72
        rect = FancyBboxPatch((x, y), 0.21, 0.06, boxstyle="round,pad=0.03",
                              facecolor='#222222', edgecolor='#666666', alpha=0.9)
        component_box.add_patch(rect)
        component_box.text(x + 0.105, y + 0.03, metric,
                           ha='center', va='center', fontsize=11, color='white')

    # Component data from context
    component_data = [
        context_data['Category_A_Components'],
        context_data['Category_B_Components'],
        context_data['Category_C_Components'],
        context_data['Category_D_Components']
    ]

    # Draw component boxes with improved spacing
    for i, (category, data) in enumerate(zip(categories, component_data)):
        # Position boxes in a row with more spacing
        x = 0.05 + i * 0.24
        y = 0.08
        box_width = 0.22
        box_height = 0.60

        # Create box with colored background
        rect = FancyBboxPatch((x, y), box_width, box_height,
                              boxstyle="round,pad=0.03",
                              facecolor=category["color"],
                              edgecolor='#666666', alpha=0.9)
        component_box.add_patch(rect)

        # Add title
        component_box.text(x + box_width / 2, y + box_height - 0.05,
                           category["title"],
                           ha='center', va='center',
                           fontsize=14, fontweight='bold', color='white')

        # Add subtitle
        component_box.text(x + box_width / 2, y + box_height - 0.11,
                           category["subtitle"],
                           ha='center', va='center', fontsize=11, color='#cccccc')

        # Add component metrics with more space
        y_pos = y + box_height - 0.18
        for key, value in data.items():
            y_pos -= 0.08
            # Format key name for display
            key_display = key.replace('_', ' ').title()

            # Add key name
            component_box.text(x + 0.02, y_pos, key_display,
                               ha='left', va='center', fontsize=10, color='white')

            # Add score bar with improved visibility
            bar_length = box_width * 0.6 * value
            bar_height = 0.015
            
            # Background track
            rect = Rectangle((x + 0.1, y_pos - bar_height / 2),
                             box_width * 0.6, bar_height,
                             facecolor='#333333', alpha=0.6)
            component_box.add_patch(rect)
            
            # Value bar with color based on value
            if value >= 0.75:
                bar_color = '#2ca02c'  # Green for high values
            elif value >= 0.25:
                bar_color = '#ff7f0e'  # Orange for medium values
            else:
                bar_color = '#d62728'  # Red for low values
                
            rect = Rectangle((x + 0.1, y_pos - bar_height / 2),
                             bar_length, bar_height,
                             facecolor=bar_color, alpha=0.8)
            component_box.add_patch(rect)

            # Add value
            component_box.text(x + box_width - 0.02, y_pos, f"{value:.2f}",
                               ha='right', va='center', fontsize=10, 
                               fontweight='bold', color='white')

    # Add location name and title
    plt.suptitle(f'PORTFOLIO OPTIMIZATION ANALYSIS: {location_name.upper()}',
                 fontsize=24, y=0.98, fontweight='bold', color='white')

    # Add classification banner
    plt.figtext(0.5, 0.94, performance_level,
                fontsize=16, ha='center', color='white',
                bbox=dict(facecolor=title_color, alpha=0.9, boxstyle='round,pad=0.5'))

    # Add timestamp and footer
    plt.figtext(0.5, 0.01, f"Generated on {pd.Timestamp.now().strftime('%Y-%m-%d')} | Score: {avg_score:.2f} / 10",
                fontsize=10, ha='center', color='#999999')

    # Save the figure with custom filename and quality settings
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    filename = f"{location_name.replace(' ', '_')}_hexagon_portfolio.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='#0a0a0a')
    plt.close()
    
    print(f"Created hexagon visualization for {location_name}: {filename}")
    return filename


if __name__ == "__main__":
    # Load the data
    location_data, kuwait_data, jeju_data, kuwait_skus, jeju_skus, kuwait_context, jeju_context = load_data()
    
    # Create hexagonal visualizations specifically for Kuwait and Jeju
    kuwait_viz = create_hexagon_visualization('Kuwait', kuwait_data, kuwait_skus, kuwait_context)
    jeju_viz = create_hexagon_visualization('Jeju', jeju_data, jeju_skus, jeju_context)
    
    print("All visualizations completed successfully.")