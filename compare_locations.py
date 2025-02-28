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
    """Load location data and create simulated SKU data"""
    # Main location data with category scores - use the most recent scores
    location_data = pd.read_csv('results/all_scores_20250228_084055.csv')

    # Find our two target locations - Kuwait and Jeju
    kuwait_data = location_data[location_data['Location'] == 'Kuwait'].iloc[0]
    jeju_data = location_data[location_data['Location'] == 'Jeju'].iloc[0]
    
    print(f"Kuwait score: {kuwait_data['Avg_Score']:.2f}")
    print(f"Jeju score: {jeju_data['Avg_Score']:.2f}")

    # Create simulated SKU data for Kuwait (high performer)
    brand_families_kuwait = ['MARLBORO', 'MARLBORO', 'PARLIAMENT', 'PARLIAMENT', 'HEETS',
                          'HEETS', 'L&M', 'L&M', 'CHESTERFIELD', 'CHESTERFIELD',
                          'MARLBORO', 'PARLIAMENT', 'HEETS', 'L&M', 'CHESTERFIELD',
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
        'Volume_2023': np.random.randint(300000, 900000, 25),
        'Volume_2024': np.random.randint(320000, 950000, 25),
        'Growth': np.random.uniform(0.02, 0.35, 25),
        'Margin': np.random.uniform(0.70, 0.90, 25),
        'Flavor': np.random.choice(['Regular', 'Menthol', 'Flavor Plus'], 25),
        'Strength': np.random.choice(['Full Flavor', 'Lights', 'Ultra Lights'], 25),
        'Length': np.random.choice(['KS', 'Super Slims', '100s'], 25),
        'TMO': 'PMI'
    })
    
    # Create simulated SKU data for Jeju (low performer)
    brand_families_jeju = ['MARLBORO', 'MARLBORO', 'PARLIAMENT', 'L&M', 'L&M',
                           'CHESTERFIELD', 'CHESTERFIELD', 'LARK', 'LARK', 'BOND',
                           'MARLBORO', 'PARLIAMENT', 'L&M', 'LARK', 'BOND',
                           'LARK', 'BOND']
    
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
        'Volume_2023': np.random.randint(200000, 600000, 17),
        'Volume_2024': np.random.randint(180000, 550000, 17),
        'Growth': np.random.uniform(-0.15, 0.10, 17),
        'Margin': np.random.uniform(0.65, 0.82, 17),
        'Flavor': np.random.choice(['Regular', 'Menthol'], 17),
        'Strength': np.random.choice(['Full Flavor', 'Lights'], 17),
        'Length': np.random.choice(['KS', '100s'], 17),
        'TMO': 'PMI'
    })
    
    # Add additional metrics for Kuwait context
    kuwait_context = {
        'Total_SKUs': 78,
        'PMI_SKUs': 25,
        'Comp_SKUs': 53,
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
    
    # Add additional metrics for Jeju context with similar total volume
    jeju_context = {
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

def create_individual_visualization(location_name, location_data, sku_data, context_data, dark_theme=True):
    """
    Create a detailed visualization for a single location with advanced formatting and improved spacing.
    
    Note: This function may produce some posx/posy warnings related to polar plots, 
    but the visualizations should still be created correctly.
    """
    # Determine if we're using dark or light theme
    if dark_theme:
        plt.style.use('dark_background')
        bg_color = '#1a1a1a'
        text_color = 'white'
        grid_color = '#333333'
        light_text = '#cccccc'
    else:
        plt.style.use('default')
        bg_color = '#f4f4f4'
        text_color = 'black'
        grid_color = '#dddddd'
        light_text = '#666666'
    
    # Set up the figure with a larger size to prevent crowding
    fig = plt.figure(figsize=(20, 15))
    fig.patch.set_facecolor(bg_color)
    
    # Create a grid layout with more spacing
    gs = GridSpec(3, 6, figure=fig, height_ratios=[1, 2.2, 1.2], hspace=0.4, wspace=0.4)
    
    # Header section with title
    ax_header = fig.add_subplot(gs[0, :])
    ax_header.set_facecolor(bg_color)
    ax_header.axis('off')
    
    # Title
    title_text = f"PORTFOLIO OPTIMIZATION ANALYSIS: {location_name.upper()}"
    subtitle_text = f"Score: {location_data['Avg_Score']:.2f} / 10.0"
    
    # Determine color theme based on score
    if location_data['Avg_Score'] >= 7.0:
        score_color = '#3498db'  # Blue for high performers
        performance = "HIGH PERFORMER"
    elif location_data['Avg_Score'] >= 4.0:
        score_color = '#f39c12'  # Orange for mid performers
        performance = "MODERATE PERFORMER"
    else:
        score_color = '#e74c3c'  # Red for low performers
        performance = "REQUIRES OPTIMIZATION"
    
    # Improved title positioning
    ax_header.text(0.5, 0.7, title_text, 
                  ha='center', va='center', fontsize=24, 
                  fontweight='bold', color=text_color, 
                  path_effects=[path_effects.withStroke(linewidth=3, foreground=bg_color)])
    
    # Add a background box for the classification for better visibility
    ax_header.text(0.5, 0.3, f"Classification: {performance}", 
                  ha='center', va='center', fontsize=18,
                  fontweight='bold', color=score_color,
                  bbox=dict(facecolor=bg_color, alpha=0.7, edgecolor=score_color, pad=5))
    
    ax_header.text(0.5, 0.1, subtitle_text, 
                  ha='center', va='center', fontsize=16,
                  color=light_text)
    
    # Category scores radar chart
    ax_categories = fig.add_subplot(gs[1, 0:3], polar=True)
    ax_categories.set_facecolor(bg_color)
    
    # Categories and scores
    categories = ['Category A', 'Category B', 'Category C', 'Category D', 'Market Share', 'Volume Growth']
    scores = [
        location_data['Cat_A'],
        location_data['Cat_B'],
        location_data['Cat_C'],
        location_data['Cat_D'],
        context_data['Market_Share'] * 10,  # Scale to 0-10
        max(0, (context_data['Category_A_Components']['Volume_Growth'] + 0.15) * 10)  # Adjust to 0-10 scale
    ]
    
    # Prepare the radar chart
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    scores += scores[:1]  # Close the loop
    
    # Configure the radar chart
    ax_categories.set_theta_offset(np.pi / 2)
    ax_categories.set_theta_direction(-1)
    
    # Improved label placement with increased fontsize
    ax_categories.set_thetagrids(np.degrees(angles[:-1]), categories, color=text_color, fontsize=12)
    
    # Plot the radar chart with gradient fill
    ax_categories.plot(angles, scores, 'o-', linewidth=3, color=score_color, alpha=0.9)
    
    # Create a gradient fill
    for i in range(len(angles)-1):
        ax_categories.fill_between([angles[i], angles[i+1]], 
                              [0, 0], 
                              [scores[i], scores[i+1]], 
                              color=score_color, alpha=0.1+0.05*i)
    
    # Add value labels at each point with improved positioning
    for i, (angle, score) in enumerate(zip(angles[:-1], scores[:-1])):
        # Use direct angle-based positioning to avoid NaN issues
        # For safety, ensure the score is valid
        valid_score = max(0.1, min(10, score))  # Clamp between 0.1 and 10
        
        # Fixed distance from center based on the score
        r_offset = 0.8 if valid_score > 8 else 0.6
        total_radius = valid_score + r_offset
        
        # Calculate label position
        label_x = total_radius * np.cos(angle)
        label_y = total_radius * np.sin(angle)
        
        # Draw with better visibility - only if positions are valid
        if np.isfinite(label_x) and np.isfinite(label_y):
            ax_categories.text(label_x, label_y, f"{score:.1f}", 
                              ha='center', va='center', fontsize=12, 
                              fontweight='bold', color=text_color,
                              bbox=dict(facecolor=bg_color, alpha=0.7, edgecolor=score_color, boxstyle='round,pad=0.2'))
    
    # Set limits and title
    ax_categories.set_ylim(0, 10.5)  # Extend limit to fit labels
    ax_categories.set_title('Category Score Distribution', fontsize=18, color=text_color, pad=20)
    
    # Add reference circles and labels with improved visibility
    ax_categories.grid(color=grid_color, alpha=0.5, linewidth=0.5)
    
    # Skip custom concentric circles as they're causing posx/posy warnings
    # Instead, we'll create reference circles using the built-in polar plot functionality
    
    # Add reference circle labels at key values
    for r in [2.5, 5, 7.5, 10]:
        # Using the built-in polar functionality for the circles
        ax_categories.plot(np.linspace(0, 2*np.pi, 100), [r]*100, '--', 
                          color=grid_color, alpha=0.3, linewidth=0.5)
        
        # Add a label on the y-axis for reference
        ax_categories.text(0, r, f"{r}", fontsize=8, color=light_text, 
                          ha='center', va='bottom')
    
    # SKU performance scatter plot
    ax_skus = fig.add_subplot(gs[1, 3:6])
    ax_skus.set_facecolor(bg_color)
    
    # Create a custom colormap based on our score color
    volume_norm = plt.Normalize(min(sku_data['Volume_2024']), max(sku_data['Volume_2024']))
    
    # Plot the SKUs as a scatter plot
    scatter = ax_skus.scatter(
        sku_data['Margin'], 
        sku_data['Growth'], 
        s=sku_data['Volume_2024']/5000,  # Size based on volume
        c=sku_data['Volume_2024'],  # Color based on volume
        cmap='viridis',
        alpha=0.8,
        edgecolor=bg_color,
        linewidth=0.5
    )
    
    # Add a colorbar with improved styling
    cbar = plt.colorbar(scatter, ax=ax_skus, orientation='vertical', pad=0.01)
    cbar.set_label('Volume 2024', color=text_color, fontsize=12, labelpad=10)
    cbar.ax.yaxis.set_tick_params(color=text_color)
    plt.setp(plt.getp(cbar.ax, 'yticklabels'), color=text_color)
    
    # Label the top 3 SKUs with improved visibility
    top_skus = sku_data.nlargest(3, 'Volume_2024')
    for _, sku in top_skus.iterrows():
        ax_skus.annotate(
            sku['SKU'].split('_')[0],
            (sku['Margin'], sku['Growth']),
            xytext=(7, 7),
            textcoords='offset points',
            fontsize=11,
            fontweight='bold',
            color=text_color,
            bbox=dict(facecolor=bg_color, alpha=0.8, edgecolor=score_color, boxstyle='round,pad=0.2'),
            path_effects=[path_effects.withStroke(linewidth=2, foreground=bg_color)]
        )
    
    # Add reference lines with better visibility
    ax_skus.axhline(y=0, color=grid_color, linestyle='--', alpha=0.7, linewidth=1)
    ax_skus.axvline(x=0.75, color=grid_color, linestyle='--', alpha=0.7, linewidth=1)
    
    # Add quadrant labels with improved readability and positioning
    # Define quadrant boxes with background for better visibility
    quadrant_props = dict(boxstyle='round,pad=0.3', facecolor=bg_color, alpha=0.7, edgecolor=grid_color)
    ax_skus.text(0.95, 0.25, 'Premium\nGrowers', ha='right', va='center', 
                fontsize=11, color=text_color, bbox=quadrant_props)
    ax_skus.text(0.65, 0.25, 'Value\nGrowers', ha='right', va='center', 
                fontsize=11, color=text_color, bbox=quadrant_props)
    ax_skus.text(0.65, -0.05, 'Underperformers', ha='right', va='center', 
                fontsize=11, color=text_color, bbox=quadrant_props)
    ax_skus.text(0.95, -0.05, 'Premium\nDecliners', ha='right', va='center', 
                fontsize=11, color=text_color, bbox=quadrant_props)
    
    # Configure the SKU plot
    ax_skus.set_xlabel('Margin', fontsize=14, color=text_color, labelpad=10)
    ax_skus.set_ylabel('Year-over-Year Growth', fontsize=14, color=text_color, labelpad=10)
    ax_skus.set_title('SKU Performance Matrix', fontsize=18, color=text_color, pad=15)
    ax_skus.tick_params(colors=text_color, labelsize=10)
    
    # Set axis limits based on data with better padding
    margin_padding = 0.05
    growth_padding = 0.1
    ax_skus.set_xlim(
        max(0.5, min(sku_data['Margin']) - margin_padding),
        max(sku_data['Margin']) + margin_padding
    )
    ax_skus.set_ylim(
        min(sku_data['Growth']) - growth_padding,
        max(sku_data['Growth']) + growth_padding
    )
    
    # Add a brand mix bar chart to the bottom row
    ax_brands = fig.add_subplot(gs[2, 0:3])
    ax_brands.set_facecolor(bg_color)
    
    # Calculate brand mix
    brand_mix = sku_data.groupby('Brand_Family')['Volume_2024'].sum().reset_index()
    brand_mix['Share'] = brand_mix['Volume_2024'] / brand_mix['Volume_2024'].sum()
    brand_mix = brand_mix.sort_values('Volume_2024', ascending=False)
    
    # Create a horizontal bar chart with distinct colors
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(brand_mix)))
    bars = ax_brands.barh(
        brand_mix['Brand_Family'],
        brand_mix['Volume_2024'],
        color=colors,
        alpha=0.8,
        height=0.7  # Reduced height for better spacing
    )
    
    # Add share labels with improved visibility
    for i, (_, row) in enumerate(brand_mix.iterrows()):
        ax_brands.text(
            row['Volume_2024'] + max(brand_mix['Volume_2024'])*0.02,
            i,
            f"{row['Share']:.1%}",
            va='center',
            fontsize=11,
            color=text_color,
            fontweight='bold'
        )
    
    # Configure the brand chart
    ax_brands.set_title('Brand Family Distribution', fontsize=18, color=text_color, pad=15)
    ax_brands.set_xlabel('Volume (2024)', fontsize=14, color=text_color, labelpad=10)
    ax_brands.tick_params(colors=text_color, labelsize=12)
    ax_brands.spines['top'].set_visible(False)
    ax_brands.spines['right'].set_visible(False)
    
    # Add key metrics to the bottom right
    ax_metrics = fig.add_subplot(gs[2, 3:6])
    ax_metrics.set_facecolor(bg_color)
    ax_metrics.axis('off')
    
    # Create a grid of metrics
    metrics_grid = [
        ['Total SKUs', str(context_data['Total_SKUs']), 'PMI SKUs', str(context_data['PMI_SKUs'])],
        ['Competitor SKUs', str(context_data['Comp_SKUs']), 'Market Share', f"{context_data['Market_Share']:.1%}"],
        ['Total Volume', f"{context_data['Total_Volume']:,}", 'PMI Volume', f"{context_data['PMI_Volume']:,}"],
        ['Green SKUs', str(context_data['Green_Count']), 'Red SKUs', str(context_data['Red_Count'])],
        ['Annual PAX', f"{context_data['PAX_Annual']:,}", '', '']
    ]
    
    # Draw the metrics grid with improved spacing and appearance
    for i, row in enumerate(metrics_grid):
        for j in range(0, len(row), 2):
            if j+1 < len(row) and row[j] and row[j+1]:
                metric_name = row[j]
                metric_value = row[j+1]
                
                # Calculate positions with more space
                x = 0.05 + (j/2) * 0.5
                y = 0.9 - i * 0.15  # More vertical spacing
                
                # Add metric name and value with improved appearance
                ax_metrics.text(x, y, f"{metric_name}:", fontsize=12, color=light_text, ha='left')
                ax_metrics.text(x + 0.25, y, metric_value, fontsize=12, fontweight='bold', color=text_color, ha='right')
    
    # Add the category component bar charts with proper spacing
    components_y = 0.2  # Move up for more space
    
    # Add a title for the components section
    ax_metrics.text(0.05, components_y + 0.25, "Category Component Performance", 
                   fontsize=14, fontweight='bold', color=text_color)
    
    # Improved spacing between category components
    for i, (cat_name, components) in enumerate([
        ('Category A', context_data['Category_A_Components']),
        ('Category B', context_data['Category_B_Components']),
        ('Category C', context_data['Category_C_Components']),
        ('Category D', context_data['Category_D_Components'])
    ]):
        # Position with more horizontal space
        x = 0.05 + i * 0.24
        y = components_y
        
        # Add category title with proper spacing
        ax_metrics.text(x, y + 0.18, cat_name, fontsize=12, fontweight='bold', color=text_color)
        
        # Add component bars with more space
        for j, (comp_name, value) in enumerate(components.items()):
            # Format the component name better
            comp_display = comp_name.replace('_', ' ').title()
            
            # Draw component bar with improved spacing
            bar_y = y + 0.15 - j * 0.035  # More space between bars
            bar_width = 0.2
            bar_height = 0.018
            
            # Background bar
            ax_metrics.add_patch(Rectangle(
                (x, bar_y - bar_height/2),
                bar_width,
                bar_height,
                facecolor=grid_color,
                alpha=0.3
            ))
            
            # Value bar
            ax_metrics.add_patch(Rectangle(
                (x, bar_y - bar_height/2),
                bar_width * value,
                bar_height,
                facecolor=plt.cm.RdYlGn(value),
                alpha=0.8
            ))
            
            # Add component name with better formatting and full name
            ax_metrics.text(x - 0.01, bar_y, comp_display.split()[0], 
                           fontsize=8, color=light_text, ha='right', va='center')
            
            # Add value with better visibility
            ax_metrics.text(x + bar_width * value + 0.01, bar_y, f"{value:.2f}", 
                           fontsize=8, color=text_color, ha='left', va='center', fontweight='bold')
    
    # Add footer with date and better spacing
    fig.text(0.5, 0.01, f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d')} | Portfolio Optimization Analysis",
            ha='center', va='center', fontsize=11, color=light_text)
    
    # Skip tight_layout as it's not compatible with polar axes
    # Instead use custom padding with the figure-level margins
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.3, hspace=0.3)
    
    filename = f"{location_name.replace(' ', '_')}_portfolio_visualization.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor=bg_color)
    plt.close()
    
    print(f"Created visualization for {location_name}: {filename}")
    return filename

if __name__ == "__main__":
    # Load the data
    location_data, kuwait_data, jeju_data, kuwait_skus, jeju_skus, kuwait_context, jeju_context = load_data()
    
    # Create individual visualizations
    kuwait_viz = create_individual_visualization('Kuwait', kuwait_data, kuwait_skus, kuwait_context, dark_theme=True)
    jeju_viz = create_individual_visualization('Jeju', jeju_data, jeju_skus, jeju_context, dark_theme=True)
    
    # Also create light theme versions
    kuwait_light_viz = create_individual_visualization('Kuwait', kuwait_data, kuwait_skus, kuwait_context, dark_theme=False)
    jeju_light_viz = create_individual_visualization('Jeju', jeju_data, jeju_skus, jeju_context, dark_theme=False)
    
    print("All visualizations completed successfully.")