"""
Kuwait and Jeju Individual Portfolio Comparison Visualizations
Based on the style of portfolio_comparison and portfolio_comparison_equal_volume
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
    kuwait_skus = pd.DataFrame({
        'SKU': [f'MARLBORO_{i}' for i in range(1, 6)] + 
               [f'PARLIAMENT_{i}' for i in range(1, 4)] + 
               [f'HEETS_{i}' for i in range(1, 5)] + 
               [f'L&M_{i}' for i in range(1, 3)] + 
               [f'CHESTERFIELD_{i}' for i in range(1, 4)],
        'Brand': ['MARLBORO']*5 + ['PARLIAMENT']*3 + ['HEETS']*4 + ['L&M']*2 + ['CHESTERFIELD']*3,
        'Volume': np.random.randint(300000, 900000, 17),
        'Growth': np.random.uniform(0.02, 0.35, 17),
        'Margin': np.random.uniform(0.70, 0.95, 17),
        'Premium': np.random.choice([True, False], 17, p=[0.7, 0.3]),
        'Segment': np.random.choice(['Full Flavor', 'Light', 'Menthol', 'Ultra Light'], 17)
    })
    
    # Create simulated SKU data for Jeju (low performer)
    jeju_skus = pd.DataFrame({
        'SKU': [f'MARLBORO_{i}' for i in range(1, 4)] + 
               [f'PARLIAMENT_{i}' for i in range(1, 3)] + 
               [f'L&M_{i}' for i in range(1, 4)] + 
               [f'BOND_{i}' for i in range(1, 3)] + 
               [f'LARK_{i}' for i in range(1, 4)],
        'Brand': ['MARLBORO']*3 + ['PARLIAMENT']*2 + ['L&M']*3 + ['BOND']*2 + ['LARK']*3,
        'Volume': np.random.randint(100000, 400000, 13),
        'Growth': np.random.uniform(-0.15, 0.10, 13),
        'Margin': np.random.uniform(0.50, 0.75, 13),
        'Premium': np.random.choice([True, False], 13, p=[0.3, 0.7]),
        'Segment': np.random.choice(['Full Flavor', 'Light'], 13)
    })
    
    # Kuwait portfolio metrics
    kuwait_metrics = {
        'Total_Volume': 5300000,
        'PMI_Volume': 3200000,
        'Market_Share': 0.48,
        'Avg_Margin': 0.76,
        'Premium_Mix': 0.70,
        'Growth_Rate': 0.12,
        'Green_SKUs': 5,
        'Yellow_SKUs': 10,
        'Red_SKUs': 2,
        'PAX': 15800000,
        'Categories': {
            'Cat_A': kuwait_data['Cat_A'],
            'Cat_B': kuwait_data['Cat_B'],
            'Cat_C': kuwait_data['Cat_C'],
            'Cat_D': kuwait_data['Cat_D']
        },
        'Segment_Distribution': {
            'Full Flavor': 0.40,
            'Light': 0.35,
            'Menthol': 0.15,
            'Ultra Light': 0.10
        }
    }
    
    # Jeju portfolio metrics
    jeju_metrics = {
        'Total_Volume': 2400000,
        'PMI_Volume': 1400000,
        'Market_Share': 0.61,
        'Avg_Margin': 0.62,
        'Premium_Mix': 0.30,
        'Growth_Rate': -0.05,
        'Green_SKUs': 0,
        'Yellow_SKUs': 5,
        'Red_SKUs': 8,
        'PAX': 9500000,
        'Categories': {
            'Cat_A': jeju_data['Cat_A'],
            'Cat_B': jeju_data['Cat_B'],
            'Cat_C': jeju_data['Cat_C'],
            'Cat_D': jeju_data['Cat_D']
        },
        'Segment_Distribution': {
            'Full Flavor': 0.70,
            'Light': 0.30,
            'Menthol': 0.0,
            'Ultra Light': 0.0
        }
    }
    
    return kuwait_data, jeju_data, kuwait_skus, jeju_skus, kuwait_metrics, jeju_metrics

def create_portfolio_performance_visual(location_name, location_data, sku_data, metrics, equal_volume=False):
    """
    Create a portfolio performance visualization similar to portfolio_comparison_equal_volume.png
    
    Parameters:
    -----------
    location_name : str
        Name of the location
    location_data : Series
        Data from the scores dataset
    sku_data : DataFrame
        SKU-level data
    metrics : dict
        Additional portfolio performance metrics
    equal_volume : bool
        Whether to use equal volume scaling
    """
    # Set up the plot style with dark theme
    plt.style.use('dark_background')
    
    # Determine colors based on performance
    score = location_data['Avg_Score']
    if score >= 7.0:
        main_color = '#3498db'  # Blue for high performers
        performance_label = "HIGH PERFORMER"
    elif score >= 4.0:
        main_color = '#f39c12'  # Orange for mid performers
        performance_label = "MODERATE PERFORMER" 
    else:
        main_color = '#e74c3c'  # Red for low performers
        performance_label = "REQUIRES OPTIMIZATION"
    
    # Create figure and grid layout
    fig = plt.figure(figsize=(16, 12), facecolor='#0a0a0a')
    gs = GridSpec(4, 3, figure=fig, height_ratios=[0.8, 1, 1.2, 1], 
                 hspace=0.4, wspace=0.3)
    
    # Add title section at the top
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis('off')
    
    # Add title and subtitle
    title_text = f"PORTFOLIO PERFORMANCE: {location_name.upper()}"
    subtitle = f"Score: {score:.2f}/10 - {performance_label}"
    
    ax_title.text(0.5, 0.7, title_text, fontsize=24, ha='center', va='center',
                 fontweight='bold', color='white')
    
    title_box = dict(boxstyle="round,pad=0.5", facecolor=main_color, alpha=0.8)
    ax_title.text(0.5, 0.3, subtitle, fontsize=16, ha='center', va='center',
                 fontweight='bold', color='white', bbox=title_box)
    
    # Create the category score radar chart
    ax_radar = fig.add_subplot(gs[1, 0], polar=True)
    
    # Categories and actual scores
    categories = ['Category A', 'Category B', 'Category C', 'Category D']
    category_scores = [
        metrics['Categories']['Cat_A'],
        metrics['Categories']['Cat_B'], 
        metrics['Categories']['Cat_C'],
        metrics['Categories']['Cat_D']
    ]
    
    # Angle calculations for radar chart
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    
    # Close the loop
    category_scores = category_scores + [category_scores[0]]
    angles = angles + [angles[0]]
    
    # Set up the radar chart
    ax_radar.plot(angles, category_scores, 'o-', linewidth=2, color=main_color)
    ax_radar.fill(angles, category_scores, color=main_color, alpha=0.25)
    
    # Fix axis to go in the right order and start at top
    ax_radar.set_theta_offset(np.pi/2)
    ax_radar.set_theta_direction(-1)
    
    # Set category labels
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(categories)
    
    # Set radar chart grid
    ax_radar.set_yticks([2, 4, 6, 8, 10])
    ax_radar.set_ylim(0, 10)
    ax_radar.grid(True, alpha=0.3)
    
    # Add score labels at each point
    for i, (angle, score) in enumerate(zip(angles[:-1], category_scores[:-1])):
        ax_radar.text(angle, score + 0.5, f"{score:.1f}", 
                     ha='center', va='center', fontsize=12, 
                     fontweight='bold', color='white',
                     bbox=dict(facecolor='#0a0a0a', alpha=0.7, boxstyle='round,pad=0.2'))
    
    ax_radar.set_title('Category Scores', fontsize=14, pad=15, color='white')
    
    # Create segment distribution bar chart
    ax_segments = fig.add_subplot(gs[1, 1])
    segments = list(metrics['Segment_Distribution'].keys())
    segment_values = list(metrics['Segment_Distribution'].values())
    
    # Segment bar colors based on overall performance - using updated matplotlib approach
    segment_colors = plt.colormaps['viridis'](np.linspace(0.2, 0.8, len(segments)))
    
    bars = ax_segments.barh(segments, segment_values, color=segment_colors)
    
    # Add percentage labels
    for i, (value, bar) in enumerate(zip(segment_values, bars)):
        if value > 0.05:  # Only show label if segment is large enough
            ax_segments.text(value + 0.02, i, f"{value:.0%}", 
                           va='center', fontsize=10, fontweight='bold', color='white')
    
    ax_segments.set_xlim(0, 1.0)
    ax_segments.set_title('Segment Distribution', fontsize=14, pad=15, color='white')
    ax_segments.set_xlabel('Share of Portfolio', color='white')
    ax_segments.tick_params(colors='white')
    
    # Key metrics summary
    ax_metrics = fig.add_subplot(gs[1, 2])
    ax_metrics.axis('off')
    
    # Create a formatted metrics summary
    y_pos = 0.9
    step = 0.15
    
    # Function to add metric with colored indicator
    def add_metric(name, value, format_str, threshold_good=None, threshold_bad=None, higher_is_better=True):
        nonlocal y_pos
        
        # Determine color based on thresholds
        if threshold_good is not None and threshold_bad is not None:
            if higher_is_better:
                if value >= threshold_good:
                    color = '#2ecc71'  # Green
                elif value <= threshold_bad:
                    color = '#e74c3c'  # Red
                else:
                    color = '#f39c12'  # Yellow
            else:
                if value <= threshold_good:
                    color = '#2ecc71'  # Green
                elif value >= threshold_bad:
                    color = '#e74c3c'  # Red
                else:
                    color = '#f39c12'  # Yellow
        else:
            color = 'white'
        
        # Add metric name
        ax_metrics.text(0.05, y_pos, name, fontsize=12, ha='left', va='center', color='white')
        
        # Add value with colored text
        formatted_value = format_str.format(value)
        ax_metrics.text(0.95, y_pos, formatted_value, fontsize=12, ha='right', 
                       va='center', color=color, fontweight='bold')
        
        # Draw separator line
        ax_metrics.axhline(y=y_pos-0.05, xmin=0.05, xmax=0.95, color='#555555', alpha=0.5, linewidth=1)
        
        y_pos -= step
    
    ax_metrics.text(0.5, 0.98, 'Key Performance Metrics', fontsize=14, ha='center', 
                   va='top', color='white', fontweight='bold')
    
    # Add key metrics with thresholds
    add_metric('Market Share', metrics['Market_Share'], '{:.1%}', 0.4, 0.2)
    add_metric('Average Margin', metrics['Avg_Margin'], '{:.2f}', 0.7, 0.5)
    add_metric('Premium Mix', metrics['Premium_Mix'], '{:.1%}', 0.6, 0.3)
    add_metric('Growth Rate', metrics['Growth_Rate'], '{:+.1%}', 0.05, -0.05)
    add_metric('Total Volume', metrics['Total_Volume'], '{:,.0f}', None, None)
    add_metric('Green SKUs', metrics['Green_SKUs'], '{:d}', 3, 1)
    add_metric('Red SKUs', metrics['Red_SKUs'], '{:d}', 2, 5, False)
    
    # SKU Performance scatter plot
    ax_sku = fig.add_subplot(gs[2, :2])
    
    # Adjust volume size based on setting
    if equal_volume:
        # All SKUs appear the same size for better visualization of margins and growth
        sizes = np.ones(len(sku_data)) * 200
    else:
        # Size represents volume
        sizes = sku_data['Volume'] / 5000
    
    # Color based on brand
    brands = sku_data['Brand'].unique()
    color_map = {}
    for i, brand in enumerate(brands):
        color_map[brand] = plt.cm.tab10(i % 10)
    
    colors = [color_map[brand] for brand in sku_data['Brand']]
    
    # Create scatter plot
    scatter = ax_sku.scatter(sku_data['Margin'], sku_data['Growth'], 
                           s=sizes, c=colors, alpha=0.7)
    
    # Add reference lines
    ax_sku.axhline(y=0, color='#555555', linestyle='--', alpha=0.5)
    ax_sku.axhline(y=0.05, color='#555555', linestyle=':', alpha=0.3)
    ax_sku.axvline(x=0.75, color='#555555', linestyle='--', alpha=0.5)
    
    # Add quadrant labels
    ax_sku.text(0.9, 0.25, 'Premium Growers', ha='center', va='center', 
              fontsize=10, color='white', alpha=0.8,
              bbox=dict(facecolor='#0a0a0a', alpha=0.7, boxstyle='round,pad=0.2'))
    
    ax_sku.text(0.6, 0.25, 'Value Growers', ha='center', va='center', 
              fontsize=10, color='white', alpha=0.8,
              bbox=dict(facecolor='#0a0a0a', alpha=0.7, boxstyle='round,pad=0.2'))
    
    ax_sku.text(0.6, -0.1, 'Underperformers', ha='center', va='center', 
              fontsize=10, color='white', alpha=0.8,
              bbox=dict(facecolor='#0a0a0a', alpha=0.7, boxstyle='round,pad=0.2'))
    
    ax_sku.text(0.9, -0.1, 'Premium Decliners', ha='center', va='center', 
              fontsize=10, color='white', alpha=0.8,
              bbox=dict(facecolor='#0a0a0a', alpha=0.7, boxstyle='round,pad=0.2'))
    
    # Label top 3 SKUs by volume
    top_skus = sku_data.nlargest(3, 'Volume')
    for _, sku in top_skus.iterrows():
        ax_sku.annotate(sku['SKU'].split('_')[0], 
                      (sku['Margin'], sku['Growth']),
                      xytext=(5, 5), textcoords='offset points',
                      fontsize=9, fontweight='bold', color='white',
                      bbox=dict(facecolor='#0a0a0a', alpha=0.7, boxstyle='round,pad=0.2'))
    
    # Format SKU plot
    ax_sku.set_xlabel('Margin', color='white')
    ax_sku.set_ylabel('Year-over-Year Growth', color='white')
    ax_sku.tick_params(colors='white')
    
    # Create legend for brands
    handles = []
    labels = []
    for brand in brands:
        handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[brand], 
                                 markersize=8, alpha=0.7, linewidth=0))
        labels.append(brand)
    
    ax_sku.legend(handles, labels, loc='upper left', framealpha=0.3)
    
    title_text = "SKU Performance Matrix"
    if equal_volume:
        title_text += " (Equal Volume Visualization)"
    ax_sku.set_title(title_text, fontsize=14, pad=15, color='white')
    
    # Brand mix pie chart
    ax_brands = fig.add_subplot(gs[2, 2])
    
    # Aggregate by brand
    brand_volumes = sku_data.groupby('Brand')['Volume'].sum()
    
    # Sort brands by volume
    brand_volumes = brand_volumes.sort_values(ascending=False)
    
    # Create pie chart with custom colors
    wedges, texts, autotexts = ax_brands.pie(
        brand_volumes,
        labels=[f"{brand}" for brand in brand_volumes.index],
        autopct='%1.1f%%',
        startangle=90,
        shadow=False,
        wedgeprops={'edgecolor': '#0a0a0a', 'linewidth': 1, 'alpha': 0.8},
        textprops={'color': 'white'},
    )
    
    # Style the percentage labels
    for text in autotexts:
        text.set_color('white')
        text.set_fontweight('bold')
        text.set_fontsize(9)
    
    # Style the category labels
    for text in texts:
        text.set_fontsize(9)
    
    ax_brands.set_title('Brand Mix by Volume', fontsize=14, pad=15, color='white')
    
    # Create SKU performance summary
    ax_summary = fig.add_subplot(gs[3, :])
    ax_summary.axis('off')
    
    # Create a table to show detailed metrics
    premium_skus = sku_data[sku_data['Premium'] == True]
    value_skus = sku_data[sku_data['Premium'] == False]
    
    # Create table-like layout
    cols = 3
    col_width = 1/cols
    
    # Header row for the table
    headers = ['Metric', 'Value', 'Context']
    for i, header in enumerate(headers):
        # Create a FancyBboxPatch for header background instead of using bbox parameter
        header_x = i*col_width + 0.05
        header_y = 0.92
        header_width = col_width * 0.9
        header_height = 0.06
        
        header_bg = Rectangle(
            (header_x, header_y),
            header_width, header_height,
            facecolor='#333333', alpha=0.8,
            edgecolor='#555555', linewidth=0.5,
            zorder=1
        )
        ax_summary.add_patch(header_bg)
        
        # Add header text
        ax_summary.text(i*col_width + col_width/2, 0.95, header, 
                      ha='center', va='center', fontsize=12, 
                      fontweight='bold', color='white',
                      zorder=2)
    
    # Define table data
    table_data = [
        ['Total SKUs', f"{len(sku_data)}", 
         f"Portfolio has {len(premium_skus)} premium and {len(value_skus)} value SKUs"],
        
        ['Portfolio Score', f"{score:.2f}/10", 
         f"Location is in the {performance_label.lower()} category"],
        
        ['Growth Trend', f"{metrics['Growth_Rate']:+.1%}", 
         f"{'Positive growth trend' if metrics['Growth_Rate'] > 0 else 'Negative growth trend'}"],
        
        ['Segment Strength', f"{max(metrics['Segment_Distribution'].items(), key=lambda x: x[1])[0]}: {max(metrics['Segment_Distribution'].values()):.0%}", 
         "Primary segment dominates portfolio"],
        
        ['Margin Profile', f"{metrics['Avg_Margin']:.2f}", 
         f"{'Strong premium positioning' if metrics['Avg_Margin'] > 0.7 else 'Value-oriented portfolio'}"],
        
        ['SKU Health', f"G:{metrics['Green_SKUs']} Y:{metrics['Yellow_SKUs']} R:{metrics['Red_SKUs']}", 
         f"{'Healthy SKU portfolio' if metrics['Green_SKUs'] > metrics['Red_SKUs'] else 'SKU optimization needed'}"]
    ]
    
    # Add table rows
    for row_idx, row_data in enumerate(table_data):
        y_pos = 0.85 - row_idx * 0.12
        
        # Add cell background
        for col_idx in range(cols):
            cell_bg_color = '#222222' if row_idx % 2 == 0 else '#1a1a1a'
            rect = Rectangle((col_idx*col_width + 0.05, y_pos - 0.05), 
                            col_width*0.9, 0.10, 
                            facecolor=cell_bg_color, edgecolor='#444444', 
                            alpha=0.8, linewidth=0.5)
            ax_summary.add_patch(rect)
        
        # Add cell content
        for col_idx, cell_data in enumerate(row_data):
            if col_idx == 1:  # Value column gets different styling
                ax_summary.text(col_idx*col_width + col_width/2, y_pos, cell_data, 
                              ha='center', va='center', fontsize=12, 
                              fontweight='bold', color=main_color)
            else:
                ax_summary.text(col_idx*col_width + col_width/2, y_pos, cell_data, 
                              ha='center' if col_idx == 0 else 'left', 
                              va='center', fontsize=10, color='white')
    
    # Title for summary section with manual background
    summary_title_bg = Rectangle(
        (0.2, 0.96), 0.6, 0.08,
        facecolor='#333333', alpha=0.8,
        edgecolor='#555555', linewidth=0.5,
        zorder=1
    )
    ax_summary.add_patch(summary_title_bg)
    
    # Add title text
    ax_summary.text(0.5, 1.0, 'Portfolio Performance Summary', fontsize=14, 
                  ha='center', va='center', color='white', fontweight='bold',
                  zorder=2)
    
    # Add footer with date
    fig.text(0.5, 0.02, f"Generated on {pd.Timestamp.now().strftime('%Y-%m-%d')} | Portfolio Analysis",
           ha='center', va='center', fontsize=10, color='#888888')
    
    # Adjust layout and save
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # Different filenames based on mode
    if equal_volume:
        filename = f"{location_name.replace(' ', '_')}_portfolio_equal_volume.png"
    else:
        filename = f"{location_name.replace(' ', '_')}_portfolio.png"
        
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='#0a0a0a')
    plt.close()
    
    print(f"Created {'equal volume' if equal_volume else 'standard'} visualization for {location_name}: {filename}")
    return filename

if __name__ == "__main__":
    # Load the data
    kuwait_data, jeju_data, kuwait_skus, jeju_skus, kuwait_metrics, jeju_metrics = load_data()
    
    # Create standard volume-based visualizations
    kuwait_viz = create_portfolio_performance_visual('Kuwait', kuwait_data, kuwait_skus, kuwait_metrics, equal_volume=False)
    jeju_viz = create_portfolio_performance_visual('Jeju', jeju_data, jeju_skus, jeju_metrics, equal_volume=False)
    
    # Create equal volume visualizations
    kuwait_equal_viz = create_portfolio_performance_visual('Kuwait', kuwait_data, kuwait_skus, kuwait_metrics, equal_volume=True)
    jeju_equal_viz = create_portfolio_performance_visual('Jeju', jeju_data, jeju_skus, jeju_metrics, equal_volume=True)
    
    print("All portfolio visualizations completed successfully.")