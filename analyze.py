import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Circle
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
import seaborn as sns

# Set style parameters
plt.style.use('seaborn-darkgrid')
sns.set_palette("viridis")


def create_advanced_location_comparison(location_data, location1, location2):
    """
    Create an advanced comparative visualization of two locations.

    Parameters:
    -----------
    location_data : pandas DataFrame
        Data containing location performance metrics
    location1 : str
        Name of first location (high performer)
    location2 : str
        Name of second location (low performer)
    """
    # Filter data for the two locations
    loc1_data = location_data[location_data['Location'] == location1].iloc[0]
    loc2_data = location_data[location_data['Location'] == location2].iloc[0]

    # Set up the radar chart parameters
    categories = ['Cat_A', 'Cat_B', 'Cat_C', 'Cat_D']
    N = len(categories)

    # Calculate angles for each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop

    # Set up the figure with two subplots
    fig = plt.figure(figsize=(16, 9), facecolor='#f9f9f9')

    # Add a title to the figure
    fig.suptitle(f'Portfolio Optimization: Performance Comparison\n{location1} vs {location2}',
                 fontsize=24, fontweight='bold', y=0.98)

    # Add a subtitle explaining the visualization
    plt.figtext(0.5, 0.92,
                'Radar charts show category scores (0-10) with color intensity representing performance level',
                fontsize=14, ha='center')

    # Create the first radar plot (left side)
    ax1 = fig.add_subplot(121, polar=True)

    # Get scores for location 1
    values1 = [loc1_data[cat] for cat in categories]
    values1 += values1[:1]  # Close the loop

    # Set color map based on average performance
    avg_score1 = loc1_data['Avg_Score']
    color1 = plt.cm.viridis(avg_score1 / 10)

    # Plot the radar chart for location 1
    ax1.plot(angles, values1, linewidth=2, linestyle='solid', color=color1, zorder=10)
    ax1.fill(angles, values1, color=color1, alpha=0.4, zorder=5)

    # Set category labels
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(['PMI Performance', 'Category Segments', 'Passenger Mix', 'Location Clusters'],
                        fontsize=12, fontweight='bold')

    # Set y-axis limits and labels
    ax1.set_ylim(0, 10)
    ax1.set_yticks([2, 4, 6, 8, 10])
    ax1.set_yticklabels(['2', '4', '6', '8', '10'], fontsize=10)

    # Set title and location information
    ax1.set_title(f'{location1}\nAvg Score: {avg_score1:.2f}', fontsize=18, pad=20, fontweight='bold')
    plt.figtext(0.25, 0.15, f'Cluster: {loc1_data["Cluster"]}\nScore Range: {loc1_data["Score_Range"]:.2f}',
                fontsize=14, ha='center')

    # Add performance indicator circle
    center_circle1 = Circle((0, 0), 0.5, color=color1, alpha=0.6)
    ax1.add_patch(center_circle1)
    ax1.text(0, 0, f'{avg_score1:.1f}', ha='center', va='center', fontsize=18,
             fontweight='bold', color='white')

    # Create annotations for each category
    for i, cat in enumerate(categories):
        angle = angles[i]
        value = values1[i]
        ax1.annotate(f'{value:.1f}',
                     xy=(angle, value + 0.5),
                     xytext=(angle, value + 1.0),
                     fontsize=11,
                     fontweight='bold',
                     ha='center',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color1, alpha=0.7))

    # Create the second radar plot (right side)
    ax2 = fig.add_subplot(122, polar=True)

    # Get scores for location 2
    values2 = [loc2_data[cat] for cat in categories]
    values2 += values2[:1]  # Close the loop

    # Set color map based on average performance
    avg_score2 = loc2_data['Avg_Score']
    color2 = plt.cm.viridis(avg_score2 / 10)

    # Plot the radar chart for location 2
    ax2.plot(angles, values2, linewidth=2, linestyle='solid', color=color2, zorder=10)
    ax2.fill(angles, values2, color=color2, alpha=0.4, zorder=5)

    # Set category labels
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(['PMI Performance', 'Category Segments', 'Passenger Mix', 'Location Clusters'],
                        fontsize=12, fontweight='bold')

    # Set y-axis limits and labels
    ax2.set_ylim(0, 10)
    ax2.set_yticks([2, 4, 6, 8, 10])
    ax2.set_yticklabels(['2', '4', '6', '8', '10'], fontsize=10)

    # Set title and location information
    ax2.set_title(f'{location2}\nAvg Score: {avg_score2:.2f}', fontsize=18, pad=20, fontweight='bold')
    plt.figtext(0.75, 0.15, f'Cluster: {loc2_data["Cluster"]}\nScore Range: {loc2_data["Score_Range"]:.2f}',
                fontsize=14, ha='center')

    # Add performance indicator circle
    center_circle2 = Circle((0, 0), 0.5, color=color2, alpha=0.6)
    ax2.add_patch(center_circle2)
    ax2.text(0, 0, f'{avg_score2:.1f}', ha='center', va='center', fontsize=18,
             fontweight='bold', color='white')

    # Create annotations for each category
    for i, cat in enumerate(categories):
        angle = angles[i]
        value = values2[i]
        ax2.annotate(f'{value:.1f}',
                     xy=(angle, value + 0.5),
                     xytext=(angle, value + 1.0),
                     fontsize=11,
                     fontweight='bold',
                     ha='center',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color2, alpha=0.7))

    # Add key insights boxes
    # High performer insights
    high_insights = [
        f"Strong in Passenger Mix ({loc1_data['Cat_C']:.1f})",
        f"Excellent Location Clustering ({loc1_data['Cat_D']:.1f})",
        f"Well-balanced across all categories",
        f"Minimal gaps between highest and lowest scores"
    ]

    insight_text1 = "\n".join(high_insights)
    plt.figtext(0.25, 0.30, "Key Success Factors:", fontsize=14, fontweight='bold', ha='center')
    plt.figtext(0.25, 0.25, insight_text1, fontsize=12, ha='center',
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=color1, alpha=0.7))

    # Low performer insights
    low_insights = [
        f"Strong in PMI Performance ({loc2_data['Cat_A']:.1f})",
        f"Very weak in Category Segments ({loc2_data['Cat_B']:.1f})",
        f"Poor Location Clustering ({loc2_data['Cat_D']:.1f})",
        f"Significant gaps between highest and lowest scores"
    ]

    insight_text2 = "\n".join(low_insights)
    plt.figtext(0.75, 0.30, "Improvement Opportunities:", fontsize=14, fontweight='bold', ha='center')
    plt.figtext(0.75, 0.25, insight_text2, fontsize=12, ha='center',
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=color2, alpha=0.7))

    # Add a color scale legend
    norm = mcolors.Normalize(vmin=0, vmax=10)
    sm = ScalarMappable(norm=norm, cmap=plt.cm.viridis)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=[ax1, ax2], orientation='horizontal', pad=0.05,
                        fraction=0.05, shrink=0.5, aspect=20)
    cbar.set_label('Performance Score Scale', fontsize=12)

    # Add a footer with other metrics
    footer_text = (
        "Portfolio Optimization Project - Comparative Location Analysis\n"
        "Score Scale: 0 (Poor) to 10 (Excellent) | Data as of February 2025"
    )
    plt.figtext(0.5, 0.02, footer_text, fontsize=10, ha='center',
                bbox=dict(boxstyle="round,pad=0.5", fc="#f0f0f0", ec="gray", alpha=0.7))

    # Save the figure
    plt.tight_layout(rect=[0, 0.03, 1, 0.90])
    plt.savefig('location_comparison_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Visualization saved as 'location_comparison_analysis.png'")
    return fig


# Load the location data
location_data = pd.read_csv('results/clustered_locations_20250225_030538.csv')

# Create the visualization
fig = create_advanced_location_comparison(location_data, 'Hohhot Baita', 'Hanoi')