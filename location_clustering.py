import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns
import os
from datetime import datetime

# Set plotting style
plt.style.use('seaborn-darkgrid')
sns.set_palette("deep")

def load_category_scores(results_dir='results'):
    """
    Load the category scores from the CSV files
    """
    # Find the most recent files
    score_files = {}
    for category in ['A', 'B', 'C', 'D']:
        pattern = f'Category {category}_scores_'
        matching_files = [f for f in os.listdir(results_dir) if f.startswith(pattern) and f.endswith('.csv')]
        if matching_files:
            # Sort by date in filename
            latest_file = sorted(matching_files, reverse=True)[0]
            score_files[category] = os.path.join(results_dir, latest_file)
    
    # Load each score file
    scores = {}
    for category, filepath in score_files.items():
        df = pd.read_csv(filepath)
        scores[category] = df
    
    return scores

def merge_category_scores(scores):
    """
    Merge all category scores into a single dataframe
    """
    # Start with Category A
    if 'A' not in scores:
        raise ValueError("Category A scores are required for the merged dataset")
    
    merged_df = scores['A'][['Location', 'Cat_A']].copy()
    
    # Add other categories
    for category in ['B', 'C', 'D']:
        if category in scores:
            category_df = scores[category][['Location', f'Cat_{category}']]
            merged_df = pd.merge(merged_df, category_df, on='Location', how='outer')
    
    # Calculate average score where all categories are available
    score_columns = [col for col in merged_df.columns if col.startswith('Cat_')]
    merged_df['Avg_Score'] = merged_df[score_columns].mean(axis=1)
    
    return merged_df

def perform_location_clustering(merged_scores, n_clusters=4):
    """
    Perform clustering on locations based on their category scores
    """
    # Select features for clustering
    features = [col for col in merged_scores.columns if col.startswith('Cat_')]
    
    # Handle missing values - fill with mean of column
    clustering_data = merged_scores[features].copy()
    for col in features:
        clustering_data[col] = clustering_data[col].fillna(clustering_data[col].mean())
    
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(clustering_data)
    
    # Determine the optimal number of clusters using the Elbow Method
    inertia = []
    K_range = range(1, 10)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(scaled_data)
        inertia.append(kmeans.inertia_)
    
    # Plot the Elbow curve
    plt.figure(figsize=(12, 6))
    plt.plot(K_range, inertia, 'bo-', marker='o')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia (Within-cluster sum of squares)')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True)
    
    # Save the plot
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    elbow_plot_path = f'results/elbow_curve_{timestamp}.png'
    plt.savefig(elbow_plot_path)
    plt.close()
    
    # Perform clustering with the specified number of clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled_data)
    
    # Add cluster labels to the dataframe
    merged_scores['Cluster'] = clusters
    
    # Analyze cluster characteristics
    cluster_centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), 
                                  columns=features)
    
    cluster_summary = merged_scores.groupby('Cluster')[features + ['Avg_Score']].mean()
    cluster_summary['Count'] = merged_scores.groupby('Cluster').size()
    
    return merged_scores, cluster_summary, elbow_plot_path

def visualize_clusters(merged_scores, cluster_summary):
    """
    Visualize the clusters using various plots
    """
    # Create a directory for visualizations if it doesn't exist
    os.makedirs('results/visualizations', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 1. Radar chart for cluster centers
    categories = [col for col in cluster_summary.columns if col.startswith('Cat_')]
    
    # Create a separate radar chart for each cluster
    for cluster_id in cluster_summary.index:
        # Number of variables
        N = len(categories)
        
        # What will be the angle of each axis in the plot
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Initialize the figure
        fig, ax = plt.figure(figsize=(10, 10)), plt.subplot(111, polar=True)
        
        # Get the values for the current cluster
        values = cluster_summary.loc[cluster_id, categories].values.flatten().tolist()
        values += values[:1]  # Close the loop
        
        # Draw one axis per variable and add labels
        plt.xticks(angles[:-1], categories, size=12)
        
        # Draw the chart
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=f'Cluster {cluster_id}')
        ax.fill(angles, values, alpha=0.25)
        
        # Set y-axis limits
        plt.ylim(0, 10)
        
        # Add cluster details in the title
        plt.title(f'Cluster {cluster_id} Profile (Count: {int(cluster_summary.loc[cluster_id, "Count"])})', 
                 size=15, color='black', y=1.1)
        
        # Save the radar chart
        radar_path = f'results/visualizations/cluster_{cluster_id}_radar_{timestamp}.png'
        plt.savefig(radar_path)
        plt.close()
    
    # 2. Scatter plot matrix for all locations, colored by cluster
    score_cols = [col for col in merged_scores.columns if col.startswith('Cat_')]
    scatter_data = merged_scores[score_cols + ['Cluster']].copy()
    
    # Create a pairplot
    plt.figure(figsize=(15, 15))
    scatter_plot = sns.pairplot(scatter_data, hue='Cluster', palette='deep', 
                              height=2.5, diag_kind='kde', plot_kws={'alpha': 0.6})
    scatter_plot.fig.suptitle('Cluster Distribution Across Score Categories', y=1.02, fontsize=16)
    
    # Save the scatter plot matrix
    scatter_path = f'results/visualizations/cluster_scatter_matrix_{timestamp}.png'
    plt.savefig(scatter_path)
    plt.close()
    
    # 3. Bar chart comparing average scores across clusters
    plt.figure(figsize=(12, 6))
    cluster_summary[categories + ['Avg_Score']].plot(kind='bar', figsize=(14, 8))
    plt.title('Average Scores by Cluster', fontsize=15)
    plt.ylabel('Score Value', fontsize=12)
    plt.xlabel('Cluster', fontsize=12)
    plt.xticks(rotation=0)
    plt.grid(axis='y')
    plt.legend(title='Score Category', loc='upper right')
    
    # Save the bar chart
    bar_path = f'results/visualizations/cluster_comparison_{timestamp}.png'
    plt.savefig(bar_path)
    plt.close()
    
    return {
        'radar_charts': f'results/visualizations/cluster_*_radar_{timestamp}.png',
        'scatter_matrix': scatter_path,
        'bar_chart': bar_path
    }

def identify_top_performers(merged_scores, n_top=20):
    """
    Identify top performing locations based on average score
    """
    # Sort by average score in descending order
    top_performers = merged_scores.sort_values('Avg_Score', ascending=False).head(n_top)
    return top_performers

def identify_improvement_candidates(merged_scores, n_candidates=20):
    """
    Identify locations with the greatest potential for improvement
    """
    # For simplicity, we'll define improvement candidates as locations with 
    # a high deviation between their highest and lowest category scores
    
    # Calculate the range of scores for each location
    score_cols = [col for col in merged_scores.columns if col.startswith('Cat_')]
    merged_scores['Score_Range'] = merged_scores[score_cols].max(axis=1) - merged_scores[score_cols].min(axis=1)
    
    # Sort by score range in descending order to find locations with most imbalanced scores
    improvement_candidates = merged_scores.sort_values('Score_Range', ascending=False).head(n_candidates)
    
    return improvement_candidates

def generate_location_insights(merged_scores, cluster_summary, top_performers, improvement_candidates):
    """
    Generate insights about locations based on clustering and performance
    """
    # Create a results directory if it doesn't exist
    os.makedirs('results/insights', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 1. Generate cluster insights
    cluster_insights = {}
    for cluster_id in cluster_summary.index:
        # Get locations in this cluster
        cluster_locations = merged_scores[merged_scores['Cluster'] == cluster_id]
        
        # Calculate key metrics
        avg_score = cluster_summary.loc[cluster_id, 'Avg_Score']
        top_cats = [col for col in cluster_summary.columns if col.startswith('Cat_')]
        top_cat = top_cats[cluster_summary.loc[cluster_id, top_cats].argmax()]
        bottom_cat = top_cats[cluster_summary.loc[cluster_id, top_cats].argmin()]
        
        # Get exemplar locations (closest to cluster center)
        cluster_center = cluster_summary.loc[cluster_id, [col for col in cluster_summary.columns if col.startswith('Cat_')]]
        
        # Calculate Euclidean distance to cluster center for each location
        distances = []
        for _, row in cluster_locations.iterrows():
            location_vector = row[[col for col in row.index if col.startswith('Cat_')]]
            distance = np.sqrt(((location_vector - cluster_center) ** 2).sum())
            distances.append((row['Location'], distance))
        
        # Sort by distance and get top 3 exemplars
        exemplars = sorted(distances, key=lambda x: x[1])[:3]
        exemplar_names = [e[0] for e in exemplars]
        
        # Store insights
        cluster_insights[cluster_id] = {
            'size': int(cluster_summary.loc[cluster_id, 'Count']),
            'avg_score': avg_score,
            'top_category': top_cat,
            'bottom_category': bottom_cat,
            'exemplar_locations': exemplar_names
        }
    
    # 2. Generate insights for top performers
    top_performer_insights = []
    for _, row in top_performers.iterrows():
        # Calculate which categories contribute most to the high score
        score_cols = [col for col in row.index if col.startswith('Cat_')]
        top_cat = score_cols[np.argmax([row[col] for col in score_cols])]
        
        insight = {
            'location': row['Location'],
            'avg_score': row['Avg_Score'],
            'top_category': top_cat,
            'cluster': row['Cluster'],
            'score_breakdown': {col: row[col] for col in score_cols}
        }
        top_performer_insights.append(insight)
    
    # 3. Generate insights for improvement candidates
    improvement_insights = []
    for _, row in improvement_candidates.iterrows():
        # Identify the weakest category
        score_cols = [col for col in row.index if col.startswith('Cat_')]
        weakest_cat = score_cols[np.argmin([row[col] for col in score_cols])]
        strongest_cat = score_cols[np.argmax([row[col] for col in score_cols])]
        
        insight = {
            'location': row['Location'],
            'avg_score': row['Avg_Score'],
            'weakest_category': weakest_cat,
            'strongest_category': strongest_cat,
            'score_range': row['Score_Range'],
            'cluster': row['Cluster'],
            'score_breakdown': {col: row[col] for col in score_cols}
        }
        improvement_insights.append(insight)
    
    # Save insights to JSON and CSV files
    insights = {
        'cluster_insights': cluster_insights,
        'top_performer_insights': top_performer_insights,
        'improvement_candidate_insights': improvement_insights
    }
    
    # Save to CSV
    # Clusters
    clusters_df = pd.DataFrame([
        {
            'Cluster': cluster_id,
            'Size': data['size'],
            'Avg_Score': data['avg_score'],
            'Top_Category': data['top_category'],
            'Bottom_Category': data['bottom_category'],
            'Exemplar_Locations': ', '.join(data['exemplar_locations'])
        }
        for cluster_id, data in cluster_insights.items()
    ])
    clusters_df.to_csv(f'results/insights/cluster_insights_{timestamp}.csv', index=False)
    
    # Top performers
    top_df = pd.DataFrame([
        {
            'Location': item['location'],
            'Avg_Score': item['avg_score'],
            'Top_Category': item['top_category'],
            'Cluster': item['cluster'],
            **item['score_breakdown']
        }
        for item in top_performer_insights
    ])
    top_df.to_csv(f'results/insights/top_performers_{timestamp}.csv', index=False)
    
    # Improvement candidates
    improve_df = pd.DataFrame([
        {
            'Location': item['location'],
            'Avg_Score': item['avg_score'],
            'Weakest_Category': item['weakest_category'],
            'Strongest_Category': item['strongest_category'],
            'Score_Range': item['score_range'],
            'Cluster': item['cluster'],
            **item['score_breakdown']
        }
        for item in improvement_insights
    ])
    improve_df.to_csv(f'results/insights/improvement_candidates_{timestamp}.csv', index=False)
    
    return {
        'cluster_insights_path': f'results/insights/cluster_insights_{timestamp}.csv',
        'top_performers_path': f'results/insights/top_performers_{timestamp}.csv',
        'improvement_candidates_path': f'results/insights/improvement_candidates_{timestamp}.csv'
    }

def run_location_clustering_analysis():
    """
    Run the entire location clustering analysis
    """
    print("Loading category scores...")
    scores = load_category_scores()
    
    print(f"Found scores for categories: {', '.join(scores.keys())}")
    
    print("Merging category scores...")
    merged_scores = merge_category_scores(scores)
    print(f"Merged data contains {len(merged_scores)} locations with scores")
    
    print("Performing clustering analysis...")
    n_clusters = 4  # You can adjust this based on the elbow curve
    clustered_scores, cluster_summary, elbow_plot = perform_location_clustering(merged_scores, n_clusters)
    print(f"Clustered {len(clustered_scores)} locations into {n_clusters} clusters")
    print(f"Elbow curve saved to: {elbow_plot}")
    
    print("Visualizing clusters...")
    visualization_paths = visualize_clusters(clustered_scores, cluster_summary)
    print("Cluster visualizations saved to:")
    for name, path in visualization_paths.items():
        print(f"- {name}: {path}")
    
    print("Identifying top performers and improvement candidates...")
    top_performers = identify_top_performers(clustered_scores)
    improvement_candidates = identify_improvement_candidates(clustered_scores)
    
    print("Generating location insights...")
    insight_paths = generate_location_insights(
        clustered_scores, cluster_summary, top_performers, improvement_candidates)
    
    print("Analysis complete. Results saved to:")
    for name, path in insight_paths.items():
        print(f"- {name}: {path}")
    
    # Save the complete clustered dataset
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = f'results/clustered_locations_{timestamp}.csv'
    clustered_scores.to_csv(output_path, index=False)
    print(f"Complete clustered dataset saved to: {output_path}")
    
    return {
        'clustered_scores': clustered_scores,
        'cluster_summary': cluster_summary,
        'visualization_paths': visualization_paths,
        'insight_paths': insight_paths,
        'output_path': output_path
    }

if __name__ == "__main__":
    run_location_clustering_analysis()
