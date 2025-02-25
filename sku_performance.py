import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# Set plotting style
plt.style.use('seaborn-darkgrid')
sns.set_palette("deep")


def load_sku_data(file_path='data/df_vols_query.csv'):
    """
    Load SKU data from the CSV file
    """
    df = pd.read_csv(file_path)
    return df


def preprocess_sku_data(df):
    """
    Preprocess the SKU data for analysis
    """
    # Convert volume columns to numeric if needed
    volume_cols = ['2022 Volume', '2023 Volume', '2024 Volume']
    for col in volume_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Calculate year-over-year growth
    if '2023 Volume' in df.columns and '2022 Volume' in df.columns:
        df['Growth_2022_2023'] = np.where(
            df['2022 Volume'] > 0,
            (df['2023 Volume'] - df['2022 Volume']) / df['2022 Volume'],
            np.nan
        )

    if '2024 Volume' in df.columns and '2023 Volume' in df.columns:
        df['Growth_2023_2024'] = np.where(
            df['2023 Volume'] > 0,
            (df['2024 Volume'] - df['2023 Volume']) / df['2023 Volume'],
            np.nan
        )

    # Add volume change columns
    if '2023 Volume' in df.columns and '2022 Volume' in df.columns:
        df['Volume_Change_2022_2023'] = df['2023 Volume'] - df['2022 Volume']

    if '2024 Volume' in df.columns and '2023 Volume' in df.columns:
        df['Volume_Change_2023_2024'] = df['2024 Volume'] - df['2023 Volume']

    return df


def analyze_sku_performance(df):
    """
    Analyze SKU performance by location, brand, and other attributes
    """
    # Create a directory for results if it doesn't exist
    os.makedirs('results/sku_analysis', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 1. Overall SKU Performance
    # ------------------------------
    # Calculate aggregated performance metrics by SKU
    sku_performance = df.groupby(['Brand Family', 'SKU']).agg({
        '2022 Volume': 'sum',
        '2023 Volume': 'sum',
        '2024 Volume': 'sum',
        'Location': 'nunique'
    }).reset_index()

    # Calculate growth rates
    sku_performance['Growth_2022_2023'] = np.where(
        sku_performance['2022 Volume'] > 0,
        (sku_performance['2023 Volume'] - sku_performance['2022 Volume']) / sku_performance['2022 Volume'],
        np.nan
    )

    sku_performance['Growth_2023_2024'] = np.where(
        sku_performance['2023 Volume'] > 0,
        (sku_performance['2024 Volume'] - sku_performance['2023 Volume']) / sku_performance['2023 Volume'],
        np.nan
    )

    # Rename the location count column
    sku_performance.rename(columns={'Location': 'Location_Count'}, inplace=True)

    # Save the SKU performance data
    sku_performance_path = f'results/sku_analysis/sku_performance_{timestamp}.csv'
    sku_performance.to_csv(sku_performance_path, index=False)

    # 2. Location Performance
    # ------------------------------
    # Calculate performance metrics by location
    location_performance = df.groupby('Location').agg({
        '2022 Volume': 'sum',
        '2023 Volume': 'sum',
        '2024 Volume': 'sum',
        'SKU': 'nunique'
    }).reset_index()

    # Calculate growth rates
    location_performance['Growth_2022_2023'] = np.where(
        location_performance['2022 Volume'] > 0,
        (location_performance['2023 Volume'] - location_performance['2022 Volume']) / location_performance[
            '2022 Volume'],
        np.nan
    )

    location_performance['Growth_2023_2024'] = np.where(
        location_performance['2023 Volume'] > 0,
        (location_performance['2024 Volume'] - location_performance['2023 Volume']) / location_performance[
            '2023 Volume'],
        np.nan
    )

    # Rename the SKU count column
    location_performance.rename(columns={'SKU': 'SKU_Count'}, inplace=True)

    # Save the location performance data
    location_performance_path = f'results/sku_analysis/location_performance_{timestamp}.csv'
    location_performance.to_csv(location_performance_path, index=False)

    # 3. Brand Family Performance
    # ------------------------------
    # Calculate performance metrics by brand family
    brand_performance = df.groupby('Brand Family').agg({
        '2022 Volume': 'sum',
        '2023 Volume': 'sum',
        '2024 Volume': 'sum',
        'SKU': 'nunique',
        'Location': 'nunique'
    }).reset_index()

    # Calculate growth rates
    brand_performance['Growth_2022_2023'] = np.where(
        brand_performance['2022 Volume'] > 0,
        (brand_performance['2023 Volume'] - brand_performance['2022 Volume']) / brand_performance['2022 Volume'],
        np.nan
    )

    brand_performance['Growth_2023_2024'] = np.where(
        brand_performance['2023 Volume'] > 0,
        (brand_performance['2024 Volume'] - brand_performance['2023 Volume']) / brand_performance['2023 Volume'],
        np.nan
    )

    # Rename the count columns
    brand_performance.rename(columns={
        'SKU': 'SKU_Count',
        'Location': 'Location_Count'
    }, inplace=True)

    # Save the brand performance data
    brand_performance_path = f'results/sku_analysis/brand_performance_{timestamp}.csv'
    brand_performance.to_csv(brand_performance_path, index=False)

    # 4. TMO Performance
    # ------------------------------
    # Calculate performance metrics by TMO
    tmo_performance = df.groupby('TMO').agg({
        '2022 Volume': 'sum',
        '2023 Volume': 'sum',
        '2024 Volume': 'sum',
        'SKU': 'nunique',
        'Location': 'nunique',
        'Brand Family': 'nunique'
    }).reset_index()

    # Calculate growth rates
    tmo_performance['Growth_2022_2023'] = np.where(
        tmo_performance['2022 Volume'] > 0,
        (tmo_performance['2023 Volume'] - tmo_performance['2022 Volume']) / tmo_performance['2022 Volume'],
        np.nan
    )

    tmo_performance['Growth_2023_2024'] = np.where(
        tmo_performance['2023 Volume'] > 0,
        (tmo_performance['2024 Volume'] - tmo_performance['2023 Volume']) / tmo_performance['2023 Volume'],
        np.nan
    )

    # Rename the count columns
    tmo_performance.rename(columns={
        'SKU': 'SKU_Count',
        'Location': 'Location_Count',
        'Brand Family': 'Brand_Count'
    }, inplace=True)

    # Save the TMO performance data
    tmo_performance_path = f'results/sku_analysis/tmo_performance_{timestamp}.csv'
    tmo_performance.to_csv(tmo_performance_path, index=False)

    return {
        'sku_performance': sku_performance,
        'location_performance': location_performance,
        'brand_performance': brand_performance,
        'tmo_performance': tmo_performance,
        'paths': {
            'sku_performance': sku_performance_path,
            'location_performance': location_performance_path,
            'brand_performance': brand_performance_path,
            'tmo_performance': tmo_performance_path
        }
    }


def identify_top_skus(sku_performance, top_n=20):
    """
    Identify top SKUs based on 2024 volume and growth
    """
    # Top SKUs by 2024 volume
    top_by_volume = sku_performance.sort_values('2024 Volume', ascending=False).head(top_n)

    # Top SKUs by 2023-2024 growth (among SKUs with significant volume)
    volume_threshold = sku_performance['2024 Volume'].quantile(0.75)  # 75th percentile
    top_by_growth = (sku_performance[sku_performance['2024 Volume'] >= volume_threshold]
                     .sort_values('Growth_2023_2024', ascending=False)
                     .head(top_n))

    return {
        'top_by_volume': top_by_volume,
        'top_by_growth': top_by_growth
    }


def identify_underperforming_skus(sku_performance):
    """
    Identify underperforming SKUs based on volume and growth
    """
    # SKUs with declining volume
    declining_skus = sku_performance[
        (sku_performance['Growth_2023_2024'] < -0.1) &  # At least 10% decline
        (sku_performance['2023 Volume'] > 0)  # Had some volume in 2023
        ].sort_values('Growth_2023_2024')

    # SKUs with low volume
    volume_threshold = sku_performance['2024 Volume'].quantile(0.25)  # 25th percentile
    low_volume_skus = sku_performance[
        (sku_performance['2024 Volume'] <= volume_threshold) &
        (sku_performance['2024 Volume'] > 0)  # Still has some volume
        ].sort_values('2024 Volume')

    return {
        'declining_skus': declining_skus,
        'low_volume_skus': low_volume_skus
    }


def visualize_sku_performance(performance_data, top_skus, underperforming_skus):
    """
    Create visualizations for SKU performance
    """
    # Create a directory for visualizations if it doesn't exist
    os.makedirs('results/sku_analysis/visualizations', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    visualization_paths = {}

    # 1. Top 10 SKUs by Volume (2024)
    plt.figure(figsize=(14, 8))
    top10_volume = top_skus['top_by_volume'].head(10)
    sns.barplot(x='2024 Volume', y='SKU', data=top10_volume, palette='viridis')
    plt.title('Top 10 SKUs by 2024 Volume', fontsize=16)
    plt.xlabel('2024 Volume', fontsize=14)
    plt.ylabel('SKU', fontsize=14)
    plt.ticklabel_format(style='plain', axis='x')
    plt.tight_layout()

    top_volume_path = f'results/sku_analysis/visualizations/top10_volume_{timestamp}.png'
    plt.savefig(top_volume_path)
    plt.close()
    visualization_paths['top10_volume'] = top_volume_path

    # 2. Top 10 SKUs by Growth (2023-2024)
    plt.figure(figsize=(14, 8))
    top10_growth = top_skus['top_by_growth'].head(10)
    sns.barplot(x='Growth_2023_2024', y='SKU', data=top10_growth, palette='viridis')
    plt.title('Top 10 SKUs by 2023-2024 Growth', fontsize=16)
    plt.xlabel('Growth Rate', fontsize=14)
    plt.ylabel('SKU', fontsize=14)
    plt.tight_layout()

    top_growth_path = f'results/sku_analysis/visualizations/top10_growth_{timestamp}.png'
    plt.savefig(top_growth_path)
    plt.close()
    visualization_paths['top10_growth'] = top_growth_path

    # 3. Volume Trend Analysis
    plt.figure(figsize=(12, 6))
    years = ['2022 Volume', '2023 Volume', '2024 Volume']
    total_volumes = performance_data['sku_performance'][years].sum()
    plt.plot(range(len(years)), total_volumes, marker='o', linewidth=2)
    plt.title('Total Volume Trend (2022-2024)', fontsize=16)
    plt.xticks(range(len(years)), years, fontsize=12)
    plt.ylabel('Total Volume', fontsize=14)
    plt.grid(True)
    plt.tight_layout()

    trend_path = f'results/sku_analysis/visualizations/volume_trend_{timestamp}.png'
    plt.savefig(trend_path)
    plt.close()
    visualization_paths['volume_trend'] = trend_path

    # 4. Declining SKUs Analysis
    plt.figure(figsize=(14, 8))
    declining = underperforming_skus['declining_skus'].head(10)
    sns.barplot(x='Growth_2023_2024', y='SKU', data=declining, palette='coolwarm_r')
    plt.title('Top 10 Declining SKUs (2023-2024)', fontsize=16)
    plt.xlabel('Growth Rate', fontsize=14)
    plt.ylabel('SKU', fontsize=14)
    plt.tight_layout()

    declining_path = f'results/sku_analysis/visualizations/declining_skus_{timestamp}.png'
    plt.savefig(declining_path)
    plt.close()
    visualization_paths['declining_skus'] = declining_path

    return visualization_paths


def analyze_brand_family_trends(performance_data):
    """
    Analyze trends across brand families
    """
    brand_perf = performance_data['brand_performance']

    # Calculate year-over-year changes in market share
    total_volume = {
        '2022': brand_perf['2022 Volume'].sum(),
        '2023': brand_perf['2023 Volume'].sum(),
        '2024': brand_perf['2024 Volume'].sum()
    }

    brand_perf['Market_Share_2022'] = brand_perf['2022 Volume'] / total_volume['2022']
    brand_perf['Market_Share_2023'] = brand_perf['2023 Volume'] / total_volume['2023']
    brand_perf['Market_Share_2024'] = brand_perf['2024 Volume'] / total_volume['2024']

    # Calculate market share changes
    brand_perf['Market_Share_Change_2023_2024'] = (
            brand_perf['Market_Share_2024'] - brand_perf['Market_Share_2023']
    )

    # Identify growing and declining brand families
    growing_brands = brand_perf[brand_perf['Growth_2023_2024'] > 0.1].sort_values(
        'Growth_2023_2024', ascending=False
    )
    declining_brands = brand_perf[brand_perf['Growth_2023_2024'] < -0.1].sort_values(
        'Growth_2023_2024'
    )

    return {
        'brand_performance': brand_perf,
        'growing_brands': growing_brands,
        'declining_brands': declining_brands
    }


def generate_summary_report(performance_data, top_skus, underperforming_skus, brand_trends):
    """
    Generate a summary report of the analysis
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = f'results/sku_analysis/summary_report_{timestamp}.txt'

    with open(report_path, 'w') as f:
        f.write("SKU Performance Analysis Summary\n")
        f.write("===============================\n\n")

        # Overall metrics
        f.write("1. Overall Performance Metrics\n")
        f.write("----------------------------\n")
        total_skus = len(performance_data['sku_performance'])
        active_skus = len(performance_data['sku_performance'][
                              performance_data['sku_performance']['2024 Volume'] > 0
                              ])
        f.write(f"Total SKUs analyzed: {total_skus}\n")
        f.write(f"Active SKUs in 2024: {active_skus}\n\n")

        # Top performers
        f.write("2. Top Performing SKUs\n")
        f.write("--------------------\n")
        f.write("Top 5 by 2024 Volume:\n")
        for _, row in top_skus['top_by_volume'].head().iterrows():
            f.write(f"- {row['SKU']}: {row['2024 Volume']:,.0f}\n")
        f.write("\n")

        # Underperforming SKUs
        f.write("3. Areas of Concern\n")
        f.write("-----------------\n")
        f.write(f"Number of declining SKUs: {len(underperforming_skus['declining_skus'])}\n")
        f.write(f"Number of low volume SKUs: {len(underperforming_skus['low_volume_skus'])}\n\n")

        # Brand family trends
        f.write("4. Brand Family Performance\n")
        f.write("-------------------------\n")
        f.write("Top 5 Growing Brand Families:\n")
        for _, row in brand_trends['growing_brands'].head().iterrows():
            f.write(f"- {row['Brand Family']}: {row['Growth_2023_2024'] * 100:.1f}% growth\n")

    return report_path


def main():
    """
    Main function to run the SKU performance analysis
    """
    # Load and process data
    df = pd.read_excel('data/baselist.xlsx')

    # Perform analyses
    performance_data = analyze_sku_performance(df)  # Changed from calculate_performance_metrics
    top_skus = identify_top_skus(performance_data['sku_performance'])
    underperforming_skus = identify_underperforming_skus(performance_data['sku_performance'])

    # Create visualizations
    visualization_paths = visualize_sku_performance(
        performance_data, top_skus, underperforming_skus
    )

    # Analyze brand trends
    brand_trends = analyze_brand_family_trends(performance_data)

    # Generate summary report
    report_path = generate_summary_report(
        performance_data, top_skus, underperforming_skus, brand_trends
    )

    print("Analysis complete. Results saved to:")
    print(f"Summary report: {report_path}")
    print("Visualization files:", *visualization_paths.values(), sep="\n- ")

if __name__ == "__main__":
    main()