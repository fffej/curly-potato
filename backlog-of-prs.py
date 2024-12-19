# Analyse a JSON file full of PRs and see how many are open on any given day
# Graph the results. 
# Runs as a command line tool taking the JSON as an argument
# Produces a graph in the same directory as the JSON file

import json
import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 backlog-of-prs.py <path-to-json-file>")
        sys.exit(1)
    
    json_file = sys.argv[1]
    with open(json_file, 'r') as f:
        prs = json.load(f)
    
    # Add debugging output
    print("First PR data:", json.dumps(prs[0], indent=2))
    print("\nDataFrame columns:", pd.DataFrame(prs).columns.tolist())
    
    # Convert PR data to DataFrame with relevant dates
    df = pd.DataFrame(prs)
    df['created_at'] = pd.to_datetime(df['createdAt'])
    df['closed_at'] = pd.to_datetime(df['closedAt'])
    
    # Calculate PR lifecycle metrics
    df['time_to_close'] = (df['closed_at'] - df['created_at']).dt.total_seconds() / (24 * 3600)  # in days
    
    # Create weekly date range from earliest PR to latest
    date_range = pd.date_range(start=df['created_at'].min(), 
                             end=max(df['closed_at'].max(), 
                                   pd.Timestamp.now(tz='UTC')),
                             freq='W')
    
    # Count open PRs for each week
    open_prs = []
    for date in date_range:
        # Count PRs created before this date and either still open or closed after this date
        count = len(df[(df['created_at'] <= date) & 
                      ((df['closed_at'].isna()) | (df['closed_at'] > date))])
        open_prs.append(count)
    
    # Weekly PR creation and closure analysis
    weekly_stats = pd.DataFrame({
        'date': date_range,
        'open_prs': open_prs,
        # Count PRs created during each week
        'new_prs': [len(df[
            (df['created_at'] >= date) & 
            (df['created_at'] < date + pd.Timedelta(weeks=1))
        ]) for date in date_range],
        # Count PRs closed during each week (excluding None/NaT values)
        'closed_prs': [len(df[
            (df['closed_at'].notna()) &  # Only include actually closed PRs
            (df['closed_at'] >= date) & 
            (df['closed_at'] < date + pd.Timedelta(weeks=1))
        ]) for date in date_range]
    })
    
    # Calculate rolling averages for smoother trends
    weekly_stats['new_prs_4w_avg'] = weekly_stats['new_prs'].rolling(window=4).mean()
    weekly_stats['closed_prs_4w_avg'] = weekly_stats['closed_prs'].rolling(window=4).mean()
    
    # Print insights
    print("\nPR Lifecycle Metrics:")
    print(f"Median time to close: {df['time_to_close'].median():.1f} days")
    print(f"Mean time to close: {df['time_to_close'].mean():.1f} days")
    print(f"90th percentile time to close: {df['time_to_close'].quantile(0.9):.1f} days")
    
    print("\nWorkflow Metrics:")
    print(f"Average PRs created per week: {weekly_stats['new_prs'].mean():.1f}")
    print(f"Average PRs closed per week: {weekly_stats['closed_prs'].mean():.1f}")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 20))
    gs = fig.add_gridspec(5, 2)  # 5 rows, 2 columns
    
    # Plot 1: Open PRs over time (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(weekly_stats['date'], weekly_stats['open_prs'], label='Open PRs')
    ax1.set_title('Open PRs Over Time')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Number of Open PRs')
    ax1.grid(True)
    
    # Plot 2: PR Creation vs Closure Rate (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(weekly_stats['date'], weekly_stats['new_prs_4w_avg'], label='New PRs (4-week avg)')
    ax2.plot(weekly_stats['date'], weekly_stats['closed_prs_4w_avg'], label='Closed PRs (4-week avg)')
    ax2.set_title('PR Creation vs Closure Rate')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Number of PRs per Week')
    ax2.grid(True)
    ax2.legend()
    
    # Calculate and plot lifecycle metrics
    lifecycle_metrics = analyze_pr_lifecycle(df)
    
    # Plot 3: PR Resolution Time Distribution (middle left)
    ax3 = fig.add_subplot(gs[1, 0])
    resolution_times = df['time_to_close_hours'][df['time_to_close_hours'] < df['time_to_close_hours'].quantile(0.95)]
    ax3.hist(resolution_times, bins=50, edgecolor='black')
    ax3.set_title('PR Resolution Time Distribution\n(excluding outliers)')
    ax3.set_xlabel('Hours to Close')
    ax3.set_ylabel('Number of PRs')
    
    # Plot 4: PR Size vs Time to Close (middle right)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.scatter(df['size'], df['time_to_close_hours'], alpha=0.1)
    ax4.set_title(f'PR Size vs Time to Close\nCorrelation: {lifecycle_metrics["size_metrics"]["size_correlation"]:.2f}')
    ax4.set_xlabel('PR Size (additions + deletions)')
    ax4.set_ylabel('Hours to Close')
    ax4.set_yscale('log')
    ax4.set_xscale('log')
    
    # Plot 5 & 6: Creation and Closure Time Heatmaps
    ax5 = fig.add_subplot(gs[2, :])
    creation_matrix = create_time_heatmap_data(df, 'created')
    im = ax5.imshow(creation_matrix, cmap='YlOrRd')
    ax5.set_title('PR Creation Patterns by Day and Hour')
    ax5.set_xlabel('Hour of Day (UTC)')
    ax5.set_ylabel('Day of Week')
    plt.colorbar(im, ax=ax5)
    
    # Add detailed statistics text
    stats_text = f"""
    PR Lifecycle Metrics:
    • Median time to close: {lifecycle_metrics['time_metrics']['median_hours']/24:.1f} days
    • Mean time to close: {lifecycle_metrics['time_metrics']['mean_hours']/24:.1f} days
    • 90th percentile: {lifecycle_metrics['time_metrics']['p90_hours']/24:.1f} days
    
    Size Metrics:
    • Median PR size: {lifecycle_metrics['size_metrics']['median_size']:.0f} changes
    • Size-Time Correlation: {lifecycle_metrics['size_metrics']['size_correlation']:.2f}
    
    Weekly Averages:
    • New PRs: {weekly_stats['new_prs'].mean():.1f}
    • Closed PRs: {weekly_stats['closed_prs'].mean():.1f}
    """
    
    fig.text(0.1, 0.02, stats_text, fontsize=10, family='monospace')
    
    plt.tight_layout()
    
    # Save plots
    output_path = json_file.rsplit('.', 1)[0] + '_pr_analysis.png'
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"\nAnalysis graphs saved as {output_path}")

def analyze_pr_lifecycle(df):
    # Calculate time metrics (in hours for more granular analysis)
    df['time_to_close_hours'] = (df['closed_at'] - df['created_at']).dt.total_seconds() / 3600
    df['size'] = df['additions'] + df['deletions']
    
    # Create time-of-day and day-of-week fields (in UTC)
    df['created_hour'] = df['created_at'].dt.hour
    df['created_day'] = df['created_at'].dt.dayofweek
    df['closed_hour'] = df['closed_at'].dt.hour
    df['closed_day'] = df['closed_at'].dt.dayofweek
    
    return {
        'time_metrics': {
            'median_hours': df['time_to_close_hours'].median(),
            'mean_hours': df['time_to_close_hours'].mean(),
            'p90_hours': df['time_to_close_hours'].quantile(0.9),
            'p95_hours': df['time_to_close_hours'].quantile(0.95)
        },
        'size_metrics': {
            'median_size': df['size'].median(),
            'mean_size': df['size'].mean(),
            'size_correlation': df['size'].corr(df['time_to_close_hours'])
        },
        'timing_data': {
            'created_hours': df['created_hour'].value_counts().sort_index(),
            'closed_hours': df['closed_hour'].value_counts().sort_index(),
            'created_days': df['created_day'].value_counts().sort_index(),
            'closed_days': df['closed_day'].value_counts().sort_index()
        }
    }

def plot_lifecycle_analysis(metrics, df):
    fig = plt.figure(figsize=(15, 20))
    gs = fig.add_gridspec(4, 2)
    
    # 1. PR Resolution Time Distribution
    ax1 = fig.add_subplot(gs[0, 0])
    plt.hist(df['time_to_close_hours'][df['time_to_close_hours'] < df['time_to_close_hours'].quantile(0.95)], 
             bins=50, edgecolor='black')
    ax1.set_title('PR Resolution Time Distribution\n(excluding outliers)')
    ax1.set_xlabel('Hours to Close')
    ax1.set_ylabel('Number of PRs')
    
    # 2. PR Size vs Time to Close
    ax2 = fig.add_subplot(gs[0, 1])
    plt.scatter(df['size'], df['time_to_close_hours'], alpha=0.1)
    ax2.set_title(f'PR Size vs Time to Close\nCorrelation: {metrics["size_metrics"]["size_correlation"]:.2f}')
    ax2.set_xlabel('PR Size (additions + deletions)')
    ax2.set_ylabel('Hours to Close')
    ax2.set_yscale('log')
    ax2.set_xscale('log')
    
    # 3. Creation Time Heatmap
    ax3 = fig.add_subplot(gs[1, :])
    creation_matrix = create_time_heatmap_data(df, 'created')
    im = ax3.imshow(creation_matrix, cmap='YlOrRd')
    ax3.set_title('PR Creation Patterns')
    ax3.set_xlabel('Hour of Day (UTC)')
    ax3.set_ylabel('Day of Week')
    plt.colorbar(im, ax=ax3)
    
    # 4. Closure Time Heatmap
    ax4 = fig.add_subplot(gs[2, :])
    closure_matrix = create_time_heatmap_data(df, 'closed')
    im = ax4.imshow(closure_matrix, cmap='YlOrRd')
    ax4.set_title('PR Closure Patterns')
    ax4.set_xlabel('Hour of Day (UTC)')
    ax4.set_ylabel('Day of Week')
    plt.colorbar(im, ax=ax4)
    
    # 5. Monthly Trends
    ax5 = fig.add_subplot(gs[3, :])
    monthly_stats = df.set_index('created_at').resample('M').agg({
        'number': 'count',
        'time_to_close_hours': 'median'
    })
    ax5_twin = ax5.twinx()
    ax5.plot(monthly_stats.index, monthly_stats['number'], color='blue', label='Number of PRs')
    ax5_twin.plot(monthly_stats.index, monthly_stats['time_to_close_hours'], 
                 color='red', label='Median Close Time (hours)')
    ax5.set_title('Monthly PR Trends')
    ax5.set_xlabel('Month')
    ax5.set_ylabel('Number of PRs (blue)')
    ax5_twin.set_ylabel('Median Close Time in Hours (red)')
    
    plt.tight_layout()
    return fig

def create_time_heatmap_data(df, prefix):
    matrix = np.zeros((7, 24))
    for day in range(7):
        for hour in range(24):
            matrix[day, hour] = len(df[
                (df[f'{prefix}_day'] == day) & 
                (df[f'{prefix}_hour'] == hour)
            ])
    return matrix

if __name__ == "__main__":
    main()
