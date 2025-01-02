import json
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import numpy as np

# Define bot usernames
BOTS = {'renovate', 'renovatebot', 'dependabot'}

def parse_arguments():
    parser = argparse.ArgumentParser(description='Analyze GitHub Pull Requests.')
    parser.add_argument('input_file', type=str, help='Path to the input JSON file.')
    parser.add_argument('output_dir', type=str, help='Directory to save output graphs.')
    return parser.parse_args()

def load_data(input_file):
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        sys.exit(1)

def preprocess_data(data):
    processed = []
    for pr in data:
        try:
            # Ignore PRs with mergedAt as null
            if not pr.get('mergedAt'):
                continue

            # Extract and lowercase author login
            author_info = pr.get('author')
            if not author_info or not isinstance(author_info, dict):
                # Skip PRs where 'author' is null or not a dict
                continue

            author_login = author_info.get('login')
            if not author_login:
                # Skip PRs where 'login' is missing or empty
                continue

            author = author_login.lower()

            # Skip bot PRs
            if author in BOTS:
                continue

            # Parse dates with timezone awareness
            pr['createdAt'] = pd.to_datetime(pr['createdAt'], utc=True)
            pr['closedAt'] = pd.to_datetime(pr['closedAt'], utc=True)
            pr['mergedAt'] = pd.to_datetime(pr['mergedAt'], utc=True)

            # Set 'author' to the string username
            pr['author'] = author

            # Append to processed list
            processed.append(pr)
        except Exception as e:
            print("Error processing PR:")
            print(json.dumps(pr, indent=2))
            print(f"Exception: {e}")
    df = pd.DataFrame(processed)

    # Verification: Print dtypes and sample data
    print("DataFrame dtypes after preprocessing:")
    print(df.dtypes)
    print("\nSample data:")
    if not df.empty:
        print(df[['author']].head())
    else:
        print("DataFrame is empty after preprocessing.")

    return df

def plot_open_prs_per_day(df, output_dir):
    """
    Plots the number of open PRs per day.
    """
    # Create a date range from the earliest createdAt to the latest closedAt
    start_date = df['createdAt'].min().floor('D')
    end_date = df['closedAt'].max().ceil('D')
    all_days = pd.date_range(start=start_date, end=end_date, freq='D', tz='UTC')

    # Initialize a series to hold open PR counts
    open_prs = pd.Series(0, index=all_days)

    # For each PR, increment the count for each day it was open
    for _, pr in df.iterrows():
        pr_days = pd.date_range(start=pr['createdAt'].floor('D'), end=pr['closedAt'].floor('D'), freq='D', tz='UTC')
        open_prs.loc[pr_days] += 1

    # Plot
    plt.figure(figsize=(15,7))
    open_prs.plot()
    plt.title('Number of Open PRs Per Day')
    plt.xlabel('Date')
    plt.ylabel('Open PRs')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'open_prs_per_day.png'))
    plt.close()

def plot_merged_prs_per_week_divided_by_contributors(df, output_dir):
    """
    Plots the number of PRs merged per week divided by the number of contributors.
    """
    # Sort by mergedAt
    df_sorted = df.sort_values('mergedAt')

    # Manually calculate the start of the week (Monday), preserving timezone
    df_sorted['week'] = (df_sorted['mergedAt'] - pd.to_timedelta(df_sorted['mergedAt'].dt.weekday, unit='D')).dt.floor('D')

    # Group by week and count merged PRs
    merged_per_week = df_sorted.groupby('week').size().rename('merged_prs')

    # Group by week and count unique contributors
    contributors_per_week = df_sorted.groupby('week')['author'].nunique().rename('unique_contributors')

    # Calculate the ratio
    ratio = (merged_per_week / contributors_per_week).rename('prs_per_contributor')

    # Plot
    plt.figure(figsize=(15,7))
    plt.plot(ratio.index, ratio.values, marker='o', label='PRs Merged per Contributor')
    plt.title('Number of PRs Merged Per Week Divided by Number of Contributors')
    plt.xlabel('Week')
    plt.ylabel('PRs Merged / Contributors')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prs_per_week_divided_by_contributors.png'))
    plt.close()

def plot_moving_average_contributors(df, output_dir):
    """
    Plots the total number of contributors per week with a rolling 4-week average.
    """
    # Create a week column manually (start of the week)
    df['week'] = (df['mergedAt'] - pd.to_timedelta(df['mergedAt'].dt.weekday, unit='D')).dt.floor('D')

    # Ensure 'author' is string type
    df['author'] = df['author'].astype(str)

    # Group by week and count unique contributors
    contributors_per_week = df.groupby('week')['author'].nunique()

    # Calculate moving average (window of 4 weeks)
    moving_avg = contributors_per_week.rolling(window=4, min_periods=1).mean()

    # Plot
    plt.figure(figsize=(15,7))
    plt.plot(contributors_per_week.index, contributors_per_week.values, marker='o', label='Unique Contributors')
    plt.plot(moving_avg.index, moving_avg.values, marker='o', label='4-Week Rolling Average', linestyle='--')
    plt.title('Total Number of Contributors with 4-Week Rolling Average')
    plt.xlabel('Week')
    plt.ylabel('Number of Contributors')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'total_contributors_with_rolling_average.png'))
    plt.close()

def plot_merge_heatmap(df, output_dir):
    """
    Plots a heatmap showing the frequency of PR merges by day of the week and hour of the day.
    """
    # Extract hour of day
    df['merge_hour'] = df['mergedAt'].dt.hour

    # Extract day of week
    df['merge_day'] = df['mergedAt'].dt.day_name()

    # Create a pivot table
    heatmap_data = df.groupby(['merge_day', 'merge_hour']).size().unstack(fill_value=0)

    # Reorder days of week
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_data = heatmap_data.reindex(days_order)

    # Plot
    plt.figure(figsize=(15,7))
    sns.heatmap(heatmap_data, cmap='YlGnBu')
    plt.title('Heatmap of Merge Times')
    plt.xlabel('Hour of Day')
    plt.ylabel('Day of Week')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'merge_heatmap.png'))
    plt.close()

def plot_merge_duration_distribution(df, output_dir):
    """
    Plots the distribution of PR merge durations categorized into defined time buckets.
    """
    # Define buckets in hours
    bins = [0, 1, 3, 6, 24, 48, 96, df['duration_hours'].max()]
    labels = ['0-1', '1-3', '3-6', '6-24', '24-48', '48-96', '96+']

    df['duration_bucket'] = pd.cut(df['duration_hours'], bins=bins, labels=labels, right=False)

    # Count the number of PRs in each bucket
    distribution = df['duration_bucket'].value_counts().sort_index()

    # Plot
    plt.figure(figsize=(10,6))
    sns.barplot(x=distribution.index, y=distribution.values, palette='viridis')
    plt.title('Distribution of PR Merge Durations')
    plt.xlabel('Duration (Hours)')
    plt.ylabel('Number of PRs')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'merge_duration_distribution.png'))
    plt.close()

# ======= New Functions Added Below =======

def plot_total_contributors_with_rolling_average(df, output_dir):
    """
    Plots the total number of unique contributors per week along with a rolling 4-week average.
    """
    # Create a week column if not already present
    if 'week' not in df.columns:
        df['week'] = (df['mergedAt'] - pd.to_timedelta(df['mergedAt'].dt.weekday, unit='D')).dt.floor('D')
    
    # Group by week and count unique contributors
    contributors_per_week = df.groupby('week')['author'].nunique().rename('unique_contributors')
    
    # Calculate rolling 4-week average
    rolling_avg = contributors_per_week.rolling(window=4, min_periods=1).mean().rename('4_week_avg')
    
    # Combine into a single DataFrame
    combined = pd.concat([contributors_per_week, rolling_avg], axis=1)
    
    # Plot
    plt.figure(figsize=(15,7))
    plt.plot(combined.index, combined['unique_contributors'], marker='o', label='Unique Contributors')
    plt.plot(combined.index, combined['4_week_avg'], marker='o', label='4-Week Rolling Average', linestyle='--')
    plt.title('Total Number of Contributors with 4-Week Rolling Average')
    plt.xlabel('Week')
    plt.ylabel('Number of Contributors')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'total_contributors_with_rolling_average.png'))
    plt.close()

def plot_prs_per_week_divided_by_contributors_with_rolling_average(df, output_dir):
    """
    Plots the number of PRs merged per week divided by the number of contributors,
    along with a rolling 4-week average to smooth the data.
    """
    # Ensure 'week' column exists
    if 'week' not in df.columns:
        df['week'] = (df['mergedAt'] - pd.to_timedelta(df['mergedAt'].dt.weekday, unit='D')).dt.floor('D')
    
    # Group by week and count merged PRs
    merged_per_week = df.groupby('week').size().rename('merged_prs')
    
    # Group by week and count unique contributors
    contributors_per_week = df.groupby('week')['author'].nunique().rename('unique_contributors')
    
    # Calculate the ratio
    ratio = (merged_per_week / contributors_per_week).rename('prs_per_contributor')
    
    # Calculate rolling 4-week average
    rolling_avg = ratio.rolling(window=4, min_periods=1).mean().rename('4_week_avg')
    
    # Combine into a single DataFrame
    combined = pd.concat([ratio, rolling_avg], axis=1)
    
    # Plot
    plt.figure(figsize=(15,7))
    plt.plot(combined.index, combined['prs_per_contributor'], marker='o', label='PRs Merged per Contributor')
    plt.plot(combined.index, combined['4_week_avg'], marker='o', label='4-Week Rolling Average', linestyle='--')
    plt.title('PRs Merged Per Week Divided by Number of Contributors with 4-Week Rolling Average')
    plt.xlabel('Week')
    plt.ylabel('PRs Merged / Contributors')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prs_per_week_divided_by_contributors_with_rolling_average.png'))
    plt.close()

# ======= End of New Functions =======

def main():
    args = parse_arguments()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    data = load_data(args.input_file)

    # Preprocess data
    df = preprocess_data(data)

    if df.empty:
        print("No data to process after filtering.")
        sys.exit(0)

    # Perform analyses
    plot_open_prs_per_day(df, args.output_dir)
    plot_merged_prs_per_week_divided_by_contributors(df, args.output_dir)
    plot_moving_average_contributors(df, args.output_dir)
    plot_merge_heatmap(df, args.output_dir)
    plot_merge_duration_distribution(df, args.output_dir)

    # ======= Call New Plot Functions =======
    plot_total_contributors_with_rolling_average(df, args.output_dir)
    plot_prs_per_week_divided_by_contributors_with_rolling_average(df, args.output_dir)
    # ======= End of New Plot Calls =======

    print(f"Analysis complete. Graphs saved to {args.output_dir}")

if __name__ == "__main__":
    main()
