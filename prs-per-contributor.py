#!/usr/bin/env python3
import json
import sys
from datetime import datetime, timedelta
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

def load_pr_data(filename):
    """Load and parse the PR JSON data."""
    with open(filename, 'r') as f:
        return json.load(f)

def is_bot(author):
    """Check if the PR author is a bot."""
    if author is None:
        return True  # Filter out PRs with no author
    
    if not isinstance(author, dict) or 'login' not in author:
        return True  # Filter out malformed author entries
        
    login = author['login']
    if login is None:
        return True
        
    bot_names = ['renovatebot', 'dependabot', 'renovate']
    return any(bot in login.lower() for bot in bot_names)

def process_prs(data):
    """Process PR data into weekly statistics."""
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Convert timestamps to datetime and handle timezone
    df['mergedAt'] = pd.to_datetime(df['mergedAt'], utc=True)
    
    # Filter out bot PRs and null authors
    df = df[~df['author'].apply(is_bot)]
    
    # Drop any rows with NaT in mergedAt
    df = df.dropna(subset=['mergedAt'])
    
    # Convert to local time and get week start
    df['week_start'] = df['mergedAt'].dt.tz_localize(None).dt.to_period('W-MON').apply(lambda x: x.start_time if pd.notna(x) else None)
    
    # Calculate weekly statistics
    weekly_stats = []
    for week_start in sorted(df['week_start'].dropna().unique()):
        week_data = df[df['week_start'] == week_start]
        
        stats = {
            'week': week_start,
            'week_num': len(weekly_stats),  # Week number starting from 0
            'merged_prs': len(week_data),
            'contributors': len(week_data['author'].apply(lambda x: x['login']).unique()),
            'prs_per_contributor': len(week_data) / len(week_data['author'].apply(lambda x: x['login']).unique()) if len(week_data) > 0 else 0
        }
        weekly_stats.append(stats)
    
    return pd.DataFrame(weekly_stats)

def calculate_moving_average(stats_df, window=4):
    """Calculate moving average of PRs per contributor."""
    stats_df['moving_avg'] = stats_df['prs_per_contributor'].rolling(window=window, min_periods=1).mean()
    return stats_df

def analyze_trend(stats_df):
    """Analyze the trend in PRs per contributor."""
    # Perform linear regression
    X = stats_df.index.values.reshape(-1, 1)
    y = stats_df['moving_avg'].values
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X, y)
    
    trend_line = model.predict(X)
    slope = model.coef_[0]
    
    # Calculate percentage change
    first_value = trend_line[0]
    last_value = trend_line[-1]
    total_change_percent = ((last_value - first_value) / first_value) * 100
    
    return trend_line, slope, total_change_percent

def get_author_login(author):
    """Safely extract login from author data."""
    if author is None or not isinstance(author, dict):
        return None
    return author.get('login')

def analyze_yearly_trends(df):
    """Analyze yearly trends starting from 2020."""
    # Extract year from mergedAt
    df['year'] = df['mergedAt'].dt.year
    
    # Filter for years >= 2020
    yearly_stats = []
    for year in sorted(df[df['year'] >= 2020]['year'].unique()):
        year_data = df[df['year'] == year]
        
        # Calculate weekly averages for the year
        weekly_stats = []
        for _, week_data in year_data.groupby(pd.Grouper(key='mergedAt', freq='W')):
            if len(week_data) > 0:  # Only include weeks with PRs
                prs = len(week_data)
                # Safely get unique contributors, filtering out None values
                logins = [login for login in week_data['author'].apply(get_author_login) if login is not None]
                contributors = len(set(logins))
                if contributors > 0:  # Avoid division by zero
                    weekly_stats.append(prs / contributors)
        
        # Calculate yearly statistics, handling empty weekly_stats
        if weekly_stats:
            avg_prs = np.mean(weekly_stats)
            median_prs = np.median(weekly_stats)
        else:
            avg_prs = 0
            median_prs = 0
            
        # Get total contributors for the year, safely
        year_logins = [login for login in year_data['author'].apply(get_author_login) if login is not None]
        
        stats = {
            'year': year,
            'avg_prs_per_contributor': avg_prs,
            'median_prs_per_contributor': median_prs,
            'total_prs': len(year_data),
            'total_contributors': len(set(year_logins)),
            'weeks_with_activity': len(weekly_stats)
        }
        yearly_stats.append(stats)
    
    return pd.DataFrame(yearly_stats)
    """Analyze yearly trends starting from 2020."""
    # Extract year from mergedAt
    df['year'] = df['mergedAt'].dt.year
    
    # Filter for years >= 2020
    yearly_stats = []
    for year in sorted(df[df['year'] >= 2020]['year'].unique()):
        year_data = df[df['year'] == year]
        
        # Calculate weekly averages for the year
        weekly_stats = []
        for _, week_data in year_data.groupby(pd.Grouper(key='mergedAt', freq='W')):
            if len(week_data) > 0:  # Only include weeks with PRs
                prs = len(week_data)
                contributors = len(week_data['author'].apply(lambda x: x['login']).unique())
                weekly_stats.append(prs / contributors)
        
        stats = {
            'year': year,
            'avg_prs_per_contributor': np.mean(weekly_stats) if weekly_stats else 0,
            'median_prs_per_contributor': np.median(weekly_stats) if weekly_stats else 0,
            'total_prs': len(year_data),
            'total_contributors': len(year_data['author'].apply(lambda x: x['login']).unique()),
            'weeks_with_activity': len(weekly_stats)
        }
        yearly_stats.append(stats)
    
    return pd.DataFrame(yearly_stats)

def create_plots(stats_df, pr_data):
    """Create plots using Plotly."""
    # Original trend plot
    trend_line, slope, total_change_percent = analyze_trend(stats_df)
    
    # Create two subplots
    fig = make_subplots(rows=2, cols=1, 
                       subplot_titles=("PRs per Contributor Over Time",
                                     "Yearly Comparison"),
                       vertical_spacing=0.3)
    
    # Add moving average to first subplot
    fig.add_trace(
        go.Scatter(
            x=stats_df['week_num'],
            y=stats_df['moving_avg'],
            name="4-Week Moving Average",
            mode='lines',
            line=dict(color='blue')
        ),
        row=1, col=1
    )
    
    # Add trend line to first subplot
    fig.add_trace(
        go.Scatter(
            x=stats_df['week_num'],
            y=trend_line,
            name="Trend Line",
            mode='lines',
            line=dict(color='red', dash='dash')
        ),
        row=1, col=1
    )
    
    # Calculate and add yearly comparison
    df = pd.DataFrame(pr_data)
    df['mergedAt'] = pd.to_datetime(df['mergedAt'], utc=True)
    yearly_stats = analyze_yearly_trends(df)
    
    # Add yearly comparison bar chart
    fig.add_trace(
        go.Bar(
            x=yearly_stats['year'],
            y=yearly_stats['avg_prs_per_contributor'],
            name="Yearly Average",
            text=yearly_stats['avg_prs_per_contributor'].round(2),
            textposition='auto',
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        height=900,
        title_text=f"Pull Request Analysis (Total Change: {total_change_percent:.1f}%)",
        showlegend=True
    )
    
    fig.update_xaxes(title_text="Week Number", row=1, col=1)
    fig.update_xaxes(title_text="Year", row=2, col=1)
    fig.update_yaxes(title_text="PRs per Contributor (4-Week Moving Average)", row=1, col=1)
    fig.update_yaxes(title_text="Average PRs per Contributor", row=2, col=1)
    
    return fig, yearly_stats

def generate_html_report(stats_df, plot, yearly_stats):
    """Generate HTML report with statistics and visualization."""
    # Calculate trend statistics
    _, slope, total_change_percent = analyze_trend(stats_df)
    
    # Calculate additional statistics
    first_month = stats_df['moving_avg'].iloc[:12].mean()  # First 3 months average
    last_month = stats_df['moving_avg'].iloc[-12:].mean()  # Last 3 months average
    percent_change = ((last_month - first_month) / first_month) * 100
    
    # Create yearly comparison table
    yearly_table = yearly_stats.to_html(index=False, float_format=lambda x: '{:.2f}'.format(x) if isinstance(x, float) else str(x))
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>PR Analysis Report</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 40px;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }}
            .summary {{
                margin: 20px 0;
                padding: 20px;
                background-color: #f5f5f5;
                border-radius: 5px;
            }}
            .trend-positive {{
                color: green;
            }}
            .trend-negative {{
                color: red;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            th, td {{
                padding: 12px;
                text-align: left;
                border: 1px solid #ddd;
            }}
            th {{
                background-color: #f5f5f5;
            }}
        </style>
    </head>
    <body>
        <h1>Pull Request Analysis Report</h1>
        
        <div class="summary">
            <h2>Overall Trend Analysis</h2>
            <p>Total Weeks Analyzed: {len(stats_df)}</p>
            <p>First 3 Months Average: {first_month:.2f} PRs per contributor</p>
            <p>Last 3 Months Average: {last_month:.2f} PRs per contributor</p>
            <p class="{'trend-positive' if percent_change > 0 else 'trend-negative'}">
                Overall Change: {percent_change:.1f}% ({total_change_percent:.1f}% trend line)
            </p>
            <p>Slope: {slope:.4f} PRs per contributor per week</p>
        </div>

        <div class="summary">
            <h2>Yearly Comparison (2020 onwards)</h2>
            {yearly_table}
        </div>

        <div id="plot"></div>
        
        <script>
            var plotData = {plot.to_json()};
            Plotly.newPlot('plot', plotData.data, plotData.layout);
        </script>
    </body>
    </html>
    """
    
    return html_content

def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <pr_data.json>")
        sys.exit(1)
    
    # Load and process data
    pr_data = load_pr_data(sys.argv[1])
    stats_df = process_prs(pr_data)
    stats_df = calculate_moving_average(stats_df)
    
    # Create visualization
    plot, yearly_stats = create_plots(stats_df, pr_data)
    
    # Generate report
    report = generate_html_report(stats_df, plot, yearly_stats)
    
    # Write to file
    with open('pr_analysis_report.html', 'w') as f:
        f.write(report)
    
    print("Analysis complete. Report generated as 'pr_analysis_report.html'")

if __name__ == "__main__":
    main()