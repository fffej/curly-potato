import json
from datetime import datetime, timedelta
import pandas as pd
from flask import Flask, render_template, jsonify, request
import plotly
import plotly.graph_objs as go
from collections import defaultdict
import os

app = Flask(__name__)

# Load repository configuration from JSON file
def load_repository_config():
    try:
        with open('repositories.json', 'r') as f:
            config = json.load(f)
            repos = config['repositories']
            if not repos:
                raise ValueError("No repositories defined in config file")
            return repos
    except Exception as e:
        print(f"Error loading repository configuration: {str(e)}")
        # Fallback to default repository if config fails
        return {
            "Default Repository": "pr_data.json"
        }

REPOSITORIES = load_repository_config()

def load_pr_data(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    # Convert date columns to datetime with UTC timezone
    for col in ['createdAt', 'closedAt', 'mergedAt']:
        df[col] = pd.to_datetime(df[col], utc=True)
    # Extract author login
    df['author_login'] = df['author'].apply(lambda x: x['login'] if x else None)
    return df

def get_repository_data(repo_name):
    """Get the DataFrame for a specific repository"""
    if repo_name not in REPOSITORIES:
        raise ValueError(f"Unknown repository: {repo_name}")
    return load_pr_data(REPOSITORIES[repo_name])

def process_pr_data(df, year=None, comparison_year=None):
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # Exclude bot authors
    bot_authors = ['renovate', 'dependabot']
    df = df[~df['author_login'].isin(bot_authors)]
    
    # Initialize results dictionary
    results = {}
    
    # Process data for primary year
    year = year or datetime.now().year
    year_data = process_year_data(df, year)
    results['primary'] = year_data
    
    # Process data for comparison year if specified
    if comparison_year:
        comparison_data = process_year_data(df, comparison_year)
        results['comparison'] = comparison_data
    
    return results

def process_year_data(df, year):
    # Filter for the specified year
    year_df = df[df['createdAt'].dt.year == year]
    
    # Group by week and keep track of first day of each week
    year_df['week'] = year_df['createdAt'].dt.to_period('W')
    year_df['month'] = year_df['createdAt'].dt.to_period('M')
    
    # Calculate metrics
    weekly_prs = year_df.groupby('week').size()
    weekly_contributors = year_df.groupby('week')['author_login'].nunique()
    weekly_ratio = weekly_prs / weekly_contributors
    
    # Get the first day of each week for x-axis
    week_dates = pd.Series([pd.Period(w).start_time for w in weekly_ratio.index])
    
    # Identify month boundaries
    month_starts = year_df.groupby('month')['createdAt'].min()
    month_labels = month_starts.dt.strftime('%b')
    
    return {
        'weekly_prs': weekly_prs,
        'weekly_contributors': weekly_contributors,
        'weekly_ratio': weekly_ratio,
        'week_dates': week_dates,
        'month_starts': month_starts,
        'month_labels': month_labels,
        'year': year
    }

def create_plot(metrics, repo_name):
    fig = go.Figure()
    
    # Add trace for primary year
    primary_data = metrics['primary']
    x_values = [d.timestamp() * 1000 for d in primary_data['week_dates']]
    fig.add_trace(go.Scatter(
        x=x_values,
        y=primary_data['weekly_ratio'].values,
        name=f'{primary_data["year"]} PRs per Contributor',
        mode='lines+markers',
        hovertemplate='%{y:.2f} PRs per Contributor<extra></extra>'
    ))
    
    # Add trace for comparison year if it exists
    if 'comparison' in metrics:
        comparison_data = metrics['comparison']
        x_values = [d.timestamp() * 1000 for d in comparison_data['week_dates']]
        # Adjust timestamps to overlay with primary year
        x_values = [x - ((comparison_data['year'] - primary_data['year']) * 365.25 * 24 * 60 * 60 * 1000) for x in x_values]
        fig.add_trace(go.Scatter(
            x=x_values,
            y=comparison_data['weekly_ratio'].values,
            name=f'{comparison_data["year"]} PRs per Contributor',
            mode='lines+markers',
            line=dict(dash='dash'),
            hovertemplate='%{y:.2f} PRs per Contributor<extra></extra>'
        ))
    
    # Use month labels from primary year for x-axis
    month_positions = [d.timestamp() * 1000 for d in primary_data['month_starts']]
    month_labels = primary_data['month_labels']
    
    fig.update_layout(
        title=f'Weekly PRs per Contributor - {repo_name}',
        yaxis_title='PRs per Contributor',
        hovermode='x unified',
        xaxis=dict(
            type='date',
            tickmode='array',
            ticktext=month_labels,
            tickvals=month_positions,
            tickformat='%b',
            tickangle=0
        )
    )
    
    return fig

@app.route('/')
def index():
    try:
        # Get the first repository's data to initialize the years list
        first_repo = next(iter(REPOSITORIES.keys()))
        df = get_repository_data(first_repo)
        available_years = sorted(df['createdAt'].dt.year.unique())
        current_year = datetime.now().year
        
        return render_template(
            'index.html',
            repositories=list(REPOSITORIES.keys()),
            years=available_years,
            current_year=current_year
        )
    except Exception as e:
        error_message = f"Error initializing dashboard: {str(e)}"
        print(error_message)
        return render_template(
            'error.html',
            error=error_message,
            repositories=list(REPOSITORIES.keys())
        )

@app.route('/get_years', methods=['POST'])
def get_years():
    try:
        repo_name = request.json.get('repository')
        df = get_repository_data(repo_name)
        # Convert numpy.int32 to native Python integers
        available_years = sorted(df['createdAt'].dt.year.unique().tolist())
        return jsonify({
            'years': available_years
        })
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/update_plot', methods=['POST'])
def update_plot():
    try:
        data = request.json
        repo_name = data.get('repository')
        df = get_repository_data(repo_name)
        
        metrics = process_pr_data(
            df,
            year=int(data.get('year')),
            comparison_year=int(data.get('comparison_year')) if data.get('comparison_year') else None
        )
        fig = create_plot(metrics, repo_name)
        
        # Calculate summary statistics for primary year
        primary_stats = {
            'year': metrics['primary']['year'],
            'avg_prs_per_week': float(metrics['primary']['weekly_prs'].mean()),
            'avg_contributors_per_week': float(metrics['primary']['weekly_contributors'].mean()),
            'avg_ratio': float(metrics['primary']['weekly_ratio'].mean()),
            'total_prs': int(metrics['primary']['weekly_prs'].sum()),
            'total_weeks': len(metrics['primary']['weekly_prs'])
        }
        
        # Calculate comparison statistics if available
        comparison_stats = None
        if 'comparison' in metrics:
            comparison_stats = {
                'year': metrics['comparison']['year'],
                'avg_prs_per_week': float(metrics['comparison']['weekly_prs'].mean()),
                'avg_contributors_per_week': float(metrics['comparison']['weekly_contributors'].mean()),
                'avg_ratio': float(metrics['comparison']['weekly_ratio'].mean()),
                'total_prs': int(metrics['comparison']['weekly_prs'].sum()),
                'total_weeks': len(metrics['comparison']['weekly_prs'])
            }
        
        return jsonify({
            'plot': json.loads(fig.to_json()),
            'primary_stats': primary_stats,
            'comparison_stats': comparison_stats
        })
    except Exception as e:
        print(f"Error in update_plot: {str(e)}")
        return jsonify({
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True)