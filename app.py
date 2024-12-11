import json
from datetime import datetime, timedelta
import pandas as pd
from flask import Flask, render_template, jsonify, request
import plotly
import plotly.graph_objs as go
from collections import defaultdict

app = Flask(__name__)

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

def process_pr_data(df, start_date=None, end_date=None):
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # Exclude bot authors
    bot_authors = ['renovate', 'dependabot']
    df = df[~df['author_login'].isin(bot_authors)]
    
    # Apply date filters
    if start_date:
        start = pd.to_datetime(start_date).tz_localize('UTC')
        df = df[df['createdAt'] >= start]
    if end_date:
        end = pd.to_datetime(end_date).tz_localize('UTC')
        df = df[df['createdAt'] <= end]
    
    # Group by week
    df['week'] = df['createdAt'].dt.to_period('W')
    
    # Sort weeks and create relative week numbers
    weekly_prs = df.groupby('week').size()
    weekly_contributors = df.groupby('week')['author_login'].nunique()
    weekly_ratio = weekly_prs / weekly_contributors
    
    # Create relative week numbers (1-based)
    weeks = sorted(weekly_ratio.index)
    week_map = {week: i+1 for i, week in enumerate(weeks)}
    
    return {
        'weekly_prs': weekly_prs,
        'weekly_contributors': weekly_contributors,
        'weekly_ratio': weekly_ratio,
        'week_map': week_map
    }

def create_plot(metrics):
    fig = go.Figure()
    
    # Get x-axis values as relative week numbers
    week_numbers = [metrics['week_map'][week] for week in metrics['weekly_ratio'].index]
    
    # Add only the PRs per contributor trace
    fig.add_trace(go.Scatter(
        x=week_numbers,
        y=metrics['weekly_ratio'].values,
        name='PRs per Contributor',
        mode='lines+markers',
        hovertemplate='Week %{x}<br>PRs per Contributor: %{y:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Weekly PRs per Contributor',
        xaxis_title='Week Number',
        yaxis_title='PRs per Contributor',
        hovermode='x unified',
        xaxis=dict(
            tickmode='linear',
            tick0=1,
            dtick=1
        )
    )
    
    return fig

@app.route('/')
def index():
    df = load_pr_data('pr_data.json')
    return render_template(
        'index.html',
        min_date=df['createdAt'].min().tz_localize(None).strftime('%Y-%m-%d'),
        max_date=df['createdAt'].max().tz_localize(None).strftime('%Y-%m-%d')
    )

@app.route('/update_plot', methods=['POST'])
def update_plot():
    try:
        data = request.json
        df = load_pr_data('pr_data.json')
        metrics = process_pr_data(
            df,
            start_date=data.get('start_date'),
            end_date=data.get('end_date')
        )
        fig = create_plot(metrics)
        
        # Calculate summary statistics
        summary_stats = {
            'avg_prs_per_week': float(metrics['weekly_prs'].mean()),
            'avg_contributors_per_week': float(metrics['weekly_contributors'].mean()),
            'avg_ratio': float(metrics['weekly_ratio'].mean()),
            'total_prs': int(metrics['weekly_prs'].sum()),
            'total_weeks': len(metrics['weekly_prs'])
        }
        
        return jsonify({
            'plot': json.loads(fig.to_json()),
            'stats': summary_stats
        })
    except Exception as e:
        print(f"Error in update_plot: {str(e)}")
        return jsonify({
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True)