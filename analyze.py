import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import argparse
from pathlib import Path
import numpy as np
from scipy import stats
import base64
from io import BytesIO

class GitHubPRAnalyzer:
    def __init__(self, input_file):
        """Initialize the analyzer with input file path."""
        self.df = self._load_data(input_file)
        self.df['year'] = pd.to_datetime(self.df['createdAt']).dt.year
        
    def _load_data(self, input_file):
        """Load JSON data into pandas DataFrame and handle missing values."""
        with open(input_file, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        
        # Fill missing values
        df['duration_hours'] = df['duration_hours'].fillna(0)
        df['additions'] = df['additions'].fillna(0)
        df['deletions'] = df['deletions'].fillna(0)
        df['changedFiles'] = df['changedFiles'].fillna(0)
        
        return df
    
    def get_unique_authors(self):
        """Get list of all unique authors, handling null values."""
        return sorted(self.df['author'].apply(lambda x: x['login'] if x is not None else 'Unknown').unique().tolist())
    
    def get_unique_years(self):
        """Get list of all unique years."""
        return sorted(self.df['year'].unique().tolist())
    
    def prepare_data_for_frontend(self):
        """Prepare all necessary data for the frontend."""
        data_dict = {
            'raw_data': self.df.to_dict(orient='records'),
            'unique_authors': self.get_unique_authors(),
            'unique_years': self.get_unique_years()
        }
        return json.dumps(data_dict)

    def generate_html_report(self):
        """Generate interactive HTML report."""
        data_json = self.prepare_data_for_frontend()
        
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>GitHub PR Analysis Report</title>
            <style>
                body { 
                    font-family: Arial, sans-serif; 
                    margin: 20px; 
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }
                .controls {
                    background-color: #f5f5f5;
                    padding: 20px;
                    border-radius: 5px;
                    margin-bottom: 20px;
                }
                .metric { 
                    background-color: #f5f5f5;
                    padding: 15px;
                    margin: 10px 0;
                    border-radius: 5px;
                }
                .plot {
                    margin: 20px 0;
                    text-align: center;
                }
                select, button {
                    padding: 5px;
                    margin: 5px;
                }
                .author-list {
                    max-height: 200px;
                    overflow-y: auto;
                    border: 1px solid #ddd;
                    padding: 10px;
                    margin: 10px 0;
                }
                .author-item {
                    display: flex;
                    align-items: center;
                    margin: 5px 0;
                }
                #charts {
                    display: flex;
                    flex-wrap: wrap;
                    justify-content: space-around;
                }
                .chart {
                    flex: 0 0 45%;
                    margin: 10px;
                }
            </style>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        </head>
        <body>
            <h1>GitHub PR Analysis Report</h1>
            
            <div class="controls">
                <h2>Filters</h2>
                <div>
                    <label for="yearSelect">Year:</label>
                    <select id="yearSelect">
                        <option value="all">All Years</option>
                    </select>
                </div>
                
                <div>
                    <h3>Exclude Authors</h3>
                    <div class="author-list" id="authorList">
                        <!-- Author checkboxes will be populated here -->
                    </div>
                </div>
            </div>
            
            <div class="metric" id="keyMetrics">
                <h2>Key Metrics</h2>
                <div id="metricsContent"></div>
            </div>
            
            <div id="charts">
                <div class="chart" id="prSizeDurationChart"></div>
                <div class="chart" id="monthlyPRsChart"></div>
                <div class="chart" id="weeklyAvgChart"></div>
            </div>

            <script>
                // Store the raw data
                const rawData = JSON.parse('DATA_PLACEHOLDER');
                
                // Initialize filters and charts
                function initializeFilters() {
                    // Populate year select
                    const yearSelect = document.getElementById('yearSelect');
                    rawData.unique_years.forEach(year => {
                        const option = document.createElement('option');
                        option.value = year;
                        option.textContent = year;
                        yearSelect.appendChild(option);
                    });
                    
                    // Populate author list
                    const authorList = document.getElementById('authorList');
                    rawData.unique_authors.forEach(author => {
                        const div = document.createElement('div');
                        div.className = 'author-item';
                        div.innerHTML = `
                            <input type="checkbox" id="author-${author}" value="${author}">
                            <label for="author-${author}">${author}</label>
                        `;
                        authorList.appendChild(div);
                    });
                    
                    // Add event listeners
                    yearSelect.addEventListener('change', updateAnalysis);
                    authorList.addEventListener('change', updateAnalysis);
                }
                
                function getFilteredData() {
                    let data = rawData.raw_data;
                    
                    // Filter by year
                    const selectedYear = document.getElementById('yearSelect').value;
                    if (selectedYear !== 'all') {
                        data = data.filter(pr => new Date(pr.createdAt).getFullYear() === parseInt(selectedYear));
                    }
                    
                    // Filter by excluded authors
                    const excludedAuthors = Array.from(document.querySelectorAll('#authorList input:checked'))
                        .map(checkbox => checkbox.value);
                    if (excludedAuthors.length > 0) {
                        data = data.filter(pr => !excludedAuthors.includes(pr.author.login));
                    }
                    
                    return data;
                }
                
                function getWeekNumber(date) {
                    const d = new Date(date);
                    d.setHours(0, 0, 0, 0);
                    d.setDate(d.getDate() + 3 - (d.getDay() + 6) % 7);
                    const week1 = new Date(d.getFullYear(), 0, 4);
                    return 1 + Math.round(((d - week1) / 86400000 - 3 + (week1.getDay() + 6) % 7) / 7);
                }

                function formatWeekKey(date) {
                    const d = new Date(date);
                    return `${d.getFullYear()}-W${String(getWeekNumber(d)).padStart(2, '0')}`;
                }

                function updateMetrics(data) {
                    const totalPRs = data.length;
                    const avgDuration = data.reduce((acc, pr) => acc + pr.duration_hours, 0) / totalPRs;
                    const avgSize = data.reduce((acc, pr) => acc + pr.additions + pr.deletions, 0) / totalPRs;
                    
                    document.getElementById('metricsContent').innerHTML = `
                        <p>Total PRs: ${totalPRs}</p>
                        <p>Average PR Duration: ${avgDuration.toFixed(2)} hours</p>
                        <p>Average PR Size: ${avgSize.toFixed(2)} changes</p>
                    `;
                }
                
                function updateCharts(data) {
                    // Process data by week
                    const weeklyStats = {};
                    data.forEach(pr => {
                        const weekKey = formatWeekKey(pr.createdAt);
                        if (!weeklyStats[weekKey]) {
                            weeklyStats[weekKey] = {
                                prCount: 0,
                                contributors: new Set(),
                                totalChanges: 0
                            };
                        }
                        weeklyStats[weekKey].prCount++;
                        if (pr.author && pr.author.login) {
                            weeklyStats[weekKey].contributors.add(pr.author.login);
                        }
                        weeklyStats[weekKey].totalChanges += (pr.additions || 0) + (pr.deletions || 0);
                    });

                    // Sort weeks chronologically
                    const sortedWeeks = Object.keys(weeklyStats).sort();
                    
                    // Weekly PR counts
                    const weeklyPRs = {
                        x: sortedWeeks,
                        y: sortedWeeks.map(week => weeklyStats[week].prCount),
                        type: 'bar',
                        name: 'PRs per Week'
                    };
                    
                    const prCountLayout = {
                        title: 'PRs per Week',
                        xaxis: { 
                            title: 'Week',
                            tickangle: -45
                        },
                        yaxis: { title: 'Number of PRs' }
                    };
                    
                    Plotly.newPlot('prSizeDurationChart', [weeklyPRs], prCountLayout);
                    
                    // Weekly contributor counts
                    const weeklyContributors = {
                        x: sortedWeeks,
                        y: sortedWeeks.map(week => weeklyStats[week].contributors.size),
                        type: 'bar',
                        name: 'Contributors per Week'
                    };
                    
                    const contributorLayout = {
                        title: 'Contributors per Week',
                        xaxis: { 
                            title: 'Week',
                            tickangle: -45
                        },
                        yaxis: { title: 'Number of Contributors' }
                    };
                    
                    Plotly.newPlot('monthlyPRsChart', [weeklyContributors], contributorLayout);

                    // Calculate 4-week moving average of PR counts
                    const movingAveragePRs = [];
                    const windowSize = 4;
                    
                    sortedWeeks.forEach((week, index) => {
                        if (index >= windowSize - 1) {
                            const windowTotal = sortedWeeks
                                .slice(index - (windowSize - 1), index + 1)
                                .reduce((sum, w) => sum + weeklyStats[w].prCount, 0);
                            movingAveragePRs.push(windowTotal / windowSize);
                        } else {
                            // For the first few weeks where we don't have enough data for a full window,
                            // calculate the average with available data
                            const windowTotal = sortedWeeks
                                .slice(0, index + 1)
                                .reduce((sum, w) => sum + weeklyStats[w].prCount, 0);
                            movingAveragePRs.push(windowTotal / (index + 1));
                        }
                    });

                    const avgPRsTrace = {
                        x: sortedWeeks,
                        y: movingAveragePRs,
                        type: 'scatter',
                        mode: 'lines',
                        name: '4-Week Moving Average',
                        line: {
                            width: 3
                        }
                    };

                    const avgLayout = {
                        title: '4-Week Moving Average of PRs',
                        xaxis: { 
                            title: 'Week',
                            tickangle: -45
                        },
                        yaxis: { 
                            title: 'Average Number of PRs',
                            rangemode: 'tozero'
                        }
                    };

                    Plotly.newPlot('weeklyAvgChart', [avgPRsTrace], avgLayout);
                }
                
                function updateAnalysis() {
                    const filteredData = getFilteredData();
                    updateMetrics(filteredData);
                    updateCharts(filteredData);
                }
                
                // Initialize the page
                initializeFilters();
                updateAnalysis();
            </script>
        </body>
        </html>
        """.replace('DATA_PLACEHOLDER', data_json.replace("'", "\\'"))
        
        return html_template

def main():
    parser = argparse.ArgumentParser(description='Analyze GitHub PR data')
    parser.add_argument('input_file', help='Path to input JSON file')
    parser.add_argument('output_dir', help='Directory for HTML report output')
    
    args = parser.parse_args()
    
    # Create analyzer instance
    analyzer = GitHubPRAnalyzer(args.input_file)
    
    # Generate report
    report_html = analyzer.generate_html_report()
    
    # Save report
    output_path = Path(args.output_dir) / f'pr_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(report_html)
    
    print(f"Report generated: {output_path}")

if __name__ == "__main__":
    main()