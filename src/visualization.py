import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pandas as pd
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def check_dirs():
    os.makedirs(os.path.join(PROJECT_ROOT, 'results', 'plots'), exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, 'results', 'premium_reports'), exist_ok=True)

def plot_risk_distribution(df, save_path=None):
    if save_path is None: save_path = os.path.join(PROJECT_ROOT, 'results', 'plots', 'risk_distribution.png')
    """Plots the distribution of driver risk components."""
    plt.figure(figsize=(10, 6))
    sns.histplot(df['driver_risk_index'], bins=30, kde=True, color='purple')
    plt.title('Distribution of Driver Risk Index')
    plt.xlabel('Overarching Risk Index')
    plt.ylabel('Frequency')
    plt.savefig(save_path)
    plt.close()

def plot_premium_distribution(df, save_path=None):
    if save_path is None: save_path = os.path.join(PROJECT_ROOT, 'results', 'plots', 'premium_distribution.png')
    """Plots the distribution of calculated final premiums."""
    plt.figure(figsize=(10, 6))
    sns.histplot(df['final_premium'], bins=30, kde=True, color='green')
    plt.title('Distribution of Calculated Insurance Premiums (INR)')
    plt.xlabel('Premium (INR)')
    plt.ylabel('Frequency')
    plt.savefig(save_path)
    plt.close()

def plot_fraud_anomalies(df):
    """Generates an interactive plotly scatter plot highlighting frauds."""
    # This creates an interactive html file rather than static png
    fig = px.scatter(df, x='accidents_last_2yr', y='simulated_severity', 
                     color='anomaly_flag', 
                     hover_data=['driver_id', 'vehicle_age'],
                     color_continuous_scale=[(0, "blue"), (1, "red")],
                     title='Fraud Detection Outliers (Isolation Forest)')
                     
    fig.write_html(os.path.join(PROJECT_ROOT, 'results', 'plots', 'fraud_detection.html'))

def plot_scenario_comparison(base_df, sim_df, metric='final_premium', save_path=None):
    if save_path is None: save_path = os.path.join(PROJECT_ROOT, 'results', 'plots', 'scenario_comparison.png')
    """Side by side comparison of a metric between baseline and simulated stress test."""
    
    comparative_df = pd.DataFrame({
        'Baseline': base_df[metric],
        'Urban Simulation': sim_df[metric]
    })
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=comparative_df)
    plt.title(f'Impact of Urban High-Congestion on {metric.replace("_", " ").title()}')
    plt.ylabel(metric.replace("_", " ").title())
    plt.savefig(save_path)
    plt.close()
    
def generate_reports(df):
    """Outputs a text/csv premium report table."""
    check_dirs()
    report_df = df[['driver_id', 'driver_risk_index', 'risk_category', 'final_premium']]
    report_df = report_df.sort_values(by='final_premium', ascending=False)
    out_path = os.path.join(PROJECT_ROOT, 'results', 'premium_reports', 'top_premiums.csv')
    report_df.to_csv(out_path, index=False)
    print(f"Saved premium report to {out_path}")
