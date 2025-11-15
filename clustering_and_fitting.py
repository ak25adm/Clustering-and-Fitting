"""
This is the template file for the clustering and fitting assignment.
You will be expected to complete all the sections and
make this a fully working, documented file.
You should NOT change any function, file or variable names,
 if they are given to you here.
Make use of the functions presented in the lectures
and ensure your code is PEP-8 compliant, including docstrings.
Fitting should be done with only 1 target variable and 1 feature variable,
likewise, clustering should be done with only 2 variables.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

def plot_relational_plot(df):
    """
    Create a relational plot showing the latitude vs the longitude of cities.
    Parameters:
    df (pd.DataFrame): DataFrame containing city data with coordinates
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Filter out rows with missing coordinates
    plot_data = df.dropna(subset=['Latitude', 'Longitude'])
    
    # Create a scatter plot of cities
    scatter = ax.scatter(plot_data['Longitude'], plot_data['Latitude'], 
                        alpha=0.7, s=50, c='blue', edgecolors='black', 
                        linewidth=0.5)
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Geographical Distribution of Pakistani Cities')
    ax.grid(True, alpha=0.3)
    
    # Add some major city labels
    major_cities = ['Karachi', 'Lahore', 'Islamabad', 'Quetta', 'Peshawar']
    for _, city in plot_data[plot_data['City'].isin(major_cities)].iterrows():
        ax.annotate(city['City'], (city['Longitude'], city['Latitude']),
                   xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('relational_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    return


def plot_categorical_plot(df):
    """
    Create a categorical plot showing city distribution by province.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing city data with province information
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Count cities by province, excluding missing values
    province_counts = df['Province/State'].value_counts().head(10)
    
    # Create horizontal bar plot
    bars = ax.barh(province_counts.index, province_counts.values, 
                   color='skyblue', edgecolor='navy', alpha=0.8)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                f'{int(width)}', ha='left', va='center', fontweight='bold')
    
    ax.set_xlabel('Number of Cities')
    ax.set_ylabel('Province/State')
    ax.set_title('Distribution of Cities by Province/State (Top 10)')
    ax.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('categorical_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    return


def plot_statistical_plot(df):
    """
    Create a statistical plot showing distribution of coordinates.
   
    Parameters:
    df (pd.DataFrame): DataFrame containing city coordinate data
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Filter data
    plot_data = df.dropna(subset=['Latitude', 'Longitude'])
    
    # Latitude distribution
    axes[0].hist(plot_data['Latitude'], bins=15, color='lightcoral', 
                alpha=0.7, edgecolor='darkred')
    axes[0].set_xlabel('Latitude')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Latitude Values')
    axes[0].grid(True, alpha=0.3)
    
    # Longitude distribution
    axes[1].hist(plot_data['Longitude'], bins=15, color='lightgreen', 
                alpha=0.7, edgecolor='darkgreen')
    axes[1].set_xlabel('Longitude')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of Longitude Values')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('statistical_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    return


def statistical_analysis(df, col: str):
    """
    Calculate statistical moments for a given column.
    """
    # Remove missing values
    data = df[col].dropna()

    mean = np.mean(data)
    stddev = np.std(data, ddof=1)
    skew = ss.skew(data)
    excess_kurtosis = ss.kurtosis(data) 
    
    return mean, stddev, skew, excess_kurtosis


def preprocessing(df):
    # You should preprocess your data in this function and
    # make use of quick features such as 'describe', 'head/tail' and 'corr'.
    print("Dataset Shape:", df.shape)
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\nDataset Info:")
    print(df.info())
    
    print("\nBasic Statistics:")
    print(df.describe())
    
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    # Clean the Population column - convert to numeric, handle errors
    df['Population'] = pd.to_numeric(df['Population'], errors='coerce')
    
    # Clean numeric columns
    numeric_columns = ['Population', 'Area (km²)', 'Latitude', 'Longitude']
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors='coerce')
    
    print("\nCorrelation Matrix (numeric columns):")
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        print(numeric_df.corr())
    
    return df


def writing(moments, col):
    print(f'For the attribute {col}:')
    print(f'Mean = {moments[0]:.2f}, '
          f'Standard Deviation = {moments[1]:.2f}, '
          f'Skewness = {moments[2]:.2f}, and '
          f'Excess Kurtosis = {moments[3]:.2f}.')
    
    # Determine skewness description
    skew_val = moments[2]
    if skew_val > 0.5:
        skew_desc = "right"
    elif skew_val < -0.5:
        skew_desc = "left"
    else:
        skew_desc = "not"
    
    # Determine kurtosis description
    kurt_val = moments[3]
    if kurt_val > 1:
        kurt_desc = "leptokurtic"
    elif kurt_val < -1:
        kurt_desc = "platykurtic"
    else:
        kurt_desc = "mesokurtic"
     # Delete the following options as appropriate for your data.
    # Not skewed and mesokurtic can be defined with asymmetries <-2 or >2.
    print(f'The data was {skew_desc} skewed and {kurt_desc}.')

    return


def perform_clustering(df, col1, col2):

    def plot_elbow_method(scaled_data, max_k=10):
        """
        Plot elbow method to determine optmal number of clusters.
        Parameters:
        scaled_data (np.array): Scaled data for clustering
        max_k (int): Maximum number of clusters for test
        Returns:
        list: Inertia values for each k
        """

        inertias = []
        k_range = range(1, max_k + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(scaled_data)
            inertias.append(kmeans.inertia_)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
        ax.set_xlabel('Number of Clusters (k)')
        ax.set_ylabel('Inertia')
        ax.set_title('Elbow Method for Optimal k')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('elbow_plot.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
      
        return inertias

    def one_silhouette_inertia(scaled_data, k=3):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(scaled_data)
        
        _score = silhouette_score(scaled_data, labels)
        _inertia = kmeans.inertia_
        
        return _score, _inertia

    # Gather data and scale
    data = df[[col1, col2]].dropna()
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    # Find best number of clusters using elbow method
    inertias = plot_elbow_method(scaled_data)
    
    # Calculate silhouette scores for different k values
    k_values = [2, 3, 4, 5]
    best_k = 3
    best_score = -1
    
    print("\nClustering Analysis:")
    for k in k_values:
        score, inertia = one_silhouette_inertia(scaled_data, k)
        print(f"k={k}: Silhouette Score = {score:.3f}, Inertia = {inertia:.3f}")
        if score > best_score:
            best_score = score
            best_k = k
    
    print(f"Optimal number of clusters: {best_k}")
    
    # Perform final clustering with optimal k
    final_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = final_kmeans.fit_predict(scaled_data)
    
    # Get cluster centers in original scale
    cluster_centers = scaler.inverse_transform(final_kmeans.cluster_centers_)
    xkmeans = cluster_centers[:, 0]
    ykmeans = cluster_centers[:, 1]
    
    cenlabels = [f'Cluster {i+1}' for i in range(best_k)]
    
    # Get cluster centers
    return labels, data.values, xkmeans, ykmeans, cenlabels


def plot_clustered_data(labels, data, xkmeans, ykmeans, centre_labels):
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create scatter plot with different colors for each cluster
    scatter = ax.scatter(data[:, 0], data[:, 1], c=labels, 
                        cmap='viridis', alpha=0.7, s=60, 
                        edgecolors='black', linewidth=0.5)
    
    # Plot cluster centers
    ax.scatter(xkmeans, ykmeans, c='red', marker='X', s=200, 
              edgecolors='black', linewidth=2, label='Cluster Centers')
    
    # Add labels to cluster centers
    for i, label in enumerate(centre_labels):
        ax.annotate(label, (xkmeans[i], ykmeans[i]), 
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=12, fontweight='bold', 
                   bbox=dict(boxstyle="round,pad=0.3", 
                           facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('K-means Clustering of Pakistani Cities by Geographic Location')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    plt.colorbar(scatter, ax=ax, label='Cluster')
    
    plt.tight_layout()
    plt.savefig('clustering.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    return


def perform_fitting(df, col1, col2):
    # Gather data and prepare for fitting
    data = df[[col1, col2]].dropna()
    x_data = data[col1].values.reshape(-1, 1)
    y_data = data[col2].values

    # Fit model
    degree = 2
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(x_data, y_data)

    # Predict across x
    x = np.linspace(x_data.min(), x_data.max(), 100).reshape(-1, 1)
    y = model.predict(x)
    
    r_squared = model.score(x_data, y_data)
    print(f"\nFitting Results (Polynomial Degree {degree}):")
    print(f"R-squared: {r_squared:.3f}")
    
    return data.values, x, y


def plot_fitted_data(data, x, y):
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot original data
    ax.scatter(data[:, 0], data[:, 1], alpha=0.7, s=50, 
              color='blue', edgecolors='black', linewidth=0.5, 
              label='Cities')
    
    # Plot fitted curve
    ax.plot(x, y, color='red', linewidth=3, 
            label='Fitted Polynomial Curve')
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Polynomial Regression Fit: Latitude vs Longitude of Pakistani Cities')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fitting.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    return


def main():
    df = pd.read_csv('data.csv')
    df = preprocessing(df)
    # Choose Latitude for statistical analysis
    col = 'Latitude'
    plot_relational_plot(df)
    plot_statistical_plot(df)
    plot_categorical_plot(df)
    moments = statistical_analysis(df, col)
    writing(moments, col)
    # Perform clustering on geographic coordinates
    clustering_results = perform_clustering(df, 'Longitude', 'Latitude')
    plot_clustered_data(*clustering_results)
    # Perform fitting on geographic coordinates
    fitting_results = perform_fitting(df, 'Longitude', 'Latitude')
    plot_fitted_data(*fitting_results)
    return


if __name__ == '__main__':
    main()
