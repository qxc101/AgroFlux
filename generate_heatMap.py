import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.cm import ScalarMappable
import os
import requests
from zipfile import ZipFile
from io import BytesIO
import ssl
import argparse

# Fix for SSL issues
try:
    import geopandas as gpd
    from shapely.geometry import Point
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False

def download_county_boundaries():
    """Download county boundaries shapefile from the US Census Bureau if not present"""
    county_shapefile = "counties/tl_2020_us_county.shp"
    
    # Check if the file already exists
    if os.path.exists(county_shapefile):
        print("County boundaries shapefile already exists.")
        return county_shapefile
    
    print("Downloading county boundaries from Census Bureau...")
    # URL for 2020 county boundaries
    url = "https://www2.census.gov/geo/tiger/TIGER2020/COUNTY/tl_2020_us_county.zip"
    
    try:
        # Create directory if it doesn't exist
        os.makedirs("counties", exist_ok=True)
        
        # Download the zip file with SSL verification disabled (for environments with SSL issues)
        response = requests.get(url, verify=False)
        
        # Extract the contents
        with ZipFile(BytesIO(response.content)) as zip_file:
            zip_file.extractall("counties")
        
        print("County boundaries downloaded and extracted successfully.")
        return county_shapefile
    
    except Exception as e:
        print(f"Error downloading county boundaries: {e}")
        return None

def create_simple_heatmap(df, metric='R2'):
    """
    Create a basic heat map of counties without external dependencies
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing county data
    metric : str, default='R2'
        Metric to display in the heatmap (e.g., 'R2', 'RMSE', 'MAE')
    """
    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Set color scale based on selected metric
    norm = mcolors.Normalize(vmin=df[metric].min(), vmax=df[metric].max())
    
    # Choose colormap based on metric type
    # For metrics where higher is better (like R2), use viridis
    # For metrics where lower is better (like RMSE, MAE), use viridis_r (reversed)
    if metric in ['R2']:
        cmap = plt.cm.viridis  # Higher values are better
    else:
        cmap = plt.cm.viridis_r  # Lower values are better for RMSE, MAE
    
    # Group counties by state
    state_counties = df.groupby('STATE_NAME')
    
    # Plot points for each county
    for state_name, counties in state_counties:
        for _, county in counties.iterrows():
            # Create a circle for each county
            circle = plt.Circle(
                (county['Lon'], county['Lat']), 
                0.2,  # Radius
                color=cmap(norm(county[metric])),  # Using selected metric for coloring
                alpha=0.8,
                edgecolor='black',
                linewidth=0.5
            )
            ax.add_patch(circle)
            
            # Optionally add county names if not too many
            if len(df) <= 50:
                ax.text(county['Lon'], county['Lat'], county['NAME'], 
                        fontsize=6, ha='center', color='black')
    
    # Add state boundaries approximation
    state_boundaries = {
        'Illinois': [(-91.5, 37.0), (-91.5, 43.0), (-87.5, 43.0), (-87.5, 37.0)],
        'Indiana': [(-88.1, 37.8), (-88.1, 41.8), (-84.8, 41.8), (-84.8, 37.8)],
        'Iowa': [(-96.6, 40.4), (-96.6, 43.5), (-91.0, 43.5), (-91.0, 40.4)]
    }
    
    # Draw approximate state boundaries
    for state, coords in state_boundaries.items():
        x = [p[0] for p in coords]
        y = [p[1] for p in coords]
        x.append(coords[0][0])  # Close the polygon
        y.append(coords[0][1])
        ax.plot(x, y, color='black', linewidth=1.5)
    
    # Add state labels
    state_positions = {
        'Illinois': (-89.4, 40.0),
        'Indiana': (-86.3, 39.8),
        'Iowa': (-93.5, 42.0)
    }
    
    for state, pos in state_positions.items():
        ax.text(pos[0], pos[1], state, fontsize=12, ha='center', va='center', 
                fontweight='bold', bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'))
    
    # Set map extent
    buffer = 1.0
    min_lon = df['Lon'].min() - buffer
    max_lon = df['Lon'].max() + buffer
    min_lat = df['Lat'].min() - buffer
    max_lat = df['Lat'].max() + buffer
    ax.set_xlim(min_lon, max_lon)
    ax.set_ylim(min_lat, max_lat)
    
    # Add colorbar
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    # When creating labels for the colorbar:
    cbar.set_label(f'${metric.replace("R2", "R^2")}$ Value')

    # For titles:
    ax.set_title(f'County ${metric.replace("R2", "R^2")}$ Value Heat Map (IL, IN, IA)', fontsize=16)
    
    # Create appropriate subtitle based on the metric
    if metric in ['R2']:
        subtitle = f'Darker colors represent higher {metric} values'
    else:
        subtitle = f'Darker colors represent lower {metric} values'
    
    plt.figtext(0.5, 0.02, subtitle, fontsize=12, ha='center')
    
    # Create a legend for states
    state_patches = []
    for state in ['Illinois', 'Indiana', 'Iowa']:
        patch = mpatches.Patch(color='white', edgecolor='black', label=state)
        state_patches.append(patch)
    ax.legend(handles=state_patches, loc='upper right')
    
    # Remove axes ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    return fig

def create_geodataframe_map(df, metric='R2'):
    """
    Create a map using GeoPandas if available
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing county data
    metric : str, default='R2'
        Metric to display in the heatmap (e.g., 'R2', 'RMSE', 'MAE')
    """
    # Import required GeoPandas and shapely modules
    import geopandas as gpd
    from shapely.geometry import Point
    
    # Download county boundaries if needed
    county_shapefile = download_county_boundaries()
    
    if county_shapefile is None or not os.path.exists(county_shapefile):
        print("Could not get county boundaries. Using simplified visualization.")
        return create_simple_heatmap(df, metric)
    
    # Read all US counties
    counties = gpd.read_file(county_shapefile)
    
    # Convert FIPS values to integers for matching
    df['FIPS'] = df['FIPS'].astype(int)
    counties['GEOID'] = counties['GEOID'].astype(int)
    
    # Filter to our three states
    states = df['STATE_NAME'].unique()
    state_fips = {
        'Illinois': '17',
        'Indiana': '18',
        'Iowa': '19'
    }
    
    # Create a mask for the three states
    state_mask = counties['STATEFP'].isin([state_fips[state] for state in states])
    three_states = counties[state_mask].copy()
    
    # Set up figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # First, draw all counties in the three states with light gray
    three_states.plot(ax=ax, color='lightgray', edgecolor='darkgray', linewidth=0.5)
    
    # Merge metric values from our dataframe to the GeoPandas dataframe
    # Create a dictionary mapping FIPS to the selected metric
    fips_to_metric = dict(zip(df['FIPS'], df[metric]))
    
    # Add metric column to the GeoPandas dataframe
    three_states[metric] = three_states['GEOID'].map(fips_to_metric)
    
    # Create a mask for counties in our dataset (those with metric values)
    in_dataset_mask = three_states['GEOID'].isin(df['FIPS'])
    dataset_counties = three_states[in_dataset_mask].copy()
    
    # Choose colormap based on metric type
    # For metrics where higher is better (like R2), use viridis
    # For metrics where lower is better (like RMSE, MAE), use viridis_r (reversed)
    if metric in ['R2']:
        cmap = plt.cm.viridis  # Higher values are better
    else:
        cmap = plt.cm.viridis_r  # Lower values are better for RMSE, MAE
    
    # Set color scale for metric values
    norm = mcolors.Normalize(vmin=df[metric].min(), vmax=df[metric].max())
    
    # Plot counties in our dataset with color based on selected metric
    dataset_counties.plot(ax=ax, column=metric, cmap=cmap, 
                         edgecolor='black', linewidth=0.7, norm=norm)
    
    # Add state boundaries with thicker lines
    for state in state_fips.keys():
        state_mask = three_states['STATEFP'] == state_fips[state]
        state_boundary = three_states[state_mask].dissolve()
        state_boundary.boundary.plot(ax=ax, color='black', linewidth=1.5)
    
    # Add state labels
    state_positions = {
        'Illinois': (-89.4, 40.0),
        'Indiana': (-86.3, 39.8),
        'Iowa': (-93.5, 42.0)
    }
    
    for state, pos in state_positions.items():
        ax.text(pos[0], pos[1], state, fontsize=12, ha='center', va='center', 
                fontweight='bold', bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'))
    
    # Add colorbar
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(f'${metric.replace("R2", "R^2")}$ Value')

    # For titles:
    ax.set_title(f'County ${metric.replace("R2", "R^2")}$ Value Heat Map (IL, IN, IA)', fontsize=16)
    
    # Create appropriate subtitle based on the metric
    # if metric in ['R2']:
    #     subtitle = f'Darker colors represent higher {metric} values'
    # else:
    #     subtitle = f'Darker colors represent lower {metric} values'
    subtitle = " "

    plt.figtext(0.5, 0.02, subtitle, fontsize=12, ha='center')
    
    # Focus the map on our three states
    ax.set_xlim(three_states.total_bounds[0], three_states.total_bounds[2])
    ax.set_ylim(three_states.total_bounds[1], three_states.total_bounds[3])
    
    # Remove axes ticks for a cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    
    return fig

def create_county_heatmap(csv_file, csv_file_metrics, metric='R2'):
    """
    Main function to create a heat map of counties colored by selected metric values
    
    Parameters:
    -----------
    csv_file : str
        Path to CSV file with county information
    csv_file_metrics : str
        Path to CSV file with metrics (R2, RMSE, MAE)
    metric : str, default='R2'
        Metric to display in the heatmap (e.g., 'R2', 'RMSE', 'MAE')
    """
    # Read the CSV files
    df = pd.read_csv(csv_file)
    df_metrics = pd.read_csv(csv_file_metrics)
    
    # Add metrics to the dataframe
    df['R2'] = df_metrics['R2'].values
    df['RMSE'] = df_metrics['RMSE'].values
    df['MAE'] = df_metrics['MAE'].values
    
    print(f"Creating heatmap with {metric} values")
    print(df)
    
    # Disable SSL certificate warnings (not recommended for production)
    requests.packages.urllib3.disable_warnings()
    
    # Check if GeoPandas is available and try to use it
    if HAS_GEOPANDAS:
        try:
            return create_geodataframe_map(df, metric)
        except Exception as e:
            print(f"Error creating map with GeoPandas: {e}")
            print("Falling back to simplified map...")
    
    # Fallback to simple map if GeoPandas is not available or fails
    return create_simple_heatmap(df, metric)

# Usage
if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Create county heatmap with selectable metrics')
    parser.add_argument('--metric', type=str, default='R2', choices=['R2', 'RMSE', 'MAE'],
                        help='Metric to display in the heatmap (R2, RMSE, or MAE)')
    parser.add_argument('--county_file', type=str, default="county_metrics/Ecosys_99points.csv",
                        help='Path to county information CSV file')
    parser.add_argument('--metrics_file', type=str, 
                        default="county_metrics/t0_mw_All_lstm_temporal_0_county_metrics.csv",
                        help='Path to metrics CSV file')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file name (default: county_METRIC_heatmap.png)')
    
    args = parser.parse_args()
    
    # Create the heatmap with the selected metric
    fig = create_county_heatmap(args.county_file, args.metrics_file, args.metric)
    
    # Default output filename based on the metric if not specified
    if args.output is None:
        source_file = args.metrics_file.replace("county_metrics/", "").replace("_county_metrics.csv", "")
        output_file = f"heat_maps/{source_file}_{args.metric.lower()}_heatmap.png"
    else:
        output_file = args.output
    
    # Save and display the figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved heatmap to {output_file}")
    plt.show()