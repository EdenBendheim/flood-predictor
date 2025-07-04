#!/usr/bin/env python3
"""
Create a flood visualization map with rainfall background, matching the reference style.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
from datetime import datetime
import glob
import os

def find_peak_flood_day():
    """Find the day with the most floods between 2012-2020."""
    print("Loading USFD flood data...")
    
    # Load flood data
    flood_df = pd.read_csv('../FloodPredictor/USFD_v1.0.csv')
    
    # Convert dates
    flood_df['DATE_BEGIN'] = pd.to_datetime(flood_df['DATE_BEGIN'], format='%Y%m%d%H%M', errors='coerce')
    flood_df = flood_df.dropna(subset=['DATE_BEGIN', 'LON', 'LAT'])
    
    # Filter for 2012-2020
    flood_df = flood_df[
        (flood_df['DATE_BEGIN'].dt.year >= 2012) & 
        (flood_df['DATE_BEGIN'].dt.year <= 2020)
    ]
    
    # Count floods per day
    flood_df['date'] = flood_df['DATE_BEGIN'].dt.date
    daily_counts = flood_df.groupby('date').size()
    
    # Find peak day
    peak_date = daily_counts.idxmax()
    peak_count = daily_counts.max()
    
    print(f"Peak flood day: {peak_date} with {peak_count} floods")
    
    # Get floods for peak day
    peak_floods = flood_df[flood_df['date'] == peak_date]
    
    return peak_date, peak_floods

def create_professional_flood_map(peak_date, peak_floods):
    """Create a professional flood map matching the reference style."""
    
    # Try to load WLDAS data for that date
    date_str = peak_date.strftime('%Y%m%d')
    wldas_pattern = f"../WLDAS/WLDAS_NOAHMP001_DA1_{date_str}*.nc"
    wldas_files = glob.glob(wldas_pattern)
    
    # Create figure with cartopy projection
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Set extent to continental US
    ax.set_extent([-125, -66.5, 20, 50], ccrs.PlateCarree())
    
    # Add map features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, color='black')
    ax.add_feature(cfeature.BORDERS, linewidth=0.8, color='black')
    ax.add_feature(cfeature.STATES, linewidth=0.5, color='black', alpha=0.8)
    ax.add_feature(cfeature.OCEAN, color='lightgray', alpha=0.3)
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False,
                      linewidth=0.5, color='gray', alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False
    
    if wldas_files:
        print(f"Loading WLDAS data from {wldas_files[0]}")
        
        # Load WLDAS data
        ds = xr.open_dataset(wldas_files[0])
        
        # Get rainfall data (try different variable names)
        rainfall_var = None
        for var in ['Rainf_tavg', 'RAINRATE', 'Rainf', 'precip']:
            if var in ds.variables:
                rainfall_var = var
                break
        
        if rainfall_var:
            rainfall = ds[rainfall_var]
            if len(rainfall.shape) == 3:  # time, lat, lon
                rainfall = rainfall[0]
            
            # Create rainfall background with green colormap like reference
            im = ax.contourf(ds.lon, ds.lat, rainfall, 
                           levels=20, 
                           cmap='Greens', 
                           alpha=0.8,
                           transform=ccrs.PlateCarree())
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
            cbar.set_label(f'Rainfall (mm/hr)', rotation=270, labelpad=15)
        
        ds.close()
    else:
        print("No WLDAS data found, creating map without rainfall background")
        # Add a light background
        ax.add_feature(cfeature.LAND, color='lightgreen', alpha=0.3)
    
    # Plot flood points as red dots
    if len(peak_floods) > 0:
        ax.scatter(peak_floods['LON'], peak_floods['LAT'], 
                  c='red', s=20, alpha=0.8, 
                  transform=ccrs.PlateCarree(),
                  label=f'Floods ({len(peak_floods)} events)')
        
        # Add legend
        ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    
    # Set title
    plt.title(f'Flood Events and Rainfall - {peak_date}', 
              fontsize=14, fontweight='bold', pad=20)
    
    # Save the map
    output_file = f'flood_map_{peak_date}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print(f"Map saved as {output_file}")
    plt.show()

def main():
    """Main function to create the flood visualization."""
    
    # Find peak flood day
    peak_date, peak_floods = find_peak_flood_day()
    
    # Create the map
    create_professional_flood_map(peak_date, peak_floods)

if __name__ == "__main__":
    main()