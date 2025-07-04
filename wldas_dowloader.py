#!/usr/bin/env python3
# filepath: /Users/edenbendheim/Dropbox/wildfire_spread_prediction_MLOS2/Caldor Paper/wldas_downloader.py

import os
import sys
import time
import requests
import getpass
from urllib.parse import urlparse, parse_qs, unquote
from concurrent.futures import ThreadPoolExecutor
from rich.progress import Progress, TextColumn, BarColumn, DownloadColumn, TimeElapsedColumn

# Constants
TXTFILE = "subset_WLDAS_NOAHMP001_DA1_D1.0_20250704_230939_.txt"
OUTDIR = "WLDAS"
TIMEOUT = 60  # Longer timeout for NASA servers
MAX_RETRIES = 10  # More retries for problematic files
RETRY_DELAY = 5  # Base delay between retries (will increase with backoff)
CONCURRENT_DOWNLOADS = 32  # Number of concurrent downloads

def get_session():
    """Create authenticated session for NASA EarthData Login."""
    print("NASA EarthData Login required")
    username = input("Username: ")
    password = getpass.getpass("Password: ")
    
    s = requests.Session()
    s.auth = (username, password)
    s.headers.update({
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:91.0) Gecko/20100101 Firefox/91.0'
    })
    s.allow_redirects = True
    return s

def get_filename(url):
    """Extract filename from URL parameters."""
    try:
        params = parse_qs(urlparse(url).query)
        if 'LABEL' in params:
            # Fix the double .nc extension issue
            filename = params['LABEL'][0].replace('.SUB.nc4', '.nc')
            # Ensure we don't end up with .nc.nc
            if filename.endswith('.nc.nc'):
                filename = filename[:-3]  # remove the trailing .nc
            return filename
        elif 'FILENAME' in params:
            basename = os.path.basename(unquote(params['FILENAME'][0]))
            # Also clean up any potential double extension here
            if basename.endswith('.nc.nc'):
                basename = basename[:-3]
            return basename
        
        path = os.path.basename(urlparse(url).path)
        if path.endswith('.nc.nc'):
            path = path[:-3]
        return path
    except:
        return f"wldas_file_{abs(hash(url)) % 10000}.nc"

def download_file(session, url, progress):
    """Download a single file by loading it into memory first."""
    filename = get_filename(url)
    outpath = os.path.join(OUTDIR, filename)
    temppath = outpath + ".part"
    
    # Skip if already exists
    if os.path.exists(outpath) and os.path.getsize(outpath) > 0:
        print(f"Skipping {filename} (already exists)")
        return "skipped"
    
    task_id = progress.add_task(f"[cyan]{filename}", total=100)

    # Retry logic
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # Make request without streaming, load entire file into memory
            response = session.get(url, timeout=TIMEOUT)
            response.raise_for_status()
            
            total_size = len(response.content)
            progress.update(task_id, total=total_size)
            
            # Write the entire content at once to a temporary file
            with open(temppath, 'wb') as f:
                f.write(response.content)
            
            # Atomically move to the final location
            os.rename(temppath, outpath)
            
            progress.update(task_id, completed=total_size, description=f"[green]{filename}")
            return "success"
            
        except requests.exceptions.RequestException as e:
            wait_time = RETRY_DELAY * attempt
            
            # Clean up partial download
            if os.path.exists(temppath):
                os.remove(temppath)
            
            progress.update(task_id, 
                           description=f"[yellow]Retrying {filename} ({attempt}/{MAX_RETRIES})")
            
            if attempt < MAX_RETRIES:
                print(f"\nError downloading {filename}: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"\nFailed to download {filename} after {MAX_RETRIES} attempts: {e}")
                progress.update(task_id, description=f"[red]Failed {filename}")
                return "failed"

def main():
    # Make sure output directory exists
    os.makedirs(OUTDIR, exist_ok=True)
    
    # Setup authentication
    session = get_session()
    
    # Read URLs from file
    with open(TXTFILE) as f:
        urls = [line.strip() for line in f if line.strip() and not line.startswith("//")]
    
    print(f"Found {len(urls)} files to download")
    
    # Create progress display
    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=50),
        "[progress.percentage]{task.percentage:>3.1f}%",
        DownloadColumn(),
        TimeElapsedColumn(),
    ) as progress:
        
        # Process files concurrently
        results = []
        with ThreadPoolExecutor(max_workers=CONCURRENT_DOWNLOADS) as executor:
            # Submit all download tasks
            future_to_url = {
                executor.submit(download_file, session, url, progress): url 
                for url in urls
            }
            
            # Collect results as they complete
            for future in future_to_url:
                try:
                    results.append(future.result())
                except Exception as exc:
                    print(f"Download generated an exception: {exc}")
                    results.append("failed")
    
    # Print summary
    successes = results.count("success")
    failures = results.count("failed")
    skipped = results.count("skipped")
    
    print("\nDownload Summary:")
    print(f"  Total files: {len(urls)}")
    print(f"  Successful: {successes}")
    print(f"  Skipped: {skipped}")
    print(f"  Failed: {failures}")

if __name__ == "__main__":
    main()
