#!/usr/bin/env python3
# filepath: /Users/edenbendheim/Dropbox/wildfire_spread_prediction_MLOS2/Caldor Paper/wldas_downloader.py

import os
import sys
import time
import requests
import getpass
import subprocess
from urllib.parse import urlparse, parse_qs, unquote
from concurrent.futures import ThreadPoolExecutor
from rich.progress import Progress, TextColumn, BarColumn, DownloadColumn, TimeElapsedColumn

# --- Constants ---
# Download settings
TXTFILE = "subset_WLDAS_NOAHMP001_DA1_D1.0_20250704_230939_.txt"
OUTDIR = "WLDAS"
TIMEOUT = 300  # 5-minute timeout for large file downloads
MAX_RETRIES = 10  # More retries for problematic files
RETRY_DELAY = 5  # Base delay between retries (will increase with backoff)
CONCURRENT_DOWNLOADS = 8  # Reduced concurrent downloads for stability

# SCP Transfer Settings
SCP_USER = "MLFEbndh"
SCP_HOST = "blp02.ccni.rpi.edu"
SCP_DEST_PATH = "/gpfs/u/home/MLFE/MLFEbndh/scratch/flood-predictor/WLDAS"
SSH_SOCKET_PATH = "~/.ssh/sockets/%r@%h:%p"


def get_remote_files():
    """Get a list of files from the remote SCP directory."""
    print(f"Checking for existing files on {SCP_HOST}...")
    scp_command = [
        "ssh",
        "-o", f"ControlPath={os.path.expanduser(SSH_SOCKET_PATH)}",
        f"{SCP_USER}@{SCP_HOST}",
        f"ls -1 \"{SCP_DEST_PATH}\""
    ]
    try:
        result = subprocess.run(scp_command, check=True, capture_output=True, text=True, timeout=120)
        return set(result.stdout.strip().split('\n'))
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        error_message = e.stderr or e.stdout if hasattr(e, 'stderr') else str(e)
        print(f"\nCould not list remote directory: {error_message}")
        print("Please ensure your SSH multiplexed connection is active and the path is correct.")
        print(f"ssh -M -S {SSH_SOCKET_PATH} -fN {SCP_USER}@{SCP_HOST}")
        sys.exit(1) # Exit if we can't verify what's on the remote server
    except FileNotFoundError:
        print("\n'ssh' command not found. Is OpenSSH client installed and in your PATH?")
        sys.exit(1)

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
    """Download a file, transfer it via scp, and then delete it locally."""
    filename = get_filename(url)
    outpath = os.path.join(OUTDIR, filename)
    temppath = outpath + ".part"
    
    # Skip if the final file already exists locally. This can happen if a
    # previous run failed after download but before or during transfer.
    if os.path.exists(outpath) and os.path.getsize(outpath) > 0:
        print(f"Local file {filename} exists, attempting transfer...")
        task_id = progress.add_task(f"[yellow]Retransferring {filename}", total=1)
        try:
            # --- Transfer via SCP ---
            scp_command = [
                "scp",
                "-o", f"ControlPath={os.path.expanduser(SSH_SOCKET_PATH)}",
                outpath,
                f"{SCP_USER}@{SCP_HOST}:{SCP_DEST_PATH}/"
            ]
            subprocess.run(scp_command, check=True, capture_output=True, text=True, timeout=600)
            os.remove(outpath) # Cleanup local file
            progress.update(task_id, completed=1, description=f"[green]Transferred {filename}")
            return "success"
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            error_message = e.stderr or e.stdout if hasattr(e, 'stderr') else str(e)
            print(f"\nFailed to re-transfer {filename}: {error_message}")
            progress.update(task_id, description=f"[red]SCP Failed")
            return "scp_failed"

    task_id = progress.add_task(f"[cyan]{filename}", total=100)

    # Retry logic for download
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.get(url, timeout=TIMEOUT)
            response.raise_for_status()
            
            total_size = len(response.content)
            progress.update(task_id, total=total_size)
            
            with open(temppath, 'wb') as f:
                f.write(response.content)
            
            os.rename(temppath, outpath)
            
            progress.update(task_id, completed=total_size / 2, description=f"[yellow]Transferring {filename}...")
            
            # --- Transfer via SCP ---
            scp_command = [
                "scp",
                "-o", f"ControlPath={os.path.expanduser(SSH_SOCKET_PATH)}",
                outpath,
                f"{SCP_USER}@{SCP_HOST}:{SCP_DEST_PATH}/"
            ]
            subprocess.run(scp_command, check=True, capture_output=True, text=True, timeout=600)
            
            os.remove(outpath) # Cleanup local file
            
            progress.update(task_id, completed=total_size, description=f"[green]Transferred {filename}")
            return "success"

        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            error_message = e.stderr or e.stdout if hasattr(e, 'stderr') else str(e)
            print(f"\nFailed to scp {filename}: {error_message}")
            progress.update(task_id, description=f"[red]SCP Failed")
            return "scp_failed"
        except requests.exceptions.RequestException as e:
            wait_time = RETRY_DELAY * attempt
            if os.path.exists(temppath): os.remove(temppath)
            progress.update(task_id, description=f"[yellow]Retrying DL {filename} ({attempt}/{MAX_RETRIES})")
            if attempt < MAX_RETRIES:
                time.sleep(wait_time)
            else:
                progress.update(task_id, description=f"[red]DL Failed {filename}")
                return "failed"

def main():
    os.makedirs(OUTDIR, exist_ok=True)
    
    print("\nIMPORTANT: Ensure your SSH multiplexed connection is active before proceeding:")
    print(f"ssh -M -S {os.path.expanduser(SSH_SOCKET_PATH)} -fN {SCP_USER}@{SCP_HOST}\n")
    
    # Get remote files first
    remote_files = get_remote_files()
    print(f"Found {len(remote_files)} files on the remote server.")

    session = get_session()
    
    with open(TXTFILE) as f:
        all_urls = [line.strip() for line in f if line.strip() and not line.startswith("//")]
    
    # Filter out URLs for files that already exist on the remote server
    urls_to_download = []
    for url in all_urls:
        filename = get_filename(url)
        if filename not in remote_files:
            urls_to_download.append(url)

    if not urls_to_download:
        print("\nAll files are already present on the remote server. Nothing to do.")
        return

    print(f"\nFound {len(all_urls)} total files in '{TXTFILE}'.")
    print(f"{len(urls_to_download)} files are missing and will be downloaded.")
    
    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=50),
        "[progress.percentage]{task.percentage:>3.1f}%",
        DownloadColumn(),
        TimeElapsedColumn(),
    ) as progress:
        results = []
        with ThreadPoolExecutor(max_workers=CONCURRENT_DOWNLOADS) as executor:
            future_to_url = {executor.submit(download_file, session, url, progress): url for url in urls_to_download}
            for future in future_to_url:
                try:
                    results.append(future.result())
                except Exception as exc:
                    print(f"A task generated an exception: {exc}")
                    results.append("failed")
    
    successes = results.count("success")
    failures = results.count("failed")
    scp_failures = results.count("scp_failed")
    skipped = results.count("skipped")
    
    print("\nTransfer Summary:")
    print(f"  Total files to process: {len(urls_to_download)}")
    print(f"  Successfully Transferred: {successes}")
    print(f"  Skipped (already local): {skipped}")
    print(f"  Download Failed: {failures}")
    print(f"  Transfer (SCP) Failed (file remains locally): {scp_failures}")

if __name__ == "__main__":
    main()
