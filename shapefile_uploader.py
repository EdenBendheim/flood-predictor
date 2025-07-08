#!/usr/bin/env python3
import os
import requests
import subprocess
import zipfile
import shutil
from rich.console import Console

# --- Constants ---
SHAPEFILE_URL = "https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_state_500k.zip"
OUTDIR = "us-border"
ZIP_PATH = os.path.join(OUTDIR, "states.zip")

# SCP Transfer Settings from wldas_downloader.py
SCP_USER = "MLFEbndh"
SCP_HOST = "blp02.ccni.rpi.edu"
# Note: We are transferring the directory to the parent folder on the remote
SCP_DEST_PARENT_PATH = "/gpfs/u/home/MLFE/MLFEbndh/scratch/flood-predictor"
SSH_SOCKET_PATH = "~/.ssh/sockets/%r@%h:%p"

def main():
    """
    Downloads, unzips, and transfers the US states shapefile to the remote server.
    """
    console = Console()

    try:
        # --- Step 1: Download ---
        console.print(f"[bold blue]Step 1: Downloading shapefile...[/bold blue]")
        os.makedirs(OUTDIR, exist_ok=True)
        
        with requests.get(SHAPEFILE_URL, stream=True) as r:
            r.raise_for_status()
            with open(ZIP_PATH, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): 
                    f.write(chunk)
        console.print(f"[green]  > Download complete: {ZIP_PATH}[/green]")

        # --- Step 2: Unzip ---
        console.print(f"\n[bold blue]Step 2: Unzipping shapefile...[/bold blue]")
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(OUTDIR)
        os.remove(ZIP_PATH) # Clean up the zip file
        console.print(f"[green]  > Unzipped contents into '{OUTDIR}' directory.[/green]")

        # --- Step 3: Transfer ---
        console.print(f"\n[bold blue]Step 3: Transferring directory via SCP...[/bold blue]")
        console.print("\n[bold yellow]IMPORTANT:[/bold yellow] Ensure your SSH multiplexed connection is active before proceeding:")
        console.print(f"ssh -M -S {SSH_SOCKET_PATH} -fN {SCP_USER}@{SCP_HOST}\n")
        
        scp_command = [
            "scp",
            "-r",  # Recursive copy for the directory
            "-o", f"ControlPath={os.path.expanduser(SSH_SOCKET_PATH)}",
            OUTDIR,
            f"{SCP_USER}@{SCP_HOST}:{SCP_DEST_PARENT_PATH}/"
        ]
        
        console.print(f"Running command: {' '.join(scp_command)}")
        process = subprocess.run(
            scp_command, 
            check=True, 
            capture_output=True, 
            text=True, 
            timeout=300
        )
        console.print("[green]  > SCP transfer successful![/green]")
        console.print(f"[dim]{process.stdout}[/dim]")

    except requests.exceptions.RequestException as e:
        console.print(f"[bold red]Error: Download failed.[/bold red]")
        console.print(e)
        return
    except zipfile.BadZipFile:
        console.print(f"[bold red]Error: Failed to unzip file. It may be corrupt.[/bold red]")
        return
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        console.print(f"[bold red]Error: SCP transfer failed.[/bold red]")
        console.print(f"  > STDOUT: {e.stdout}")
        console.print(f"  > STDERR: {e.stderr}")
        return
    except Exception as e:
        console.print(f"[bold red]An unexpected error occurred.[/bold red]")
        console.print(e)
        return
    finally:
        # --- Step 4: Cleanup ---
        if os.path.exists(OUTDIR):
            console.print(f"\n[bold blue]Step 4: Cleaning up local directory...[/bold blue]")
            shutil.rmtree(OUTDIR)
            console.print(f"[green]  > Removed local directory: '{OUTDIR}'[/green]")

    console.print("\n[bold green]All steps completed successfully![/bold green]")


if __name__ == "__main__":
    main() 