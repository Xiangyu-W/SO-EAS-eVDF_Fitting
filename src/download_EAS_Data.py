import requests
from bs4 import BeautifulSoup
import re
import os
import time
from datetime import datetime
import sys
from requests.exceptions import ChunkedEncodingError, ConnectionError, RequestException

# os.chdir(sys.path[0])

# Convert the input time from the filename into datetime objects
def parse_time_from_filename(filename):
    pattern = r"_(\d{8}T\d{6})-(\d{8}T\d{6})_"
    match = re.search(pattern, filename)
    if match:
        start_str, end_str = match.groups()
        start_time = datetime.strptime(start_str, '%Y%m%dT%H%M%S')
        end_time = datetime.strptime(end_str, '%Y%m%dT%H%M%S')
        return start_time, end_time
    return None, None

# Function to determine if two time ranges overlap
def time_ranges_overlap(file_start_time, file_end_time, user_start_time, user_end_time):
    # Check if the file time range overlaps with the user-specified time range
    return not (file_end_time < user_start_time or file_start_time > user_end_time)

# Function to find files within the specified time range
def find_files_in_time_range(url, start_time_str, end_time_str):
    # Convert input strings to datetime objects
    user_start_time = datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S')
    user_end_time = datetime.strptime(end_time_str, '%Y-%m-%d %H:%M:%S')

    # Send a request to the webpage
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find all file links ending with .cdf
    files_in_range = []
    for link in soup.find_all('a', href=re.compile(r'\.cdf$')):
        file_name = link['href']
        file_start_time, file_end_time = parse_time_from_filename(file_name)
        
        # Check if the file's time range overlaps with the user's time range
        if file_start_time and file_end_time:
            if time_ranges_overlap(file_start_time, file_end_time, user_start_time, user_end_time):
                files_in_range.append(url + file_name)

    return files_in_range

# Function to download files and estimate remaining time
def download_files(file_urls, save_dir, max_retries=5, retry_delay=10):
    # Ensure the save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    downloaded_file_paths = []  # List to store paths of downloaded files

    for file_url in file_urls:
        file_name = file_url.split('/')[-1]
        file_path = os.path.join(save_dir, file_name)  # Full path where the file will be saved
        temp_file_path = file_path + ".part"  # Temporary file for partial downloads

        # Add to list of returned file paths regardless of whether it's downloaded now or already exists
        downloaded_file_paths.append(file_path)

        # Check if the file already exists and is complete
        if os.path.exists(file_path):
            print(f"{file_name} already exists in {save_dir}, skipping download.")
            continue  # Skip the download if the file exists

        # Get the file size on the server
        try:
            file_size_on_server = int(requests.head(file_url).headers.get('content-length', 0))
        except RequestException:
            print(f"Error determining file size for {file_name}, skipping.")
            continue
            
        # Check if partial download exists
        downloaded_size = 0
        headers = {}
        if os.path.exists(temp_file_path):
            downloaded_size = os.path.getsize(temp_file_path)
            if downloaded_size < file_size_on_server:
                # Resume download
                headers = {'Range': f'bytes={downloaded_size}-'}
                print(f"Resuming download for {file_name} from {downloaded_size/1024/1024:.2f} MB")
            else:
                # If the temp file is already the full size, just rename it
                os.rename(temp_file_path, file_path)
                print(f"Using previously downloaded file for {file_name}")
                continue

        retries = 0
        while retries <= max_retries:
            try:
                print(f"Downloading {file_name} to {file_path}... (Attempt {retries + 1})")
                
                # Start downloading the file in chunks
                response = requests.get(file_url, stream=True, headers=headers)
                
                # If resuming, adjust the total size
                total_size = file_size_on_server
                if downloaded_size > 0 and response.status_code == 206:  # Partial content
                    total_size = downloaded_size + int(response.headers.get('content-length', 0))
                
                chunk_size = 1024  # 1 KB per chunk
                
                start_time = time.time()  # Track the start time
                
                # Open in append mode if resuming, otherwise write mode
                with open(temp_file_path, 'ab' if downloaded_size > 0 else 'wb') as f:
                    current_downloaded_size = downloaded_size
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            current_downloaded_size += len(chunk)
                            
                            # Calculate download progress and estimated time
                            progress = (current_downloaded_size / total_size) * 100
                            elapsed_time = time.time() - start_time
                            speed = (current_downloaded_size - downloaded_size) / elapsed_time if elapsed_time > 0 else 0
                            estimated_time_remaining = (total_size - current_downloaded_size) / speed if speed > 0 else 0

                            print(f"\rProgress: {progress:.2f}% | Downloaded: {current_downloaded_size / 1024 / 1024:.2f} MB "
                                  f"| Speed: {speed / 1024:.2f} KB/s | ETA: {estimated_time_remaining:.2f} seconds", end="")
                
                # If download completed successfully, rename the temp file
                os.rename(temp_file_path, file_path)
                print(f"\nDownloaded {file_name}")
                break  # Exit the retry loop if successful
                
            except (ChunkedEncodingError, ConnectionError, RequestException) as e:
                retries += 1
                if retries <= max_retries:
                    print(f"\nConnection error: {e}. Retrying in {retry_delay} seconds... (Attempt {retries+1}/{max_retries+1})")
                    time.sleep(retry_delay)
                    # Don't reset downloaded_size so we can resume from where we left off
                    headers = {'Range': f'bytes={current_downloaded_size}-'}
                else:
                    print(f"\nFailed to download {file_name} after {max_retries+1} attempts.")
                    # Keep the partial file for future resume
                    break

    return downloaded_file_paths

def download_eas_data(start_time_str, end_time_str, url=None, save_dir=None, max_retries=5, retry_delay=10):
    """
    Main function to download EAS data and return paths of downloaded files.
    
    Args:
        start_time_str (str): Start time in format 'YYYY-MM-DD HH:MM:SS'
        end_time_str (str): End time in format 'YYYY-MM-DD HH:MM:SS'
        url (str, optional): URL to download files from. Defaults to the 2022-PSD-Corrected URL.
        save_dir (str, optional): Directory to save files. Defaults to '<repo>/data/vdfs/'.
        max_retries (int, optional): Maximum number of download retry attempts. Defaults to 5.
        retry_delay (int, optional): Delay between retries in seconds. Defaults to 10.
        
    Returns:
        list: Paths of downloaded files
    """
    # Set default values if not provided
    if url is None:
        url = 'https://mssl.ucl.ac.uk/missions/solo_swa_quicklooks/EAS_calibrated_cdfs/2022-PSD-Corrected/'
    
    if save_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.abspath(os.path.join(script_dir, '..', 'data', 'vdfs'))
    
    # Find files in the time range
    files_in_range = find_files_in_time_range(url, start_time_str, end_time_str)
    
    if not files_in_range:
        print(f"No files found for the specified time range: {start_time_str} to {end_time_str}")
        return []
    
    # Download the found files and get the file paths
    downloaded_file_paths = download_files(files_in_range, save_dir, max_retries, retry_delay)
    
    return downloaded_file_paths

# This allows the script to be imported as a module or run as a standalone script
if __name__ == "__main__":
    url = 'https://mssl.ucl.ac.uk/missions/solo_swa_quicklooks/EAS_calibrated_cdfs/2022-PSD-Corrected/'
    start_time_str = '2022-03-01 23:59:59'  # Example start time string
    end_time_str = '2022-03-02 23:59:59'    # Example end time string

    file_paths = download_eas_data(start_time_str, end_time_str, url)

    print("\nDownloaded files:")
    for path in file_paths:
        print(path)