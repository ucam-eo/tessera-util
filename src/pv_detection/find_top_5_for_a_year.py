import os
import re

def get_pv_pixels_from_log(log_path):
    """
    Parses a single log file to extract the 'PV pixels detected' value.
    This function correctly handles numbers with comma separators.

    Args:
        log_path (str): The full path to the log file.

    Returns:
        int or None: The number of detected PV pixels, or None if not found or an error occurs.
    """
    # This pattern captures digits and commas in the number part of the line.
    pv_pixel_pattern = re.compile(r'^\s*PV pixels detected:\s*([\d,]+)\s*$')
    try:
        with open(log_path, 'r') as f:
            for line in f:
                match = pv_pixel_pattern.match(line)
                if match:
                    # Get the captured string (e.g., '316,029')
                    pixel_str_with_commas = match.group(1)
                    # Remove commas before converting to an integer
                    pixel_str_no_commas = pixel_str_with_commas.replace(',', '')
                    return int(pixel_str_no_commas)
    except (IOError, ValueError) as e:
        print(f"Warning: Could not read or parse file {log_path}. Error: {e}")
    # Return None if the line isn't found or an error occurs
    return None

def find_top_pv_grids(base_dir, target_year):
    """
    Finds and prints the top 20 grids with the most PV pixels for a specific year.

    Args:
        base_dir (str): The root directory containing the year folders.
        target_year (str): The year to analyze (e.g., '2020').
    """
    print(f"--- Finding Top 20 PV Grids for the Year: {target_year} ---")
    
    year_path = os.path.join(base_dir, target_year)

    # Check if the target year directory exists
    if not os.path.isdir(year_path):
        print(f"Error: Directory for year {target_year} not found at '{year_path}'")
        return

    grid_data = []

    # Iterate through all files in the year's directory
    log_files = [f for f in os.listdir(year_path) if f.endswith('_prediction.log')]
    
    print(f"Analyzing {len(log_files)} log files...")

    for log_file in log_files:
        log_path = os.path.join(year_path, log_file)
        
        # Get the pixel count from the log file
        pixel_count = get_pv_pixels_from_log(log_path)
        
        if pixel_count is not None:
            # Extract the grid name from the filename
            grid_name = log_file.replace('_prediction.log', '')
            grid_data.append({'grid': grid_name, 'pixels': pixel_count})

    # Check if any grid data was successfully collected
    if not grid_data:
        print("No valid grid data found for the specified year.")
        return

    # Sort the list of grids in descending order based on the pixel count
    sorted_grids = sorted(grid_data, key=lambda x: x['pixels'], reverse=True)

    # --- Print the top 20 results ---
    print(f"\n--- Top 20 Grids in {target_year} by PV Pixels Detected ---")
    
    num_to_print = min(20, len(sorted_grids))

    if num_to_print == 0:
        print("No grids to display.")
    else:
        for i in range(num_to_print):
            item = sorted_grids[i]
            # Use f-string formatting to add commas to the number for better readability
            print(f"{i+1}. Grid Name: {item['grid']}")
            print(f"   - PV Pixels: {item['pixels']:,}")
            print("-" * 25)

if __name__ == '__main__':
    # --- CONFIGURATION ---
    # Define the root directory containing the year folders
    prediction_root_dir = '/maps/zf281/btfm4rs/data/downstream/pv_detection/uk_prediction'
    
    # Set the year you want to analyze
    year_to_analyze = '2020'
    # --- END CONFIGURATION ---
    
    # Run the main function
    find_top_pv_grids(prediction_root_dir, year_to_analyze)