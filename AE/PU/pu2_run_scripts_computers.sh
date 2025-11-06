#!/bin/bash

# Directory containing the scripts to run
SCRIPT_DIR="AE/PU/scripts"

# Directory for output files (will be created if it doesn't exist)
OUTPUT_DIR="AE_outputs/PU/pu2_computers"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Counter for tracking progress
total=$(find "$SCRIPT_DIR" -maxdepth 1 -name "*d.sh" | wc -l)
current=0

echo "Found $total scripts to run"
echo "Output will be saved to: $OUTPUT_DIR"
echo "----------------------------------------"

# Loop through all .sh files in the directory
for script in "$SCRIPT_DIR"/*d.sh; do
    # Check if file exists (handles case where no .sh files are found)
    [ -f "$script" ] || continue
    
    # Get the base name of the script (without path and extension)
    script_name=$(basename "$script" .sh)
    
    # Run only if the script name contains "computers"
    if [[ $script_name == *"computers"* ]]; then
        # Output file path
        output_file="$OUTPUT_DIR/${script_name}.txt"
    else
        continue
    fi
    
    # Increment counter
    ((current++))
    
    # Display progress
    echo "[$current/$total] Running: $script_name"
    
    # Make script executable (if not already)
    chmod +x "$script"
    
    # Run the script and redirect both stdout and stderr to the output file
    # Also display start time in the output file
    {
        echo "=========================================="
        echo "Script: $script_name"
        echo "Started at: $(date)"
        echo "=========================================="
        echo ""
        bash "$script" 2>&1
        echo ""
        echo "=========================================="
        echo "Finished at: $(date)"
        echo "Exit code: $?"
        echo "=========================================="
    } > "$output_file" 2>&1
    
    echo "  -> Output saved to: $output_file"
done

echo "----------------------------------------"
echo "All scripts completed!"
echo "Output files saved in: $OUTPUT_DIR"

