#!/bin/bash

# Directory containing the checkpoints
CHECKPOINT_ROOT="checkpoints"

# Output directory for videos
OUTPUT_ROOT="sample_videos"
mkdir -p "$OUTPUT_ROOT"

# Iterate through each directory in checkpoints
for dir in "$CHECKPOINT_ROOT"/*/; do
    # Remove trailing slash for processing
    dir=${dir%/}
    [ -d "$dir" ] || continue
    folder_name=$(basename "$dir")
    
    echo "==================================================="
    echo "Processing: $folder_name"

    # Find the first .zip file in the directory
    checkpoint_file=$(find "$dir" -maxdepth 1 -name "*.zip" | head -n 1)
    
    if [ -z "$checkpoint_file" ]; then
        echo "No .zip checkpoint found in $dir, skipping."
        continue
    fi

    # Initialize arguments
    # We use --deterministic by default for cleaner samples
    ARGS="--checkpoint \"$checkpoint_file\" --episodes 5 --output-dir \"$OUTPUT_ROOT/$folder_name\" --deterministic"

    # Parse difficulty level (e.g., lvl1, lvl2, lvl5)
    if [[ $folder_name =~ lvl([0-5]) ]]; then
        difficulty=${BASH_REMATCH[1]}
        ARGS="$ARGS --difficulty $difficulty"
    fi

    # Parse feature flags
    # -h- or -h-sarl or -h-lattice usually indicates hierarchical
    if [[ $folder_name == *"-h-"* ]]; then
        ARGS="$ARGS --use-hierarchical"
    fi
    if [[ $folder_name == *"-lattice-"* ]]; then
        ARGS="$ARGS --use-lattice"
    fi
    if [[ $folder_name == *"-sarl-"* ]]; then
        ARGS="$ARGS --use-sarl"
    fi
    if [[ $folder_name == *"-lstm-"* ]]; then
        ARGS="$ARGS --use-lstm"
    fi

    # Run the sample script
    echo "Command: uv run sample.py $ARGS"
    eval "uv run sample.py $ARGS"
done

echo "==================================================="
echo "Done! Videos are saved in $OUTPUT_ROOT"

