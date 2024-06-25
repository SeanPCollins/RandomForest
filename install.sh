#!/bin/bash

# Function to create a directory if it doesn't exist
create_directory() {
    local dir="$1"
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        echo "Created $dir"
    else
        echo "$dir already exists"
    fi
}

# Create .defaults directory in user's home directory if it doesn't exist
defaults_dir="$HOME/.defaults"
create_directory "$defaults_dir"

# Create bin directory in user's home directory if it doesn't exist
bin_dir="$HOME/bin"
create_directory "$bin_dir"

# Path to the file to create a symlink to
file_to_link="Source_code/random_forest.py"

# Check if the file exists
if [ ! -f "$file_to_link" ]; then
    echo "Error: $file_to_link does not exist."
    exit 1
fi

# Create a symbolic link in the bin directory
ln -s "$(realpath "$file_to_link")" "$bin_dir/random_forest.py"

# Check if symlink was created successfully
if [ $? -ne 0 ]; then
    echo "Error: Failed to create symlink to $file_to_link in $bin_dir"
    exit 1
fi

echo "Successfully created symlink to $file_to_link in $bin_dir"

# Path to the source file to copy
source_file="Useful_Files/RF_defaults.ini"

# Check if the source file exists
if [ ! -f "$source_file" ]; then
    echo "Error: $source_file does not exist."
    exit 1
fi

# Copy the source file to the .defaults directory
cp "$source_file" "$defaults_dir/"

# Check if the copy was successful
if [ $? -ne 0 ]; then
    echo "Error: Failed to copy $source_file to $defaults_dir"
    exit 1
fi

echo "Successfully copied $source_file to $defaults_dir"
