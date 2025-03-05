#!/usr/bin/env zsh

#
# Note this assumes you have created a conda environment with the name of the agent
#

# Set error handling
set -e  # Exit on error
set -u  # Exit on undefined variables

# Initialize conda for zsh
source /opt/anaconda3/etc/profile.d/conda.sh

# Store the starting directory and script directory
ORIGINAL_DIR=$PWD
SCRIPT_DIR=${0:a:h}  # Get absolute path of script directory

# Setup logging
LOG_DIR="${SCRIPT_DIR}/../logs"
LOG_FILE="${LOG_DIR}/benchmark_$(date '+%Y%m%d_%H%M%S').log"

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Function to log messages
log_message() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] $1" | tee -a "$LOG_FILE"
}

# Function to handle conda environment
handle_conda_env() {
    local env_name=$1
    
    # Deactivate any active conda environment
    if (( ${+CONDA_DEFAULT_ENV} )); then
        while [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; do
            conda deactivate
        done
    fi
    
    # Activate the specified environment
    if ! conda activate "$env_name"; then
        log_message "Error: Failed to activate conda environment '$env_name'"
        return 1
    fi
}

# Function to run a single agent
run_agent() {
    local agent=$1
    local env_name=$2
    local config_path=$3
    
    log_message "Starting benchmark for $agent"
    log_message "Using config: $config_path"
    
    # Move to the correct directory
    cd "${SCRIPT_DIR}/../../${agent}/"
    log_message "Working directory: $PWD"
    
    # Handle conda environment
    if ! handle_conda_env "$env_name"; then
        return 1
    fi
    
    # Execute the Python script
    if python -m main --config "$config_path" 2>&1 | tee -a "$LOG_FILE"; then
        log_message "✓ $agent completed successfully"
        return 0
    else
        log_message "✗ $agent failed"
        return 1
    fi
}

# Define the agents and their corresponding conda environments
# Declare associative arrays properly
typeset -A MULTIAGENT_ENVS
typeset -A CROP_ENVS

# Initialize the associative arrays
MULTIAGENT_ENVS=(
#    [langgraph_multi_agent]="langgraph-multi-agent"
#    [crewai_multi_agent]="crewai-multi-agent"
#    [autogen_multi_agent]="autogen-multi-agent"
)

CROP_ENVS=(
    [crewai_crop_yield_simple_agent]="crewai_crop_yield"
    [autogen_crop_yield_simple_agent]="autogen_crop_yield"
)

# Run multiagent agents loop
log_message "Starting multiagent benchmarks"
for agent in ${(k)MULTIAGENT_ENVS}; do
    run_agent "$agent" "${MULTIAGENT_ENVS[$agent]}" "../benchmark_data/config/multiagent_questions_config.yaml"
done

# Run crop yield agents loop
log_message "Starting crop yield benchmarks"
for agent in ${(k)CROP_ENVS}; do
    run_agent "$agent" "${CROP_ENVS[$agent]}" "../benchmark_data/config/crop_yield_config.yaml"
done

log_message "All benchmarks completed"

# Return to the original directory
cd "$ORIGINAL_DIR"