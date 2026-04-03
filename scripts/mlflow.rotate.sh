#!/bin/bash

# mlflow_rotate.sh
# Usage: ./mlflow.rotate.sh

# Prompt for experiment name
read -p "Enter MLflow experiment name: " EXPERIMENT_NAME
read -p "Delete runs older than how many days? " DAYS

# Validate DAYS is a number
if ! [[ "$DAYS" =~ ^[0-9]+$ ]]; then
    echo "Error: DAYS must be a number."
    exit 1
fi

# Convert days to timestamp (ms since epoch)
CURRENT_TS=$(date +%s)
OLDER_THAN_TS=$(( CURRENT_TS - DAYS*24*60*60 ))  # seconds
OLDER_THAN_TS_MS=$(( OLDER_THAN_TS * 1000 ))     # ms for MLflow

echo "Deleting runs in experiment '$EXPERIMENT_NAME' older than $DAYS days..."

python3 - <<EOF
from mlflow.tracking import MlflowClient
from datetime import datetime

client = MlflowClient()
# Find experiment ID by name
exp = client.get_experiment_by_name("$EXPERIMENT_NAME")
if exp is None:
    print("Experiment not found!")
    exit(1)

experiment_id = exp.experiment_id

# List runs older than timestamp
runs = client.search_runs(
    experiment_ids=[experiment_id],
    filter_string=f"attributes.start_time < {OLDER_THAN_TS_MS}"
)

if not runs:
    print("No runs to delete.")
else:
    print(f"Found {len(runs)} runs to delete...")
    for run in runs:
        print(f"Deleting run: {run.info.run_id}, started at {datetime.fromtimestamp(run.info.start_time/1000)}")
        client.delete_run(run.info.run_id)
EOF

echo "Done."