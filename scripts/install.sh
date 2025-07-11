echo "Running full installation..."

# Create a Python environment
python3 -m venv .venv
# python -m venv .venv

# Activate the Python environment
source .venv/bin/activate

# Install the Python package in editable mode
pip install -e .

# Run the config
python -m ep.config

