echo "Running full installation..."
python3 -m venv .venv
# python -m venv .venv
source .venv/bin/activate
pip install -e .
python -m ep.config

