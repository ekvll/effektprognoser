echo "Running full installation..."
python3 -m venv .venv
# python -m venv .venv
source .venv/bin/activate
python -m ep.config
pip install -e .

