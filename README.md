## Effektprognoser

Repository is under development.

Website: [Effektprognoser.se](https://effektprognoser.se/)

### Clone the repository

```bash
git clone https://github.com/ekvll/effektprognoser.git
cd effektprognoser
```

### Installation

After cloning the repository, you can install the project using either `uv` or `pip`.

#### Using uv

To install the project using [uv](https://github.com/astral-sh/uv):

```bash
# Create and activate a virtual environment
uv venv .venv
source .venv/bin/activate
```

Then, either:

```bash
# Install dependencies defined in pyproject.toml
uv pip install .
```

or:

```bash
# Install dependencies defined in requirements.txt
uv pip install -r requirements.txt
```

or in editable mode:

```bash
# Install dependencies defined in pyproject.toml
uv pip install -e .
```

#### Using pip

...


### Command Line Interface (CLI)

To run the CLI, use the following command:

```bash
source .venv/bin/activate
python -m ep.cli.sql2parquet
```

### Tests

To run the tests, use the following command:

```bash
pytest
```

### Coverage

To check coverage, run:

```bash
pytest --cov=ep tests/
pytest --cov=ep --cov-report=html
vulture . --exclude=.venv
```

### License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
