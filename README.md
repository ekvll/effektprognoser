## Effektprognoser.se

This repository is under development.

---

### Website

Visit [Effektprognoser.se](https://effektprognoser.se/).

---

### Installation

#### Clone the repository

```bash
git clone https://github.com/ekvll/effektprognoser.git
cd effektprognoser
```

#### Install the repository locally

To install locally in normal mode:

```bash
pip install .
```

Or, to install locally in editable/development mode:

```bash
pip install -e .
```

---

### Usage

#### Command Line Interface (CLI)

##### List CLI commands

To list all available CLI commands:

```bash
./scripts/list_cli_commands.sh
```

which will output:

```bash
Available CLI commands:
__init__
geojson2statistics
parquet2all
parquet2all_table
parquet2geojson
parquet2kommun
parquet2kommun_maxeffekt
parquet2region
sql2parquet
sql2parquet_chunk
```

##### Run a CLI command

First, activate the Python virtual environment:

```bash
source .venv/bin/activate
```

Thereafter, to run `sql2parquet`:

```bash
python -m ep.cli.sql2parquet
```
