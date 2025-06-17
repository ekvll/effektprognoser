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

##### Run configuration

First off, run the configuration script:

```bash
./scripts/config.sh
```

which tells you to define the path to where the SQLite tables kept.

##### Execute processing pipelines

To run the whole pipeline, which begins with processing SQLite tables, and in the end outputs various products, type:

```bash
./scripts/all.sh
```

To just process SQLite tables and save each table as a Parquet file, type:

```bash
./scripts/sql2parquet.sh
```

To just process all Parquet files and output various products, type:

```bash
./scripts/parquet2products.sh
```

##### List all processing steps

To list all available CLI commands:

```bash
./scripts/list_processing_steps.sh
```

which will output:

```bash
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

##### Run a single processing step

First, activate the Python virtual environment:

```bash
source .venv/bin/activate
```

Thereafter, to run `sql2parquet`:

```bash
python -m ep.cli.sql2parquet
```
