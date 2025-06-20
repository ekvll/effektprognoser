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

To install the whole repository in one go:

```bash
./scripts/install.sh
```

Or if you prefer a step-by-step approach:

First, create a virtual environment:

```bash
python3 -m venv .venv
#or
python -m venv .venv
```

and activate the environment:

```bash
source .venv/bin/activate
```

To install locally in normal mode:

```bash
pip install .
```

Or, to install locally in editable/development mode:

```bash
pip install -e .
```

Lastly, run the configuration script:

```bash
./scripts/config.sh
```

which tells you to define or accept the creation of various data paths.

If you would need to re-define the path where SQLite data are kept you can do so manually in `ep/config.py`.

---

### Usage

#### Execute processing pipelines

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

#### List all processing steps

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

#### Run a single processing step

First, activate the Python virtual environment:

```bash
source .venv/bin/activate
```

Thereafter, to run `sql2parquet`:

```bash
python -m ep.cli.sql2parquet
```

#### Tests

To run implemented Python tests, simply

```bash
pytest
```

#### Run webmap in development mode

```bash
live-server
# or
live-server index.html
```
