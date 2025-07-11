# Effektprognoser.se

**Effektprognoser.se** is a data processing and web visualization project for power forecasts in Sweden.  
This repository contains scripts and tools for data ingestion, transformation, and web presentation.

> 🚧 **Note:** This repository is under active development.

---

## 🌐 Website

Explore the live website: [Effektprognoser.se](https://effektprognoser.se/)

---

## 🛠️ Prerequisites

- Python 3.8+
- Bash shell (for running scripts)
- [live-server](https://www.npmjs.com/package/live-server) (for webmap development)
- [pytest](https://docs.pytest.org/) (for testing)

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/ekvll/effektprognoser.git
cd effektprognoser
```

### 2. Install dependencies and configure

Run the installation script:

```bash
./scripts/install.sh
```

This script will also run `/scripts/config.sh` to help you define or accept default data paths.

> **Note:** If you need to change the location for SQLite data, edit `ep/config.py` manually.

---

## ⚡ Usage

### Run the full data pipeline

Processes all stages, from SQLite tables to product outputs:

```bash
./scripts/all.sh
```

### Run specific pipelines

- **Convert SQLite tables to Parquet files:**
  ```bash
  ./scripts/sql2parquet.sh
  ```
- **Process all Parquet files and generate output products:**
  ```bash
  ./scripts/parquet2products.sh
  ```

### List available processing steps

See all available CLI commands:

```bash
./scripts/list_processing_steps.sh
```

Sample output:

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

### Run a single processing step

1. Activate the Python virtual environment:
   ```bash
   source .venv/bin/activate
   ```
2. Execute a single step (e.g. `sql2parquet`):
   ```bash
   python -m ep.cli.sql2parquet
   ```

---

## 🧪 Testing

End-to-end and unit tests are included.

1. **Download the test data set:**  
   [Effektmodell_test.sqlite](https://nppd.se/effektprognoser/Effektmodell_test.sqlite)

2. **Place the file in:**

   ```
   data/_test/sqlite/Effektmodell_test/
   ```

3. **Run all tests:**
   ```bash
   pytest
   ```

---

## 🗺️ Run the Webmap (Development)

Start a local server to develop or view the webmap:

```bash
live-server
# or
live-server index.html
```

---

## 📄 License

[MIT](LICENSE)

---

## 🙌 Contributions

Contributions and feedback are welcome!  
Feel free to open an issue or submit a pull request.
