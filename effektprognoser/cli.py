import argparse

from .table_to_csv.app import main as table_to_csv
from .csv_to_excel.app import main as csv_to_excel
from .csv_to_category.app import main as csv_to_category
from .regions import get_regions
from .rut_id.app import main as rut_id
from .anomalies.app import main as anomalies

VERSION = "0.1.0"


def main():
    parser = argparse.ArgumentParser(description="Effektprognoser")

    # Parser for showing VERSION
    parser.add_argument(
        "-V", "--version", action="version", version=f"%(prog)s {VERSION}"
    )

    # Initialize sub-parsers
    subparser = parser.add_subparsers(dest="command", required=True)

    # Sub-parser for running the 'table to csv' pipeline
    parser1 = subparser.add_parser(
        "table2csv",
        help="Load, transform and save each table in a SQLite database as a CSV.",
    )
    parser1.add_argument("--region", help="Which region shall be processed?")

    # Sub-parser for creating Excel tables from Parquet
    parser2 = subparser.add_parser(
        "csv2excel",
        help="Make Excel sheet tables from Parquet.",
    )
    parser2.add_argument("--region", help="Which region shall be processed?")

    # Sub-parser for matching each csv into a category (Bostader, Industri och Bygg, and so on)
    parser3 = subparser.add_parser(
        "csv2category",
        help="Match each CSV to category (Bostader, Industri och Bygg, and so on).",
    )
    parser3.add_argument("--region", help="Which region shall be matched?")

    parser4 = subparser.add_parser(
        "view-square",
        help="View all data associated with a square",
    )
    parser4.add_argument("--region", help="Which region does the square belong to?")

    # Sub-parser for analysing data per rut id
    parser5 = subparser.add_parser(
        "rut-id", help="Get a plot of all data associated with a rut ID"
    )
    parser5.add_argument(
        "--region", help="Which region contain the rut ID of interest?"
    )
    parser5.add_argument("--id", help="Which rut ID to plot?")

    # Sub-parser for analysing data for anomalies
    parser6 = subparser.add_parser("anomalies", help="Analyze data for anomalies")
    parser6.add_argument("--region", help="Which region to look for anomalies")

    # Store parser arguments
    args = parser.parse_args()

    # Process args.region
    regions = get_regions(args.region)

    # Pipelines
    if args.command == "table2csv":
        table_to_csv(regions)
    elif args.command == "csv2excel":
        csv_to_excel(regions)
    elif args.command == "csv2category":
        csv_to_category(regions)
    elif args.command == "rut-id":
        rut_id(args.region, args.id)
    elif args.command == "anomalies":
        anomalies(regions)
