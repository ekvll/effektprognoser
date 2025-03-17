import argparse

from .table_to_csv.app import main as table_to_csv
from .csv_to_excel.app import main as csv_to_excel
from .csv_to_category.app import main as csv_to_category
from .qc.app import main as qc
from .plot_rutid.app import main as plot_rutid

VERSION = "0.1.0"


def get_regions(regions):
    if regions == "all":
        return ["06", "07", "08", "10", "12", "13"]
    if not isinstance(regions, list):
        return [regions]
    return regions


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

    # Sub-parser for performing quality check
    parser_qc = subparser.add_parser("qc", help="Perform quality check per region.")
    parser_qc.add_argument(
        "--region", help="Which region contain the rut ID of interest?"
    )

    # Sub-parser for plotting a single rut id
    parser_plot_rutid = subparser.add_parser(
        "plot-rutid", help="Plot data associated with a single rut ID"
    )
    parser_plot_rutid.add_argument(
        "--region", help="Which region contain the rut ID of interest?"
    )
    parser_plot_rutid.add_argument("--rutid", help="The rut ID of interest")

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
    elif args.command == "qc":
        qc(regions)
    elif args.command == "plot-rutid":
        plot_rutid(args.region, args.rutid)
