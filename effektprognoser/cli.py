import argparse
from .sql_to_parquet.main import main as sqltable_to_parquet
from .parquet_to_excel.main import main as parquet_to_excel
from .parquet_to_category.main import main as parquet_to_category
from .qc.main import main as quality_check
from .plot_rutid.main import main as plot_rutid

VERSION = "0.1.0"


def get_regions(regions):
    if regions == "all":
        return ["06", "07", "08", "10", "12", "13"]
    if not isinstance(regions, list):
        return [regions]
    return regions


class Argparse:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Effektprognoser.se")

    def run(self):
        self.version()
        self.subparser()
        self.table2csv()
        self.csv2excel()
        self.csv2category()
        self.quality_check()
        self.plot_rutid()
        args = self.parser.parse_args()
        return args

    def version(self) -> None:
        """Add version functionality"""
        self.parser.add_argument(
            "-V", "--version", action="version", version=f"%(prog)s {VERSION}"
        )

    def subparser(self) -> None:
        """Add a subparser"""
        self.subparser = self.parser.add_subparsers(dest="command", required=True)

    def table2csv(self) -> None:
        sub = self.subparser.add_parser(
            "table2csv",
            help="Processera varje tabell i SQLite-databas och spara varje processad tabell som en Parquet-fil.",
        )
        sub.add_argument("--region", help="Vilken region skall bearbetas?")

    def csv2excel(self) -> None:
        sub = self.subparser.add_parser(
            "csv2excel",
            help="Skapa statistik per kommun utifrån Parquet-filer.",
        )
        sub.add_argument("--region", help="Vilken region skall bearbetas?")

    def csv2category(self) -> None:
        sub = self.subparser.add_parser(
            "csv2category",
            help="Para ihop varje Parquet-fil med en kategori (Bostäder, Industri och Bygg, osv).",
        )
        sub.add_argument("--region", help="Vilken region skall bearbetas?")

    def quality_check(self) -> None:
        sub = self.subparser.add_parser(
            "qc", help="Utför kvalitetskontroll per region."
        )
        sub.add_argument("--region", help="Vilken region skall kvalitetskontrolleras?")

    def plot_rutid(self) -> None:
        sub = self.subparser.add_parser(
            "plot-rutid",
            help="Visualisera alla modelldata som är associerad med en viss ruta.",
        )
        sub.add_argument(
            "--region",
            help="I vilken region ligger rutan vars modelldata ska visualiseras?",
        )
        sub.add_argument(
            "--rutid", help="Ange RutID för den ruta vars modelldata ska visualiseras."
        )


def main() -> None:
    args = Argparse().run()

    regions = get_regions(args.region)

    if args.command == "table2csv":
        sqltable_to_parquet(regions)

    elif args.command == "csv2excel":
        parquet_to_excel(regions)

    elif args.command == "csv2category":
        parquet_to_category(regions)

    elif args.command == "qc":
        quality_check(regions)

    elif args.command == "plot-rutid":
        plot_rutid(args.region, args.rutid)
