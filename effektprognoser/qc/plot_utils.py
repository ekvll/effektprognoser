import os
import matplotlib.pyplot as plt
from effektprognoser.paths import DATA_DIR


def plot_row(row, table: str, region: str) -> None:
    """
    Plot data contained in a Pandas dataframe row:

    Args:
        row (): The data row.
        table (str): Name of the table being quality checked.
        region (str): Region number of the region being quality checked.

    Returns:
        None
    """
    rid = str(row.rid)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_title(f"Region: {region}\nTabell: {table}\nID: {rid}")
    ax.plot(row["lp"], linewidth=1, color="black")
    ax.set_xlabel("Timma på året [h]")
    ax.set_ylabel("Effektbehov [MW]")
    fig.tight_layout()

    # Prepare output file path
    output_dir = os.path.join(DATA_DIR, "quality_check", region)
    os.makedirs(output_dir, exist_ok=True)

    output_name = f"{region}_{rid}_{table}.png"
    output_filepath = os.path.join(output_dir, output_name)

    plt.savefig(output_filepath)
    plt.close(fig)
