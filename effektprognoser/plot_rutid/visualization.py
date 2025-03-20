import os
import matplotlib.pyplot as plt
from effektprognoser.paths import DATA_DIR


def make_plot(data, region, rutid, year):
    fz = 15
    fig, ax = plt.subplots(
        nrows=len(data) + 1, figsize=(18, 5 * len(data)), sharex=True, sharey=True
    )
    ax[0].set_title(
        f"Region: {region} - ID: {rutid} - År: {year}\nAlla data", fontsize=fz
    )

    for i, (table, ser) in enumerate(data.items()):
        label_ = table.split(f"EF_{year}_")[-1].split("_V1")[0].replace("_", " ")
        ax[0].plot(ser, "-", linewidth=1, label=label_)
        ax[i + 1].set_title(label_, fontsize=fz)
        ax[i + 1].plot(ser, "-", color="black", linewidth=1)

    for axe in ax:
        axe.set_ylabel("Effektbehov [MW]", fontsize=fz)
        axe.tick_params(axis="both", labelsize=fz)
    ax[-1].set_xlabel("Timma på året [h]", fontsize=fz)
    ax[0].legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=fz, title="År")

    fig.tight_layout()
    output_name = f"{region}_{rutid}_{year}.png"
    plt.savefig(os.path.join(DATA_DIR, "rut_id", region, output_name))
    plt.close(fig)


def make_plot_year(region, rutid, data_dict):
    color = ["red", "blue", "green", "magenta"]

    for table, data in data_dict.items():
        fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(12, 7))
        ax[0].set_title(f"Region: {region} - ID:{rutid}\n{table}")
        ax[1].set_title("Delta-kurvor")

        for i, (year, ser) in enumerate(data.items()):
            ax[0].plot(ser, linewidth=1, color=color[i], label=year, zorder=4 - i)
            if i == 0:
                ref_ser = ser
            else:
                ax[1].plot(
                    ser - ref_ser,
                    linewidth=1,
                    color=color[i],
                    label=f"{year} - {list(data.keys())[0]}",
                    zorder=4 - i,
                )

        ax[1].set_xlabel("Timma på året [h]")
        for axe in ax:
            axe.set_ylabel("Effektbehov [MW]")
            axe.legend(loc="upper left", bbox_to_anchor=(1, 1), title="År")

        output_name = f"{region}_{rutid}_{table}.png"
        fig.tight_layout()
        plt.savefig(os.path.join(DATA_DIR, "rut_id", region, output_name))
