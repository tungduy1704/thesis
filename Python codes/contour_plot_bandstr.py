import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_valence_conduction_separate_colorbars(
    eigen_csv="eigenvalues_map.csv",
    val_idx=6,
    cond_idx=7,
    nk=100
):
    # Read data from CSV file
    df = pd.read_csv(eigen_csv)

    kx = df["kx"].values
    ky = df["ky"].values

    a_lat = 3.190315
    kx = kx * (2 * np.pi / a_lat)
    ky = ky * (2 * np.pi / a_lat)

    band_cols = [col for col in df.columns if col.startswith("band_")]
    num_bands = len(band_cols)

    #if val_idx >= num_bands or cond_idx >= num_bands:
    #    raise ValueError(f"val_idx or cond_idx exceeds the number of bands: ({num_bands} bands).")

    valence_band = df[f"band_{val_idx+1}"].values
    conduction_band = df[f"band_{cond_idx+1}"].values

    kx_grid = kx.reshape((nk, nk))
    ky_grid = ky.reshape((nk, nk))
    valence_grid = valence_band.reshape((nk, nk))
    conduction_grid = conduction_band.reshape((nk, nk))

    fig, axs = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    # === Plot valence band
    cf1 = axs[0].contourf(kx_grid / (2*np.pi/a_lat), ky_grid / (2*np.pi/a_lat),
                          valence_grid, levels=100, cmap="jet")
    axs[0].set_title(f"Valence band")
    axs[0].set_xlabel(r"$k_x$ ($2\pi/a$)")
    axs[0].set_ylabel(r"$k_y$ ($2\pi/a$)")
    axs[0].set_aspect('equal')
    fig.colorbar(cf1, ax=axs[0], shrink=0.8)

    # === Plot conduction band
    cf2 = axs[1].contourf(kx_grid / (2*np.pi/a_lat), ky_grid / (2*np.pi/a_lat),
                          conduction_grid, levels=100, cmap="jet")
    axs[1].set_title(f"Conduction band")
    axs[1].set_xlabel(r"$k_x$ ($2\pi/a$)")
    axs[1].set_ylabel(r"$k_y$ ($2\pi/a$)")
    axs[1].set_aspect('equal')
    fig.colorbar(cf2, ax=axs[1], shrink=0.8, label="Energy (eV)")

    plt.show()


# Call the function with the specified parameters
plot_valence_conduction_separate_colorbars(
     eigen_csv="eigenvalues_map294.csv",
     val_idx=6,
     cond_idx=7,
     nk=199
 )
