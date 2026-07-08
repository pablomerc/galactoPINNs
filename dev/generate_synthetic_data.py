import json
import logging
import os
import time

import numpy as np

import galax.potential as gp
import jax.numpy as jnp

from galactoPINNs.data import generate_static_data

## Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

### Generate synthetic data ###

true_MW_potential = gp.MilkyWayPotential()
true_lmc_potential = gp.NFWPotential(m=1.0e11, r_s=5.0, units="galactic")
lmc_center = jnp.array([50.0, 0.0, 0.0])  # kpc
lmc_pot_centered = gp.TranslatedPotential(true_lmc_potential, translation=lmc_center)

true_MWpLMC_potential = true_MW_potential + lmc_pot_centered
analytic_baseline_potential = gp.NFWPotential(m=5.4e11, r_s=15.62, units="galactic")


def generate_raw_dataset(
    potential,
    n_train,
    n_test,
    r_max_train,
    r_max_test,
    save_path=None,
    meta=None,
):
    """Generate a raw (position, acceleration, potential) pool from a galax potential.

    Thin wrapper around ``generate_static_data`` that optionally persists the
    result. Returns the dict of arrays; if ``save_path`` is given, also writes a
    compressed ``.npz`` (named NumPy arrays, no pickle) plus a ``_meta.json``
    provenance sidecar.

    Args:
        potential: A galax potential (e.g. ``true_MWpLMC_potential``).
        n_train, n_test: Number of training / validation samples.
        r_max_train, r_max_test: Galactocentric sampling radii (kpc). Use a value
            >= R_sun + r_pos_meas (~23 kpc) so a later heliocentric selection
            isn't truncated.
        save_path: Optional ``.npz`` path. Parent dir is created if needed.
        meta: Optional provenance dict written alongside the ``.npz``.

    Returns:
        dict[str, Array]: keys ``x_train``, ``a_train``, ``u_train``, ``r_train``
        and the ``*_val`` equivalents.
    """
    raw = generate_static_data(
        galax_potential=potential,
        n_samples_train=n_train,
        n_samples_test=n_test,
        r_max_train=r_max_train,
        r_max_test=r_max_test,
    )
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # Save named NumPy arrays (no pickle). Load later with:
        #   d = np.load(save_path); d["x_train"], d["a_train"], ...
        np.savez_compressed(
            save_path, **{k: np.asarray(v) for k, v in raw.items()}
        )
        if meta is None:
            meta = {
                "n_samples_train": n_train,
                "n_samples_test": n_test,
                "r_max_train": r_max_train,
                "r_max_test": r_max_test,
            }
        with open(save_path.replace(".npz", "_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
    return raw


if __name__ == "__main__":
    logging.info("Generating synthetic data...")
    start_time = time.perf_counter()

    n = 5_000_000
    r_max = 25.0       # kpc; >= R_sun + r_pos_meas so a heliocentric cut isn't clipped
    r_max_test = 30.0  # kpc

    out_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    out_path = os.path.join(out_dir, "synthetic_data_raw_dict.npz")

    generate_raw_dataset(
        true_MWpLMC_potential,
        n_train=n,
        n_test=n,
        r_max_train=r_max,
        r_max_test=r_max_test,
        save_path=out_path,
        meta={
            "n_samples_train": n,
            "n_samples_test": n,
            "r_max_train": r_max,
            "r_max_test": r_max_test,
            "true_potential": "MilkyWayPotential + NFW(m=1e11, r_s=5) @ (50,0,0) kpc",
            "analytic_baseline": "NFW(m=5.4e11, r_s=15.62)",
            "sampler_seed": "PRNGKey(0) (hardcoded in rejection_sample_sphere)",
        },
    )

    end_time = time.perf_counter()
    logging.info(
        f"Data generation took {end_time - start_time:.2f} seconds for {n} samples. "
        f"Saved to {out_path}"
    )