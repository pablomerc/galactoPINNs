"""Checkpoint save / reload for noisy_data_arch.

A bundle is a self-contained pickle: architecture tag, the scaling cfg (its
transformers, picklable), the trained Param state (as numpy), and -- for IV/V
-- the baseline scalars needed to rebuild the galax NFW at load time (galax
potential objects are stripped from cfg rather than pickled). load_model
reconstructs a StaticModel that predicts identically to the trained one, so it
can be wrapped with make_galax_potential and evaluated anywhere.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from arch_datasets import BASE_M, BASE_RS, make_model, trainable_mr


def save_model(outdir, arch, cfg, model, meta=None):
    """Write a reload bundle to ``outdir/bundle.pkl``; return the path."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    state_np = jax.tree.map(np.asarray, nnx.state(model, nnx.Param))
    cfg_core = {k: v for k, v in cfg.items() if k != "ab_potential"}
    bundle = {
        "arch": arch,
        "cfg_core": cfg_core,
        "state": state_np,
        "baseline_mr": trainable_mr(model) if arch == "V" else None,
        "base_m": BASE_M, "base_rs": BASE_RS,
        "meta": meta or {},
    }
    path = outdir / "bundle.pkl"
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    return path


def load_model(bundle_path):
    """Rebuild (model, cfg, meta) from a bundle written by save_model."""
    import galax.potential as gp  # local: keep import cost off module load

    with open(bundle_path, "rb") as f:
        b = pickle.load(f)
    arch, cfg = b["arch"], dict(b["cfg_core"])
    if arch == "IV":
        cfg["ab_potential"] = gp.NFWPotential(
            m=b["base_m"], r_s=b["base_rs"], units="galactic")
    if arch == "V":
        m, r_s = b["baseline_mr"]  # rebuild the layer AT THE LEARNED values
        model = make_model("V", cfg, seed=0, m=m, r_s=r_s)
    else:
        model = make_model(arch, cfg, seed=0)
    nnx.update(model, jax.tree.map(jnp.asarray, b["state"]))
    return model, cfg, b.get("meta", {})
