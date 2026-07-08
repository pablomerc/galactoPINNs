"""
This file will contain functions that Pablo is working on developing.
"""
import numpy as np
import jax
import jax.numpy as jnp


def project_accelerations(a, x, point_of_view=np.array([0.0, 0.0, 0.0]),
                          pov_acceleration=np.array([0.0, 0.0, 0.0])):
    """Project acceleration vectors onto the line of sight from an observer.

    Computes both the *absolute* LOS acceleration (a·n̂, the GC-frame projection)
    and the *Sun-relative* LOS acceleration ((a - a_obs)·n̂) — which is what a
    pulsar actually measures. With the default ``pov_acceleration`` of zero, the
    relative outputs reduce to the absolute ones.

    Parameters
    ----------
    a : np.ndarray, shape (N, 3)
        Acceleration vectors in the galactocentric (GC) frame.
    x : np.ndarray, shape (N, 3)
        Positions in the galactocentric frame [kpc].
    point_of_view : np.ndarray, shape (3,)
        Observer position [kpc]. Sightlines point observer -> point.
    pov_acceleration : np.ndarray, shape (3,)
        GC-frame acceleration at the observer, a_obs. Pass
        ``potential.acceleration(point_of_view)`` to get the true pulsar
        observable; leave at zero for the absolute projection only.

    Returns
    -------
    dict
        "n_vecs"  : (N, 3) unit sightline directions (observer -> point).
        "los_abs" : (N,)   absolute LOS accel, a·n̂.
        "vec_abs" : (N, 3) absolute projection, (a·n̂) n̂.
        "los_rel" : (N,)   Sun-relative LOS accel, (a - a_obs)·n̂  <-- pulsar observable.
        "vec_rel" : (N, 3) Sun-relative projection, ((a - a_obs)·n̂) n̂.
    """
    dx = x - point_of_view                              # observer -> point
    n_vecs = dx / np.linalg.norm(dx, axis=1)[:, None]

    # absolute (GC-frame) LOS projection
    los_abs = np.einsum("ij,ij->i", a, n_vecs)
    vec_abs = los_abs[:, None] * n_vecs

    # Sun-relative LOS projection (subtract the observer's own acceleration)
    rel_acc = a - pov_acceleration[None, :]
    los_rel = np.einsum("ij,ij->i", rel_acc, n_vecs)
    vec_rel = los_rel[:, None] * n_vecs

    return {
        "n_vecs": n_vecs,
        "los_abs": los_abs,
        "vec_abs": vec_abs,
        "los_rel": los_rel,
        "vec_rel": vec_rel,
    }


def generate_observers_dataset(data_dict, observer_position=jnp.array([-8.2, 0.0, 0.0]),
    r_accel_meas=4, r_pos_meas=15,
    n_acc=50, n_pos=1_000_000, key=None):
    """
    Generate a synthetic dataset with n_pos position measurements out to r_pos_meas (resembling Gaia position measurements)
    and n_acc acceleration measurements centered around a given observer position (resembling pulsar timing or other form of acceleration measurement)

    Args:
        data_dict: Precomputed from `generate_static_data` with keys
            `"x_train"` (N, 3), `"a_train"` (N, 3), etc. Positions in kpc, accelerations in kpc / Myr².
        observer_position: Observer location (3,) in kpc, e.g. the Sun at
            ~(-8.2, 0, 0). All selection radii are measured from this point (heliocentric).
        r_accel_meas: Max heliocentric radius (kpc) for acceleration
            measurements (pulsars). Defaults to 4.
        r_pos_meas: Max heliocentric radius (kpc) for position measurements (Gaia). Defaults to 15.
        n_acc: Number of acceleration measurements to draw (few). Defaults to 50.
        n_pos: Number of position-only measurements to draw (many). Defaults to 1_000_000.
        key: JAX PRNG key. If None, defaults to PRNGKey(0).

    Returns:
        dict with keys:
            "x_acc_train"  (n_acc, 3)  - training positions with acceleration labels
            "a_acc_train"  (n_acc, 3)  - corresponding accelerations
            "x_pos_train"  (n_pos, 3)  - training positions without accelerations (Gaia-like)
            "x_acc_val"    (M, 3)      - validation positions within r_accel_meas
            "a_acc_val"    (M, 3)      - corresponding validation accelerations
            "x_pos_val"    (K, 3)      - validation positions within r_pos_meas

    Note:
        Validation sets are trimmed to the same heliocentric radii as training
        (r_accel_meas and r_pos_meas). Validation metrics therefore reflect
        in-distribution performance only, not extrapolation beyond the observed
        volume. Revisit if we want to evaluate generalization/extrapolation.
    Note:
        A limitation of this is that we are not accounting for the selection effects 
        of Gaia and real tracer distribution of stas and pulsars. 
        This is a first pass at generating a synthetic dataset for testing the PINN.
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    x_train = data_dict['x_train']
    a_train = data_dict['a_train']
    x_val   = data_dict['x_val']
    a_val   = data_dict['a_val']

    distance_train = jnp.linalg.norm(x_train - observer_position, axis=1)
    distance_val   = jnp.linalg.norm(x_val   - observer_position, axis=1)

    # --- Acceleration measurements (pulsars, within r_accel_meas) ---
    mask_acc_train = distance_train <= r_accel_meas
    x_within_r_acc = x_train[mask_acc_train]
    a_within_r_acc = a_train[mask_acc_train]

    assert mask_acc_train.sum() >= n_acc, (
        f"n_acc ({n_acc}) exceeds the number of training samples within "
        f"r_accel_meas={r_accel_meas} kpc ({mask_acc_train.sum()})"
    )

    print(f'{x_within_r_acc.shape[0]:,}/{x_train.shape[0]:,} training samples within {r_accel_meas} kpc of the observer.')
    print(f'Sampling {n_acc} acceleration measurements.')

    key, subkey = jax.random.split(key)
    idx_acc = jax.random.choice(subkey, a=x_within_r_acc.shape[0], shape=(n_acc,), replace=False)
    x_acc_train = x_within_r_acc[idx_acc]
    a_acc_train = a_within_r_acc[idx_acc]

    # --- Position-only measurements (Gaia, within r_pos_meas) ---
    mask_pos_train = distance_train <= r_pos_meas
    x_within_r_pos = x_train[mask_pos_train]

    assert mask_pos_train.sum() >= n_pos, (
        f"n_pos ({n_pos}) exceeds the number of training samples within "
        f"r_pos_meas={r_pos_meas} kpc ({mask_pos_train.sum()})"
    )

    print(f'{x_within_r_pos.shape[0]:,}/{x_train.shape[0]:,} training samples within {r_pos_meas} kpc of the observer.')
    print(f'Sampling {n_pos} position-only measurements.')

    key, subkey = jax.random.split(key)
    idx_pos = jax.random.choice(subkey, a=x_within_r_pos.shape[0], shape=(n_pos,), replace=False)
    x_pos_train = x_within_r_pos[idx_pos]

    # --- Validation (keep all points inside each radius) ---
    mask_acc_val = distance_val <= r_accel_meas
    x_acc_val = x_val[mask_acc_val]
    a_acc_val = a_val[mask_acc_val]

    mask_pos_val = distance_val <= r_pos_meas
    x_pos_val = x_val[mask_pos_val]

    print(f'\nValidation: {x_acc_val.shape[0]:,} samples within {r_accel_meas} kpc, '
          f'{x_pos_val.shape[0]:,} samples within {r_pos_meas} kpc.')

    return {
        "x_acc_train": x_acc_train,
        "a_acc_train": a_acc_train,
        "x_pos_train": x_pos_train,
        "x_acc_val":   x_acc_val,
        "a_acc_val":   a_acc_val,
        "x_pos_val":   x_pos_val,
    }
