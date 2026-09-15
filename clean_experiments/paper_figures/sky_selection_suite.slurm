#!/bin/bash
# Sky-selection analysis over the FIRE-2 m12 suite: one SLURM array task per simulation.
# Fill in the <...> site-specific fields.  CPU only; the memory is for reading one star
# snapshot with gizmo_analysis (a few GB once cache/<sim>_oldstars_*.npz exists).
#
#   sbatch sky_selection_suite.slurm                      # from clean_experiments/paper_figures/
#   python sky_selection_summary.py                       # afterwards, once every task is done

#SBATCH -J sky_sel
#SBATCH -A <project>
#SBATCH -p <cpu-partition>
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH -t 01:00:00
#SBATCH --array=0-6
#SBATCH -o sky_sel_%A_%a.out

set -euo pipefail

SIMS=(m12i m12b m12c m12f m12m m12r m12w)
SIM=${SIMS[$SLURM_ARRAY_TASK_ID]}
FIRE_ROOT=${FIRE_ROOT:-/path/to/fire}                  # holds <sim>_res7100/ with track/host_coordinates.hdf5
CAT=${CAT:-/path/to/donlon2025/data.csv}               # v3 catalog (53 rows + units row)

cd "$SLURM_SUBMIT_DIR"
source ../../.venv/bin/activate                        # or: module load python; conda activate <env>

# 1. distance selection only (builds the old-star cache and the 6-panel figures)
python sampling_suite.py      --sim "$SIM" --fire-dir "$FIRE_ROOT/${SIM}_res7100" --catalog "$CAT"
# 2. distance + declination selection, rate and count weights (reuses the cache)
python sky_selection_suite.py --sim "$SIM" --fire-dir "$FIRE_ROOT/${SIM}_res7100" --catalog "$CAT"
# optional: sensitivity to the simulation frame's sign conventions
# python sky_selection_suite.py --sim "$SIM" --catalog "$CAT" --flip l
