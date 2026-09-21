#!/usr/bin/env bash
# Install a conda activation hook that makes this environment's libstdc++ take
# precedence over an older system one. Run once after `conda env create`:
#
#     conda activate CoaSTT_env
#     bash utils/misc_utils/install_libstdcxx_hook.sh
#
# Re-running is safe; it just rewrites the hook.
#
# WHY THIS IS NEEDED (and why it cannot be fixed by a pin in environment.yml)
#
# On older Linux distributions (e.g. RHEL/CentOS 8, as on midway3) the system
# /lib64/libstdc++.so.6 predates GLIBCXX_3.4.29, while numpy >=2.3 and other
# conda-forge builds require it. The environment already installs a modern
# libstdcxx-ng, and numpy's own extension carries an RPATH pointing into the
# environment, so `python -c "import numpy"` works on its own.
#
# The failure is import-order dependent. Several *pip* wheels in this
# environment (highspy -- pulled in by snakemake's SLURM executor plugin --
# plus cvxpy/_cvxcore, qdldl, osqp, frozenlist and all of vtk) are built as
# broad manylinux wheels and carry RUNPATH=$ORIGIN only. They therefore resolve
# libstdc++ to /lib64. The dynamic loader keeps exactly one libstdc++ per
# process, so once one of those is imported first, a later `import numpy` dies
# with:
#
#     ImportError: /lib64/libstdc++.so.6: version `GLIBCXX_3.4.29' not found
#
# That is why the error looks intermittent and why it hits `snakemake` (which
# imports highspy before pandas/numpy) but not a bare numpy import. Adding
# libstdcxx-ng to environment.yml does not help: it is already installed and
# already new enough -- it is simply not being *found* by those wheels.
#
# Prepending $CONDA_PREFIX/lib to LD_LIBRARY_PATH at activation makes the
# environment's newer libstdc++ win for every process started from the env.
set -euo pipefail

if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "error: no active conda environment. Run 'conda activate CoaSTT_env' first." >&2
    exit 1
fi

# CONDA_PREFIX is set to the base installation even when no environment is
# activated, and base is often a read-only system directory. Installing the hook
# there would be wrong (it would apply to every environment) so refuse it.
if [[ "${CONDA_DEFAULT_ENV:-base}" == "base" || "${CONDA_PREFIX}" == "${CONDA_PREFIX_1:-}" ]]; then
    echo "error: the 'base' environment is active, not the project environment." >&2
    echo "       Run 'conda activate CoaSTT_env' first." >&2
    exit 1
fi

if [[ ! -w "${CONDA_PREFIX}" ]]; then
    echo "error: ${CONDA_PREFIX} is not writable." >&2
    exit 1
fi

activate_dir="${CONDA_PREFIX}/etc/conda/activate.d"
deactivate_dir="${CONDA_PREFIX}/etc/conda/deactivate.d"
mkdir -p "$activate_dir" "$deactivate_dir"

# zz_ prefix so this sorts after any other activation script that might itself
# set LD_LIBRARY_PATH; this one must be applied last to stay in front.
cat > "${activate_dir}/zz_libstdcxx_activate.sh" <<'EOF'
#!/bin/sh
# Prepend this environment's lib/ so its libstdc++ outranks an older system copy.
# Installed by utils/misc_utils/install_libstdcxx_hook.sh -- see that file for why.
_coastt_libdir="${CONDA_PREFIX}/lib"
export _COASTT_LIBSTDCXX_OLD_LD_LIBRARY_PATH="${LD_LIBRARY_PATH-}"
export _COASTT_LIBSTDCXX_HAD_LD_LIBRARY_PATH="${LD_LIBRARY_PATH+yes}"
case ":${LD_LIBRARY_PATH-}:" in
    *":${_coastt_libdir}:"*) ;;  # already present, don't duplicate
    *) export LD_LIBRARY_PATH="${_coastt_libdir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
esac
unset _coastt_libdir
EOF

cat > "${deactivate_dir}/zz_libstdcxx_deactivate.sh" <<'EOF'
#!/bin/sh
# Restore LD_LIBRARY_PATH saved by activate.d/zz_libstdcxx_activate.sh
if [ "${_COASTT_LIBSTDCXX_HAD_LD_LIBRARY_PATH-}" = "yes" ]; then
    export LD_LIBRARY_PATH="${_COASTT_LIBSTDCXX_OLD_LD_LIBRARY_PATH}"
else
    unset LD_LIBRARY_PATH
fi
unset _COASTT_LIBSTDCXX_OLD_LD_LIBRARY_PATH
unset _COASTT_LIBSTDCXX_HAD_LD_LIBRARY_PATH
EOF

echo "Installed libstdc++ activation hook into ${CONDA_PREFIX}"
echo "Re-activate the environment for it to take effect:"
echo "    conda deactivate && conda activate ${CONDA_DEFAULT_ENV:-CoaSTT_env}"
