.. Ares documentation master file, created by
   sphinx-quickstart on Thu Jun 22 15:21:19 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

================================
Welcome to Ares's documentation!
================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   :hidden:

   api
   dev 

Getting Started
===============

Installation
------------

Dependencies (Required)
^^^^^^^^^^^^^^^^^^^^^^^

* CMake 3.13 or greater
* C++17 compatible compiler
* Parthenon (using the submodule version provided by AthenaPK)
* Kokkos (using the submodule version provided by AthenaPK)

Dependencies (Optional)
^^^^^^^^^^^^^^^^^^^^^^^

* MPI
* OpenMP (for host parallelism. Note that MPI is the recommended option for on-node parallelism.)
* HDF5 (for outputs)

Building Ares
^^^^^^^^^^^^^

`Ares` is also used for integration testing and therefore closely tracks the `develop` branch of Parthenon.
For this reason, it is highly recommended to only use `Ares` with the Kokkos and Parthenon versions that are
provided by the submodules and to build everything together from source.
Neither other versions or nor using preinstalled Parthenon/Kokkos libraries have been tested.

Obtain all (Ares, Parthenon, and Kokkos) sources
.. code-block::

    git clone https://github.com/pgrete/parthenon-hydro
    cd parthenon-hydro

    # get submodules (mainly Kokkos and Parthenon)
    git submodule init
    git submodule update --init --recursive

Most of the general build instructions and options for Parthenon (see `here <https://github.com/lanl/parthenon/blob/develop/docs/building.md>`_) also apply to `Parthenon-Hydro`.
The following examples are a few standard cases.

Most simple configuration (only CPU, no MPI, no HDF5)
.. code-block::

    # enabling Broadwell architecture (AVX2) instructions
    cmake -S. -Bbuild-host -DKokkos_ARCH_BDW=ON -DPARTHENON_DISABLE_MPI=ON -DPARTHENON_DISABLE_HDF5=ON ../
    cd build-host && make

An Intel Skylake system (AVX512 instructions) with NVidia Volta V100 GPUs and with MPI and HDF5 enabled (the latter is the default option, so they don't need to be specified)
.. code-block::

    cmake -S. -Bbuild-gpu -DKokkos_ARCH_SKX=ON -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_VOLTA70=ON ../
    cd build-gpu && make

Run Ares
^^^^^^^^

Some example input files are provided in the [inputs](inputs/) folder.
.. code-block::

    # for a simple linear wave test run
    ./bin/ares -i ../inputs/linear_wave3d.in

    # to run a convergence test:
    for M in 16 32 64 128; do
      export N=$M;
      ./bin/ares -i ../inputs/linear_wave3d.in parthenon/meshblock/nx1=$((2*N)) parthenon/meshblock/nx2=$N parthenon/meshblock/nx3=$N parthenon/mesh/nx1=$((2*M)) parthenon/mesh/nx2=$M parthenon/mesh/nx3=$M
    done

    # and check the resulting errors
    cat linearwave-errors.dat


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
