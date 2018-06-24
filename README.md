## Synopsis

The defcon library implements the deflated continuation algorithm of
Farrell, Beentjes and Birkisson. Its objective is to compute the
solutions of

f(u, λ) = 0,

where u is the solution of a PDE and λ is a parameter on which the PDE
depends.

The algorithm that defcon implements has two main advantages over the
previous state of the art:

* Defcon can compute disconnected bifurcation diagrams as well as connected
  ones.
* The algorithm can scale to massive discretisations of PDEs if a scalable
  preconditioner is available (although we have not pushed in this direction
  this yet).

For a full description of the algorithm, see

http://arxiv.org/abs/1603.00809

## Dependencies

The easiest way to run defcon is inside the docker images supplied by the FEniCS
project (http://fenicsproject.org/download); all dependencies are installed
there. This is described in more detail below.

If you're compiling things yourself, defcon depends on

* mpi4py (http://pythonhosted.org/mpi4py/)
* petsc4py (https://bitbucket.org/petsc/petsc4py)
* FEniCS, compiled with PETSc, petsc4py and HDF5 support (http://fenicsproject.org)

Defcon recommends (and some of the examples depend on)

* h5py, compiled against parallel HDF5 (http://www.h5py.org/)
* matplotlib (http://matplotlib.org, for rendering bifurcation diagrams)
* mshr (https://bitbucket.org/fenics-project/mshr)
* slepc4py (https://bitbucket.org/slepc/slepc4py, for computing stabilities)
* PyQT4/PyQT5 (https://riverbankcomputing.com/software/pyqt/intro, for the GUI)
* latex (https://www.tug.org/texlive/, for rendering TeX strings in the GUI)

Defcon adopts same versioning scheme as the FEniCS project. Development of
defcon closely follows FEniCS and some features require also a recent version of
PETSc. Released versions of defcon are compatible with a matching version of
FEniCS (since version 2017.1.0).

## Current status and automated testing

Defcon's serial capabilities are reasonably well tested. Its parallel features
are experimental.

Defcon is being automatically tested against a development version of FEniCS
docker images using Bitbucket Pipelines and CircleCI. This ensures that defcon
 should always run with a recent FEniCS development version.

[![Pipelines](https://bitbucket-badges.useast.atlassian.io/badge/pefarrell/defcon.svg)](https://bitbucket.org/pefarrell/defcon/addon/pipelines/home)
[![CircleCI](https://circleci.com/bb/pefarrell/defcon.svg?style=svg)](https://circleci.com/bb/pefarrell/defcon)

## Code Examples

The easiest way to learn how to use it is to examine the examples in
`examples/`. Start with `examples/elastica`, and compare to the Euler elastica
section of the manuscript cited above.

## Installation

    pip3 install .

or

    pip3 install --user -e .

for *editable* installs into the user directory (typically `~/.local`).

## Use in anaconda environments

Here is an example of how to use defcon with anaconda:

    # Install FEniCS
    conda create -n fenicsproject -c conda-forge fenics
    source activate fenicsproject

    # Install other dependencies
    conda install scipy
    pip install h5py # conda-supplied h5py is incompatible with FEniCS
    pip install https://bitbucket.org/mpi4py/mpi4py/get/master.tar.gz
    conda install matplotlib scipy ipython pyqt

    # Install defcon
    cd /path/to/defcon
    pip install -e .

    # Try it out
    cd examples/wingedcusp
    mpiexec -n 2 python wingedcusp.py
    defcon gui

## Use in FEniCS Docker containers

FEniCS Docker containers introduce a convenient way of distributing FEniCS on
many platforms, see

http://fenics-containers.readthedocs.io/en/latest/index.html .

To use defcon in a FEniCS docker container, simply fire up a container with
development version of FEniCS, e.g. using `fenicsproject run dev`, and in the
container type

    CC=mpicc HDF5_MPI=ON pip3 install --no-binary=h5py --no-cache-dir --user h5py
    pip3 install --user https://bitbucket.org/pefarrell/defcon/get/master.tar.gz

Then you can navigate to defcon demos and run them

    cd /home/fenics/.local/share/defcon/examples/elastica
    mpirun -n 2 python3 elastica.py

## Running the GUI in docker

To use the defcon GUI, a much more complicated procedure is needed.
We suggest installing FEniCS via anaconda as described above, instead.
However, if you're very keen on GUIs + docker, here are some notes.

### Connecting X11 applications on Linux
One needs to allow a docker container to connect to host's X11 system:

    xhost +
    docker run -ti -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix quay.io/fenicsproject/dev

or another docker image in place of quay.io/fenicsproject/dev.
Don't forget to run `xhost -` when finished with the container.

### Running the GUI in docker on OSX
If running on OSX, you need to have the appropriate XQuartz program running, and to
convince it to talk to the GUI in docker. To do so, perform the following steps:

* Run XQuartz
* Go to Preferences > 'Security' tab; turn on 'Allow connection from network clients'
* Restart XQuartz
* Get your local machine's IP:

        ip=$(ifconfig en0 | grep inet | awk '$1=="inet" {print $2}') && echo "My IP is: $ip"

* Allow the local machine to talk to XQuartz and run the docker container:

        xhost + ${ip}
        docker run -ti -e DISPLAY=${ip}:0 -v /tmp/.X11-unix:/tmp/.X11-unix quay.io/fenicsproject/dev /bin/bash

  or another docker image in place of quay.io/fenicsproject/dev.

Don't forget to run `xhost -` when finished with the container.

### Running the GUI in the FEniCS-supplied images
In the container one needs to install latex and one of PyQt5, PyQt4, or PySide.
The most convenient is

    sudo apt update
    sudo apt install -y python3-pyqt5 dvipng texlive-latex-extra texlive-fonts-recommended

After installing h5py and defcon as described above, one can run the gui and
start a defcon application by

    cd /home/fenics/.local/share/defcon/examples/elastica
    export QT_GRAPHICSSYSTEM=native # may not be necessary on all systems
    defcon gui &
    mpirun -n 2 python3 elastica.py

Note that only aspects of FEniCS docker containers directly related to defcon
have been shown. To setup a practical workflow (allowing preservation of JIT
cache, etc.), we recommend the FEniCS Docker manual, at

http://fenics-containers.readthedocs.io/en/latest/index.html .

## Troubleshooting

* Make sure all `Expressions` and `CompiledSubDomains` take in the `mpi_comm` argument
  (see e.g. `examples/navier-stokes`). This is the most common cause of silent
  deadlocks.

## Contributors

* Patrick E. Farrell <patrick.farrell@maths.ox.ac.uk>
* Joe Pollard <j.pollard@protonmail.com>
* Robert C. Kirby <robert_kirby@baylor.edu>
* Jan Blechta <blechta@karlin.mff.cuni.cz>
* Matteo Croci <croci@maths.ox.ac.uk>
* Nate J. C. Sime <njcs4@cam.ac.uk>

## License

GNU LGPL, version 3.
