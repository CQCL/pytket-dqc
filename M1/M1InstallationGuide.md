# This is a guide for installing pytket-dqc on an M1 processor based device (YMMV)

## This guide was made on a system that employed homebrew and python environments.

### Install cmake
1. brew install cmake

### Install kahypar with python interface
Note that the `-DKAHYPAR_USE_MINIMAL_BOOST=ON` flag is used. It may not be possible to get it to play nicely with an already present installation of boost.
1. cd ~
2. git clone --depth=1 --recursive git@github.com:SebastianSchlag/kahypar.git
3. mkdir build && cd build
4. cmake .. -DCMAKE_BUILD_TYPE=RELEASE -DKAHYPAR_PYTHON_INTERFACE=1 -DKAHYPAR_USE_MINIMAL_BOOST=ON
5. cd python
6. make
7. cp kahypar.so <path-to-site-packages>

### Install graphviz
1. brew install graphviz
2. pip install --global-option=build_ext --global-option="-I/opt/homebrew/include/" --global-option="-L/opt/homebrew/lib/graphviz" pygraphviz

### Install pytket-dqc
Make sure that setup.py does not include kahypar as a required package (you should already have it anyway if you followed the earlier steps.
1. cd <path-to-pytket-dqc-directory>
2. pip install .
