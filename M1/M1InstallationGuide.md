# Installing pytket-dqc on M1 processor based device

## This guide was made on a system that employed homebrew and pyenv.

Instructions formatted as `code` should be run from the terminal.

### Install cmake

1. `brew install cmake`

### Install kahypar with python interface

1. `cd ~`
1. `git clone --depth=1 --recursive git@github.com:SebastianSchlag/kahypar.git`
1. `cd kahypar`
1. `mkdir build && cd build`
1. `cmake .. -DCMAKE_BUILD_TYPE=RELEASE -DKAHYPAR_PYTHON_INTERFACE=1 -DKAHYPAR_USE_MINIMAL_BOOST=ON`
1. `cd python`
1. `make`
1. `cp kahypar.so \<path-to-site-packages\>`

Note that the `-DKAHYPAR_USE_MINIMAL_BOOST=ON` flag is used. It may not be possible to get it to play nicely with an already present installation of boost. See <https://githubhot.com/repo/kahypar/kahypar/issues/98> for relevant discussion.

### Install graphviz

1. `brew install graphviz`
1. `pip install --global-option=build_ext --global-option="-I/opt/homebrew/include/" --global-option="-L/opt/homebrew/lib/graphviz" pygraphviz`

### Install pytket-dqc

1. Go to \<path-to-pytket-dqc-directory\>, rename setup.py -> setup_old.py (or some other name that you desire which is not setup.py).
1. Copy the setup.py found in \<path-to-pytket-dqc-directory\>/M1 directory and place it in \<path-to-pytket-dqc-directory\>.
1. `cd \<path-to-pytket-dqc-directory\>`
1. `pip install .`
