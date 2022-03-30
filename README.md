# pytket-dqc

This package additionally requires the installation of
[kahypar](https://github.com/kahypar/kahypar) %%Add proper instructions to install this package. There must be a mistake in the author's ones.%% and 
[graphviz](https://graphviz.org/download/). With these packages installed,
running `pip install .` will install pytket-dqc. Auto generated html 
documentation can be found as an artifact of the latest push action. 
You may also find the `example_notebook.ipynb` instructive.

## Using with M1 devices %%I think we can ommit this and only discuss "previously-installed boost.build vs non-previously-installed boost.build%%

Please follow these instructions to install this package on a device with an M1 processor.
Instructions formatted as `code` should be run from the terminal.

Note that this guide was made assuming the use of `homebrew`.

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

1. `pip install .`
