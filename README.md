# pytket-dqc

This package additionally requires the installation of
[kahypar](https://github.com/kahypar/kahypar) and 
[graphviz](https://graphviz.org/download/).

In the Requirements section of [kahypar](https://github.com/kahypar/kahypar), if you are not sure you have Boost.build installed or if you want to install it, please check https://www.boost.org/doc/libs/1_78_0/tools/build/doc/html/index.html#bbv2.installation.

We are using an experimental feature from KaHyPar that let us select a different maximum weight for each block of the partition (i.e. a different qubit capacity for each server). The python interface for this feature was added in [this commit](https://github.com/kahypar/kahypar/commit/ff8fdf06c4e50af2faecddb9c5b6f7553e232df2) on July 2020. It looks like KaHyPar >1.2.1 is the least requirement for this feature. If KaHyPar is built from their repository (following the instructions given below), this feature is automatically available.

With these packages installed,
running `pip install .` will install pytket-dqc. Auto generated html 
documentation can be found as an artifact of the latest push action. 
You may also find the `example_notebook.ipynb` instructive.

## Installation instructions (tested on M1 Mac systems and Ubuntu 20.04) 

Please follow these instructions to install this package on a device with an M1 processor.
Instructions formatted as `code` should be run from the terminal.

Note that this guide was made assuming the use of `homebrew`. Other package managers may be used if applicable; for instance, if in Debian/Ubuntu, replacing `brew` with `apt` should do the job.

### Install cmake

1. `brew install cmake`

### Install kahypar with python interface

1. Navigate to the directory where you wish to download and build KaHyPar. For instance, `cd ~` will do so in your home directory. The location of this directory is up to you and it will not affect installation.
2. Clone the repository (using HTTPS, since the SSH link to the repository seems to be broken):
  ```
  git clone --depth=1 --recursive https://github.com/kahypar/kahypar.git
  ```
3. Create and move to the build directory:
  ```
  mkdir kahypar/build && cd kahypar/build
  ```
4. Build KaHyPar
```
cmake ../ -DCMAKE_BUILD_TYPE=RELEASE -DKAHYPAR_PYTHON_INTERFACE=1
```
If an error occurs it might be that you do not have the Boost library installed in your computer. You may choose to install it yourself (your package manager brew/apt is likely able to do it for you, e.g. `brew install boost`) or ask CMake to fetch the minimal requirements and install them using the following command instead of the above:
```
cmake ../ -DCMAKE_BUILD_TYPE=RELEASE -DKAHYPAR_PYTHON_INTERFACE=1 -DKAHYPAR_USE_MINIMAL_BOOST=ON
```
Note that the `-DKAHYPAR_USE_MINIMAL_BOOST=ON` flag is used. It may not be possible to get it to play nicely with an already present installation of boost. See <https://githubhot.com/repo/kahypar/kahypar/issues/98> for relevant discussion.

5. Build KaHyPar's Python interface:
```
cd python
make
```
6. Copy the dynamic library to the appropriate path so that Python can import the kahypar module and use its contents:
```
cp kahypar.so \<path-to-site-packages\>
```
You must replace `\<path-to-site-packages\>` with the path to the directory `site-packages` listed by the command `python -m site`. For instance:
```
cp kahypar.so /home/myuser/anaconda3/lib/python3.9/site-packages'
```
If you get an error saying that the file `kahypar.so` does not exist, try to find a file in the directory `kahypar/build/python` with the extension `.so` and use that instead. You do not need to rename the file.

#### Test KaHyPar

You should now be able to use KaHyPar in Python. To test this, run `pytest tests/distributor_test.py -k test_kahypar_install`. If an error is thrown saying either that the line `import kahypar as kahypar` fails or "Hypergraph is not an attribute of kahypar module" this means that Python cannot access the dynamic library (the `.so` file). If you get such an error it is likely something went wrong in step 6 from the instructions above, i.e. the path you copied to is not the right one, or you copied the wrong `.so` file.


### Install graphviz

1. `brew install graphviz`
2. `pip install --global-option=build_ext --global-option="-I/opt/homebrew/include/" --global-option="-L/opt/homebrew/lib/graphviz" pygraphviz`

### Install pytket-dqc

1. `pip install .`
