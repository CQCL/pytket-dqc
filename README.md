# pytket-dqc

Automated entanglement-efficient distribution of quantum circuits.

## Table of Contents

- [About](#about)
- [Requirements](#requirements)
- [Installing pytket-dqc](#installing-pytket-dqc)
- [Contributing](#contributing)

## About
This package takes a quantum circuit and network description and produces a circuit distributed across the given network,
using entanglement-assisted local operations and classical communication to implement non-local gates in the circuit,
with the aim of reducing the amount of entanglement required for the circuit implementation.
A more in-depth presentation of the methods implemented here can be found [here](**INSERT LINK TO PAPER**).

## Requirements
In addition to specified Python packages that will automatically be installed when you install pytket-dqc,
the following packages are also required.
We provide installation steps for MacOS,
but they should apply just the same for Linux systems by replacing `brew` with your package manager
(e.g. `apt` for Debian/Ubuntu users).

### [CMake](https://cmake.org/)
This is required for building the other packages listed here.

<details>

<summary>Installation Steps</summary>

1. `brew install cmake`

</details>

### [KaHyPar (with Python interface)](https://github.com/kahypar/kahypar)

This is required for the hypergraph partioning methods used in `pytket-dqc`.

We use an experimental feature from KaHyPar that let us select a different maximum weight for each block of the partition
(i.e. a different qubit capacity for each server).
The python interface for this feature was added in [this commit](https://github.com/kahypar/kahypar/commit/ff8fdf06c4e50af2faecddb9c5b6f7553e232df2).
It looks like KaHyPar >1.2.1 is the least requirement for this feature.
If KaHyPar is built from their repository (following the instructions given below), this feature is automatically available.

<details>
<summary>Boost</summary>

[KaHyPar requires Boost.Program_options](https://github.com/kahypar/kahypar#requirements).
If you do not have Boost installed then you can:
- Install it (see [here](https://www.boost.org/doc/libs/1_78_0/tools/build/doc/html/index.html#bbv2.installation))
- Add the flag `-DKAHYPAR_USE_MINIMAL_BOOST=ON` to the command given in Step 4 below.
</details>

<details>
<summary>Installation Steps</summary>

1. Navigate to the directory where you wish to download and build KaHyPar. 
 
 For instance, `cd ~` will do so in your home directory, which is probably fine for most people.
 The location of this directory is up to you and it will not affect installation.

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

5. Build KaHyPar's Python interface:  
  ```
  cd python
  make
  ```

6. Copy the dynamic library to the appropriate path so that Python can import the kahypar module and use its contents:  
  ```
  cp kahypar.so <path-to-site-packages>
  ```
  You must replace `<path-to-site-packages>` with the path to the directory `site-packages` listed by the command `python -m site`. For instance:
  ```
  cp kahypar.so /home/myuser/anaconda3/lib/python3.9/site-packages'
  ```
</details>
  
<details>
<summary>Test KaHyPar</summary>

You should now be able to use KaHyPar in Python.
To test this, in your terminal open a Python 3 shell with command `python3` and then do `import kahypar`.
If this fails then Python cannot access the dynamic library (the `.so` file).
If you get such an error it is likely something went wrong in step 6 from the instructions above,
i.e. the `.so` file was not copied to the right directory, or the wrong `.so` file was copied.

</details>

<details>

<summary>Troubleshooting</summary>

<details>

<summary>I got an error when building KaHyPar! (Step 4)</summary>

If an error occurs it might be that you do not have the Boost library installed in your computer.
You may choose to install it yourself
(your package manager is likely able to do it for you, e.g. `brew install boost`)
or ask CMake to fetch the minimal requirements and install them using the following command instead of the one given in Step 4
```
cmake ../ -DCMAKE_BUILD_TYPE=RELEASE -DKAHYPAR_PYTHON_INTERFACE=1 -DKAHYPAR_USE_MINIMAL_BOOST=ON
```
Note that the `-DKAHYPAR_USE_MINIMAL_BOOST=ON` flag is used.
It may not be possible to get it to play nicely with an already present installation of boost.
See <https://githubhot.com/repo/kahypar/kahypar/issues/98> for relevant discussion.

</details>

<details>

<summary>My terminal is telling me `kahypar.so` doesn't exist when I try to copy it! (Step 6)</summary>

Try to find a file in the directory `kahypar/build/python` with the extension `.so` and copy that instead.

You do not need to rename the file.

</details>

</details>

### [Graphviz](https://graphviz.org/download/)
This package is used for graph visualisation.

<details>
<summary>Installation Steps</summary>

1. `brew install graphviz`
1. `pip install --global-option=build_ext --global-option="-I/opt/homebrew/include/" --global-option="-L/opt/homebrew/lib/graphviz" pygraphviz`
</details>

## Installing pytket-dqc

1. Ensure that CMake, KaHyPar, and Graphviz are installed. See above for guidance on this.
1. In your terminal, navigate to the directory you wish to download `pytket-dqc`'s source files to.
(If unsure, then `cd ~` is probably fine.)
1. Clone this repository with `git clone git@github.com:CQCL/pytket-dqc.git`.
1. Navigate to the cloned repository `cd pytket-dqc`.
1. Run `pip install .`

Documentation for using `pytket-dqc` can be found [here](INSERT URL HERE WHEN AVAILABLE)
You may also find the example Jupyter notebooks in `examples/` instructive.
(Note that these require a seperate installation of packages capable of opening Jupyter notebooks,
such as [JupyterLab](https://jupyter.org/install).)

#### Testing pytket-dqc

You may wish to test your installation has succeeded.

You can do this by running `pip install pytest` to install `pytest`.
Then running `pytest tests -m "not high_compute` from the directory that these source files were `git clone`d to.
(The additional flag skips tests that will take a long time.)
Assuming that all the tests pass then you have succeeded in installing `pytket-dqc`!


## Contributing

If you wish to contribute to the development of `pytket-dqc` (which would be extremely welcome!),
you can do so by creating a branch and submitting a PR.

For code consistency, we'd ask you to include type annotations on your code,
and ensure that your code conforms to standards set by `flake8` (with a line length of 79 characters).
This will be checked automatically on any PRs submitted to this project, by `mypy` and `flake8` respectively.