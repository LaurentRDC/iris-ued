# Iris - Ultrafast Electron Scattering Data Exploration

[![Documentation Build Status](https://readthedocs.org/projects/iris-ued/badge/?version=master)](http://iris-ued.readthedocs.io/) [![PyPI Version](https://img.shields.io/pypi/v/iris-ued.svg)](https://pypi.python.org/pypi/iris-ued) [![Conda-forge Version](https://img.shields.io/conda/vn/conda-forge/iris-ued.svg)](https://anaconda.org/conda-forge/iris-ued)


Iris is both a library for interacting with ultrafast electron
diffraction data, as well as a GUI frontend for interactively exploring
this data.

Iris also includes a plug-in manager so that you can explore your data.

![Two instances of the iris GUI showing data exploration for ultrafast
electron diffraction of single crystals and
polycrystals.](iris_screen.png)

## Contents:
  - [Installation](#installation)
  - [Usage](#usage)
  - [Test Data](#test-data)
  - [Documentation](#documentation)
  - [Citations](#citations)
  - [Support / Report Issues](#support--report-issues)
  - [License](#license)

## Installation

To interact with [iris]{.title-ref} datasets from a Python environment,
the [iris-ued]{.title-ref} package must be installed. [iris]{.title-ref}
is available on PyPI; it can be installed with
[pip](https://pip.pypa.io).:

    python -m pip install iris-ued

[iris]{.title-ref} is also available on the conda-forge channel:

    conda config --add channels conda-forge
    conda install iris-ued

To install the latest development version from
[Github](https://github.com/LaurentRDC/iris-ued):

    python -m pip install git+git://github.com/LaurentRDC/iris-ued.git

Each version is tested against Python 3.6+. If you are using a different
version, tests can be run using the standard library\'s
[unittest]{.title-ref} module.

## Usage

Once installed, the package can be imported as `iris`.

The GUI component can be launched from a command line interpreter as
`python -m iris` or `pythonw -m iris` (no console window).

## Test Data

Test datasets are made available on the Siwick research group public
data server, which can be [accessed anonymously
here](http://www.physics.mcgill.ca/siwicklab/publications.html).

## Documentation

The [Documentation on readthedocs.io](https://iris-ued.readthedocs.io)
provides API-level documentation, as well as tutorials.

## Citations

If you find this software useful, please consider citing the following
publications:

## Support / Report Issues

All support requests and issue reports should be [filed on Github as an
issue](https://github.com/LaurentRDC/iris-ued/issues).

## License

iris is made available under the GPLv3 License. For more details, see
[LICENSE.txt](https://github.com/LaurentRDC/iris-ued/blob/master/LICENSE.txt).
