Iris
====

.. image:: https://img.shields.io/appveyor/ci/LaurentRDC/iris-ued/master.svg
    :target: https://ci.appveyor.com/project/LaurentRDC/iris-ued
    :alt: Windows Build Status
.. image:: https://readthedocs.org/projects/iris-ued/badge/?version=master
    :target: http://iris-ued.readthedocs.io/
    :alt: Documentation Build Status
.. image:: https://img.shields.io/pypi/v/iris-ued.svg
    :target: https://pypi.python.org/pypi/iris-ued
    :alt: PyPI Version


Iris is both a library for interacting with ultrafast electron diffraction data, as well as a GUI frontend
for interactively exploring this data.

Iris also includes a plug-in manager so that you can explore your data.

Installation
------------

To install the latest development version from `Github <https://github.com/LaurentRDC/iris-ued>`_::

    python -m pip install git+git://github.com/LaurentRDC/iris-ued.git

Iris is available on PyPI; it can be installed with `pip <https://pip.pypa.io>`_.::

    python -m pip install iris-ued

Each version is tested against Python 3.6. If you are using a different version, tests can be run
using the standard library's `unittest` module.

Usage
-----

Once installed, the package can be imported as :code:`iris`. 

The GUI component can be launched from a command line interpreter as :code:`python -m iris`
or :code:`pythonw -m iris` (no console window).

Documentation
-------------

The `Documentation on readthedocs.io <https://iris-ued.readthedocs.io>`_ provides API-level documentation, as 
well as tutorials.

Support / Report Issues
-----------------------

All support requests and issue reports should be
`filed on Github as an issue <https://github.com/LaurentRDC/iris-ued/issues>`_.

License
-------

iris is made available under the MIT License. For more details, see `LICENSE.txt <https://github.com/LaurentRDC/iris-ued/blob/master/LICENSE.txt>`_.
