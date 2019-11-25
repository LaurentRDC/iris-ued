.. include:: references.txt

.. _installation:

************
Installation
************

Standalone Installation
=======================

Starting with `iris` 5.1.0, **standalone Windows installers and executables are available**. You can find them on
the `GitHub release page <https://github.com/LaurentRDC/iris-ued/releases/latest/>`_.

The standalone installers and executables make the installation of `iris` completely separate from any other 
Python installation. This method should be preferred, unless Python scripting using the `iris` library is required.

Installing the Python Package
=============================

If you want to script using `iris` data structures and algorithms, you need to install the `iris-ued` package.

.. note::

    Users are strongly recommended to manage these dependencies with the
    excellent `Intel Distribution for Python <https://software.intel.com/en-us/intel-distribution-for-python>`_
    which provides easy access to all of the above dependencies and more.

:mod:`iris` is available on PyPI as **iris-ued**::

    python -m pip install iris-ued

:mod:`iris` is also available on the conda-forge channel::

    conda config --add channels conda-forge
    conda install iris-ued

You can install the latest developer version of :mod:`iris` by cloning the git
repository::

    git clone https://github.com/LaurentRDC/iris-ued.git

...then installing the package with::

    cd iris-ued
    python setup.py install

In Python code, :mod:`iris` can be imported as follows ::

    import iris

Test data
=========

Test reduced datasets are made available by the Siwick research group. The data can be accessed on the 
`public data repository <http://www.physics.mcgill.ca/siwicklab/publications.html>`_

Testing
=======

If you want to check that all the tests are running correctly with your Python
configuration, type::

    python setup.py test
