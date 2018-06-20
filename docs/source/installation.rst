.. include:: references.txt

.. _installation:

************
Installation
************

Requirements
============

.. note::

    Users are strongly recommended to manage these dependencies with the
    excellent `Intel Distribution for Python <https://software.intel.com/en-us/intel-distribution-for-python>`_
    which provides easy access to all of the above dependencies and more.

:mod:`iris` works on Linux, Mac OS X and Windows. It requires Python 3.6+.

GUI Libraries
=============

The GUI frontend of :mod:`iris` requires PyQt5. This requirement cannot, at this
time, be substituted with Pyside.

Install iris
============

:mod:`iris` is available on PyPI as **iris-ued**::

    python -m pip install iris-ued

Iris is also available on the conda-forge channel::

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

Testing
=======

If you want to check that all the tests are running correctly with your Python
configuration, type::

    python setup.py test
