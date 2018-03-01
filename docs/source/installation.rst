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

**iris** works on Linux, Mac OS X and Windows. It requires Python 3.5+.

GUI Libraries
=============

The GUI frontend of iris requires PyQt5. Since PyQt5 is not available on the Python Package Index,
you will need to install it either via conda or using an installer. This requirement cannot, at this
time, be substituted with Pyside.

Install iris
============

You can install the latest developer version of iris by cloning the git
repository::

    git clone https://github.com/LaurentRDC/iris.git

...then installing the package with::

    cd iris
    python setup.py install


Testing
=======

If you want to check that all the tests are running correctly with your Python
configuration, type::

    python setup.py test
