.. include:: references.txt

.. _dataset:

****************
Datasets in Iris
****************

.. currentmodule:: iris

The :class:`DiffractionDataset` object
======================================

The :class:`DiffractionDataset` object is the basis for :mod:`iris`'s interaction with
ultrafast electron diffraction data. :class:`DiffractionDataset` objects are simply
HDF5 files with a specific layout, and associated methods::

    from iris import DiffractionDataset
    import h5py

    assert issubclass(DiffractionDataset, h5py.File)    # yep

You can take a look at :ref:`h5py's documentation <http://docs.h5py.org/en/latest/>` to familiarize yourself
with :class:`h5py.File`.

You can also use other HDF5 bindings to inspect :class:`DiffractionDataset` instances.

Creating a :class:`DiffractionDataset`
--------------------------------------

An easy way to create a DiffractionDataset is through the :meth:`DiffractionDataset.from_collection` method, which
saves diffraction patterns and metadata:

.. automethod:: DiffractionDataset.from_collection

The required metadata that must be passed to :meth:`DiffractionDataset.from_collection` is also listed in
:attr:`DiffractionDataset.required_metadata`. Valid optional metadata is listed in :attr:`DiffractionDataset.optional_metadata`.

An other possibility is to create a :class:`DiffractionDataset` from a :class:`RawDatasetBase` subclass using the 
:meth:`DiffractionDataset.from_raw` method :

.. automethod:: DiffractionDataset.from_raw



Important Methods for the :class:`DiffractionDataset`
-----------------------------------------------------

The following three methods are the bread-and-butter of interacting with data:

.. automethod:: DiffractionDataset.averaged_data
.. automethod:: DiffractionDataset.averaged_error
.. automethod:: DiffractionDataset.time_series

The :class:`PowderDiffractionDataset` object
============================================

For polycrystalline data, we can define more data structures and methods. A :class:`PowderDiffractionDataset` is a strict
subclass of a :class:`DiffractionDataset`, and hence all methods previously described are also available.

Specializing a :class:`DiffractionDataset` object into a :class:`PowderDiffractionDataset` is done as follows::

    from iris import PowderDiffractionDataset
    dataset_path = 'C:\\path_do_dataset.hdf5'   # DiffractionDataset already exists

    with PowderDiffractionDataset(dataset_path) as dset:
        dset.compute_angular_averages(center)