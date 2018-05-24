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

You can take a look at `h5py's documentation <http://docs.h5py.org/en/latest/>`_ to familiarize yourself
with :class:`h5py.File`.

You can also use other HDF5 bindings to inspect :class:`DiffractionDataset` instances.

Creating a :class:`DiffractionDataset`
--------------------------------------

An easy way to create a DiffractionDataset is through the :meth:`DiffractionDataset.from_collection` method, which
saves diffraction patterns and metadata:

.. automethod:: DiffractionDataset.from_collection
    :noindex:

The required metadata that must be passed to :meth:`DiffractionDataset.from_collection` is also listed in
:attr:`DiffractionDataset.valid_metadata`. Metadata not listed in :attr:`DiffractionDataset.valid_metadata`
will be *ignored*.

An other possibility is to create a :class:`DiffractionDataset` from a :class:`AbstractRawDataset` subclass using the 
:meth:`DiffractionDataset.from_raw` method :

.. automethod:: DiffractionDataset.from_raw


Important Methods for the :class:`DiffractionDataset`
-----------------------------------------------------

The following three methods are the bread-and-butter of interacting with data. See the API section
for a complete description.

.. automethod:: DiffractionDataset.diff_data
    :noindex:
.. automethod:: DiffractionDataset.diff_eq
    :noindex:
.. automethod:: DiffractionDataset.time_series
    :noindex:

The :class:`PowderDiffractionDataset` object
============================================

For polycrystalline data, we can define more data structures and methods. A :class:`PowderDiffractionDataset` is a strict
subclass of a :class:`DiffractionDataset`, and hence all methods previously described are also available.

Specializing a :class:`DiffractionDataset` object into a :class:`PowderDiffractionDataset` is done as follows::

    from iris import PowderDiffractionDataset
    dataset_path = 'C:\\path_do_dataset.hdf5'   # DiffractionDataset already exists

    with PowderDiffractionDataset.from_dataset(dataset_path, center) as dset:
        # Do computation

Important Methods for the :class:`PowderDiffractionDataset`
----------------------------------------------------------

The following methods are specific to polycrystalline diffraction data. See the API section
for a complete description.

.. automethod:: PowderDiffractionDataset.powder_eq
    :noindex:
.. automethod:: PowderDiffractionDataset.powder_data
    :noindex:
.. automethod:: PowderDiffractionDataset.powder_calq
    :noindex:
.. automethod:: PowderDiffractionDataset.compute_baseline
    :noindex: