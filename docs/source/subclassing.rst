.. include:: references.txt

.. _subclassing:

**************************
Subclassing RawDatasetBase
**************************

.. currentmodule:: iris

To take advantage of :mod:`iris`'s :class:`DiffractionDataset` and :class:`PowderDiffractionDataset`,
an appropriate subclass of :class:`RawDatasetBase` must be implemented. This subclass can then be fed
to :func:`process` to produce a :class:`DiffractionDataset`.

How to assemble a RawDatasetBase subclass
-----------------------------------------

Ultrafast electron diffraction experiments typically have multiple *scans*. Each scan consists
of a time-delay sweep. You can think of it as one scan being an experiment, and so each dataset
is composed of multiple, equivalent experiments.

With this in mind, :class:`RawDatasetBase` subclasses must implement the following attributes:

* ``fluence`` (`float`) : photoexcitation fluence [mJ/cm^2]
* ``resolution`` (`tuple`): Dataset resolution [px], e.g. ``(2048, 2048)``
* ``energy`` (`float`): Electron energy [keV]
* ``nscans`` (`iterable`): Scan numbers starting from 1.
* ``time_points`` (`iterable`): Iterable of time-delay values [ps]

Optional properties that are supported:

* ``acquisition_date`` (`str`): Date of dataset acquisition. Format is unimportant.
* ``current`` (`float`): Eletron beam current [pA].
* ``exposure`` (`float`): Exposure time [s].
* ``pumpon_background`` (`~numpy.ndarray`): background to be removed from all pictures in which
  the laser pump is present. Default value is an array of zeros. 
* ``pumpoff_background`` (`~numpy.ndarray`): background to be removed from all pictures in which
  the laser pump is NOT present. Default value is an array of zeros. 

The following method is required:

* ``raw_data_filename(self, timedelay, scan = 1, **kwargs)`` : returns path to a raw picture at a specific time-delay and scan.
  Default value for ``scan`` must be 1.

Other methods and attributes can be added at will, but the above is the minimum.

RawDatasetBase subclasses
-------------------------

Based on the :meth:`raw_data_filename` method, any subclass of :class:`RawDatasetBase` will have the following
method automatically available:

    .. automethod::
        RawDatasetBase.raw_data