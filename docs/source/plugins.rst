.. include:: references.txt

.. _plugins:

.. currentmodule:: iris

****************
Dataset Plug-ins 
****************

To use your own raw data with :mod:`iris`, a plug-in functionality is made available.

Plug-ins are Python modules that implement a subclass of :class:`AbstractRawDataset`, and should be placed in :file:`~/iris_plugins` (:file:`C:\\Users\\UserName\\iris_plugins` on Windows). Subclasses
of :class:`AbstractRawDataset` are automatically detected by :mod:`iris` and can be used via the GUI.

Installed plug-ins can be imported from :mod:`iris.plugins`::

  from iris.plugins import DatasetSubclass

which would work if the :class:`DatasetSubclass` is defined in the file :file:`~/iris_plugins/<anything>.py`. Example plug-ins is available `here <https://github.com/LaurentRDC/iris-ued/tree/master/example_plugins>`_

Installing a plug-in
--------------------

To install a plug-in that you have written in a file named :file:`~/myplugin.py`::

  import iris
  iris.install_plugin('~/myplugin.py')

Installing a plug-in in the above makes it immediately available.

.. autosummary::
  :toctree: functions/

  install_plugin

******************************
Subclassing AbstractRawDataset
******************************

To take advantage of :mod:`iris`'s :class:`DiffractionDataset` and :class:`PowderDiffractionDataset`,
an appropriate subclass of :class:`AbstractRawDataset` must be implemented. This subclass can then be fed
to :meth:`DiffractionDataset.from_raw` to produce a :class:`DiffractionDataset`.

How to assemble a AbstractRawDataset subclass
---------------------------------------------

Ultrafast electron diffraction experiments typically have multiple *scans*. Each scan consists
of a time-delay sweep. You can think of it as one scan being an experiment, and so each dataset
is composed of multiple, equivalent experiments.

To subclass :class:`AbstractRawDataset`, the method :func:`AbstractRawDataset.raw_data` must minimally implemented.
It must follow the following specification:

.. automethod::
  AbstractRawDataset.raw_data

For better performance, or to tailor data reduction to your data acquisition scheme,
the following method can also be overloaded:

.. automethod::
  AbstractRawDataset.reduced

AbstractRawDataset metadata
---------------------------

:class:`AbstractRawDataset` subclasses automatically include the following metadata:

* ``date`` (`str`): Acquisition date. Date format is up to you.
* ``energy`` (`float`): Electron energy in keV.
* ``pump_wavelength`` (`int`): photoexcitation wavelength in nanometers.
* ``fluence`` (`float`): photoexcitation fluence :math:`\text{mJ}/\text{cm}**2`.
* ``time_zero_shift`` (`float`): Time-zero shift in picoseconds. 
* ``temperature`` (`float`): sample temperature in Kelvins.
* ``exposure`` (`float`): picture exposure in seconds.
* ``resolution`` (2-`tuple`): pixel resolution of pictures.
* ``time_points`` (`tuple`): time-points in picoseconds.
* ``scans`` (`tuple`): experimental scans.
* ``camera_length`` (`float`): sample-to-camera distance in meters.
* ``pixel_width`` (`float`): pixel width in meters.
* ``notes`` (`str`): notes.

Subclasses can add more metadata or override the current metadata with new defaults.

All proper subclasses of :class:`AbstractRawDataset` are automatically added to the possible raw dataset formats
that can be loaded from the GUI.