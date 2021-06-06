Release 5.3.1
-------------

* Fixed an issue where `iris` would use up all available memory for datasets with a large number of time-delays (>500)
* Releases are now automatically performed using Github Actions
 
Release 5.3.0
-------------

* Added the :meth:`DiffractionDataset.mask_apply` to modify the diffraction pattern mask.
* The center of diffraction is now calculated and updated as needed automatically.
* Better handling of write permissions.
* Added the :class:`MigrationWarning` and :class:`MigrationError` classes. Warnings/errors of these classes tell the user that migration should
  be performed. This is automatically done by opening a :class:`DiffractionDataset` with writing permission. The GUI does this
  automatically.
* Windows installers are now built with pynsist/NSIS instead of PyInstaller (#15).
* `Support for Python 3.6 and NumPy<1.17 has been dropped <https://numpy.org/neps/nep-0029-deprecation_policy.html>`_
* Fixed an issue where creating the plug-in directory would rarely fail.

5.2.5
-----

* Parallel operations on datasets (via HDF5 single-writer multiple-reader) is now possible on all platforms. 
* Code snippets in documentation are now tested for correctness.
* Migration of test infrastructure to pytest.
* Tests are now included in source distributions.

5.2.4
-----

* Added support for h5py 3.*.

5.2.3
-----

* Re-licensing `iris-ued` to GPLv3.
* Changed the default colormap for processed datasets, to visually distinguish between raw and processed data viewers
* Added support for Python 3.9

5.2.2
-----

* Fixed an issue where a broken plug-in would crash Iris. Instead, broken plug-ins will not be loaded.

5.2.1
-----

* Added the `DiffractionDataset.time_series_selection` method, which allows to create time-series integrated across an arbitrary momentum-space selection mask.
  This allows to create time-series from shapes that are not rectangular, at the expense of performance.
* Added a few methods to create selection masks: `DiffractionDataset.selection_rect`, `DiffractionDataset.selection_disk`, and `DiffractionDataset.selection_ring`.
* Added the ability to show/hide dataset control bar;
* Added the ability to export time-series data in CSV format;

* Fixed an issue where calculations of time-series, relative to pre-time-zero, would raise an error.
* Symmetrization dialog is no longer in "beta".

5.2.0
-----

* Official support for Linux.
* Plug-ins installed via the GUI can now be used right away. No restarts required.
* Added the `iris.plugins.load_plugin` function to load plug-ins without installing them. Useful for testing.
* Plug-ins can now have the ``display_name`` property which will be displayed in the GUI. This is optional and backwards-compatible.
* Siwick Research Group-specific plugins were removed. They can be found here: https://github.com/Siwick-Research-Group/iris-ued-plugins
* Switched to Azure Pipelines for continuous integration builds;
* Added cursor information (position and image value) for processed data view;

* Fixed an issue where very large relative differences in datasets would crash the GUI displays;
* Fixed an issue where time-series fit would not display properly in fractional change mode;

5.1.3
-----

* Added logging support for the GUI component. Logs can be reached via the help menu
* Added an update check. You can see whether an update is available via the help menu, as well as via the status bar.
* Added the ability to view time-series dynamics in absolute units AND relative change.
* Pinned dependency to scikit-ued, to prevent upgrade to scikit-ued 2.0 unless appropriate.
* Pinned dependency to npstreams, to prevent upgrade to npstreams 2.0 unless appropriate.

5.1.2
-----

* Fixed an issue where the QDarkStyle internal imports were absolute.

5.1.1
-----

* Fixed an issue where data reduction would freeze when using more than one CPU;
* Removed the auto-update mechanism. Update checks will run in the background only;
* Fixed an issue where the in-progress indicator would freeze;
* Moved tests outside of source repository;
* Updated GUI stylesheet to QDarkStyle 2.6.6;

5.1.0
-----

* Added explicit support for Python 3.7;
* Usability tweaks, for example more visible mask controls;
* Added the ability to create standalone executables via PyInstaller;
* Added the ability to create Windows installers;

5.0.5.1
-------

* Due to new forced image orientation, objects on screens were not properly registered (e.g. diffraction center finder).

5.0.5
-----

* Added the ability to fit exponentials to time-series;
* Added region-of-interest text bounds for easier time-series exploration
* Enforced PyQtGraph to use row-major image orientation
* Datasets are now opened in read-only mode unless absolutely necessary. This should make it safer to handler multiple instances of iris at the same time.

5.0.4
-----

* Better plug-in handling and command-line interface.

5.0.3
-----

The major change in this version is the ability to guess raw dataset formats using the `iris.open_raw` function. 
This allows the possibility to start the GUI and open a dataset at the same time.

5.0.2
-----

The package now only has dependencies that can be installed through `conda`

5.0.1
-----

This is a minor bug-fix release that also includes user interface niceties (e.g. link to online documentation) and user 
experience niceties (e.g. confirmation message if you forget pixel masks).

5.0.0
-----

This new version includes a completely rewritten library and GUI front-end. Earlier datasets will need to be re-processed.
New features:

* Faster performance thanks to better data layout in HDF5;
* Plug-in architecture for various raw data formats;
* Faster performance thanks to npstreams package;
* Easier to extend GUI skeleton;
* Online documentation accessible from the GUI;
* Continuous integration.