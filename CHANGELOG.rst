Changelog
=========

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