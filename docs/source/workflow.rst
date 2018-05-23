.. include:: references.txt

.. _workflow:

*******************************
Using the GUI: typical workflow
*******************************

Startup
-------

To start the GUI from the command line::

    python -m iris

The first screen is shown below.

.. image:: startup.png

Loading raw data
----------------
The file menu can be used to load raw data. Depending on the installed plugins, options will be available.

.. image:: load_raw.png
    :scale: 50 %
    :align: center

.. image:: raw_data.png
    :align: center

Raw data controls are available to the right.

Data reduction
--------------

Once raw data is loaded, the following option becomes available:

.. image:: reduction_dialog.png
    :align: center

This opens the data reduction dialog.

.. image:: reduction_window.png
    :align: center

Parts of the data can be masked. To add a mask, use the controls on the top of the dialog. Masks can be moved and resized. 
Note that all images will be masked, so this is best for beam blocks, known hot pixels, etc. 

.. image:: reduction_mask.png
    :align: center

A preview of the mask can be generated:

.. image:: mask_preview.png
    :align: center

Once you are satisfied with the processing parameters, the 'Launch processing' button will open a file dialog so that 
you can choose where to save the reduced HDF5 file. Processing might take a few minutes.

Data exploration
----------------

Once processing is complete, the resulting diffraction dataset will be loaded. New controls will be available. 

.. image:: processed_view.png
    :align: center

The 'Show/hide peak dynamics' button can be toggled. Doing so allows for the exploration of the time-evolution of the data.

.. image:: peak_dynamics_single.png
    :align: center

When a diffraction dataset is loaded, new options become available.

.. image:: dataset_options.png
    :align: center

One of these options, 'Compute angular averages', is best suited for polycrystalline diffraction. It opens the following dialog:

.. image:: azimuthal_dialog_1.png
    :align: center

Drag and resize the red circle so it coincides with a diffraction ring. This will allow for the determination of the diffraction center. 
The averaging will happen after clicking 'Promote'. This might take a few minutes.

Polycrystalline data exploration
--------------------------------

After the azimuthal averages have been computed, a new section of the GUI will be made available, with additional controls.

.. image:: poly_view.png
    :align: center

The top screen shows the superposition of all radial profiles. Dragging the yellow lines allows for exploration of time-evolution 
on the bottom screen. Note that the trace colors on the top are associated with the time-points and colors of the bottom image.

.. image:: poly_view_2.png
    :align: center

The baseline can be removed using the controls on the right. You can play with the baseline parameters and compute a baseline many times 
without any problems.

.. image:: poly_view_3.png
    :align: center

The 'Relative' button toggles a different point-of-view: the top image will now show the deviation of all time-traces with respect to
the average of all traces before time-zero.

.. image:: poly_view_4.png
    :align: center

Polycrystalline scattering vector calibration
---------------------------------------------

On the above images, the scattering vector range is meaningless. To calibrate the scattering vector range
based on a known structure, select the 'Calibrate scattering vector' option from the 'Dataset' menu.

.. image:: calibrate_option.png
    :scale: 50 %
    :align: center

This opens the calibration dialog.

.. image:: calibration_dialog.png
    :align: center

You must either select a structure file (CIF) or one of the built-in structures. Once a structure is selected, it's description 
will be printed on the screen. Make sure this is the crystal structure you expect.

Then, drag the left and right yellow bars on two diffraction peaks with known Miller indices. Click 'Calibrate' to calibrate the 
scattering vector range.

.. image:: calibration_dialog_2.png
    :align: center

The resulting pattern makes more sense.

.. image:: poly_view_5.png
    :align: center