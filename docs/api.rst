.. include:: references.txt

.. _api:

*************
Reference/API
*************

.. currentmodule:: iris

Opening raw datasets
====================

To open any raw dataset, take a look at the :func:`open_raw` function.

.. autofunction:: open_raw

Raw Dataset Classes
===================

.. autoclass:: AbstractRawDataset
    :members:
    

Diffraction Dataset Classes
===========================


:class:`DiffractionDataset`
---------------------------

.. autoclass:: DiffractionDataset
    :show-inheritance:
    :members:

:class:`PowderDiffractionDataset`
---------------------------------

.. autoclass:: PowderDiffractionDataset
    :show-inheritance:
    :members:

Migrating older datasets
------------------------

The work "migration" here is used to signify that a particular dataset
needs to be *migrated* to a slightly updated form. This is done automatically
if the dataset is opened with write permissions.

.. autoclass:: MigrationWarning
    :show-inheritance:

.. autoclass:: MigrationError
    :show-inheritance: