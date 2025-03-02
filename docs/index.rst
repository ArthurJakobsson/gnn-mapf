.. SSIL GNN MAPF with Simulation and DAgger Pipeline documentation master file, created by
   sphinx-quickstart on Sun Feb 23 09:52:00 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

SSIL GNN MAPF with Simulation and DAgger Pipeline's documentation
=============================================================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


Installation
============

To install the repository, run the following command:

.. code-block:: bash

   https://github.com/ArthurJakobsson/gnn-mapf.git

Please switch branch to **GNN-development** to access the latest code. *GNN-development2* is an active branch that is being used for development.

To install the required packages, run the following command:

.. code-block:: bash

   pip install -r requirements.txt

Understanding the Repository Layout
====================================

This reposity is divided into 

#. *data_collection*

   #. This section controls the main flow of the data_collection process. It controls the calling of the simulator and eecbs in parallel. 
   #. The data should also live in this folder.

#. *slurm DAgger implementation*

   #. 

#. *benchmarking*

#. *gnn*

#. *map_generation*

#. *map_generation*

Understanding Data Structure
====================================

The data is organized as multiple npz files. There should be one containing the maps, the bd values for the maps, and then the paths.
Because the paths get very large this had to be split into multiple files to reduce load time. This gets processed by whatever files load the npzs.
There should also lay a folder with the benchmark data and you can toggle between the held and unheld dataset. Place the maps and scens in respectively named
folders under the benchmark folder.

.. note::

   This project is under active development and has many moving parts. Please feel free to email me at ajakobss@cmu.edu if you have any questions or would like to contribute.
