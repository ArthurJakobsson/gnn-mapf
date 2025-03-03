.. SSIL GNN MAPF with Simulation and DAgger Pipeline documentation master file, created by
   sphinx-quickstart on Sun Feb 23 09:52:00 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

SSIL GNN MAPF with Simulation and DAgger Documentation
=============================================================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   data_collection
   slurm_implementation
   benchmarking
   gnn
   simulator
   new_map_generation


.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :doc:`data_collection`
.. * :doc:`slurm_implementation`
.. * :doc:`benchmarking`
.. * :doc:`gnn`
.. * :doc:`simulator`
.. * :doc:`new_map_generation`
.. * :ref:`search`


Installation
============

To install the repository, run the following command:

.. code-block:: bash

   https://github.com/ArthurJakobsson/gnn-mapf.git

Please switch branch to **GNN-development** to access the latest code. *GNN-development2* is an active branch that is being used for development.

To install the required packages, run the following command

.. code-block:: bash

   pip install -r requirements.txt

Understanding the Repository Layout
====================================

This reposity is divided into 

#. :doc:`data_collection`

   #. This section controls the main flow of the data_collection process. It controls the calling of the simulator and eecbs in parallel. 
   #. The data should also live in this folder.

#. :doc:`slurm_implementation`

   #. This is to run large tests on a cluster. It is organized so that the files will write their own sh files to start the next process. Run the master runner, or the specific version you wish to test. This will call the files named *run_* and then those will call the next file needed and they chain together. There is a hardcoded limit to the number of dagger cyles, but you can also let it run infinitely and stop it when needed. The code should be able to resume from the last position with relative ease because checkpoint files are saved. 

#. :doc:`benchmarking`

   #. This is to benchmark the models. It has files to generate plots and visualizations and also to compare to other benchmarks such as EPH.

#. :doc:`gnn`

   #. This is the main folder for the GNN. It contains the model, the training loop, the data processing, and the evaluation.
   #. It also contains a file to split the bd files.
   #. It also contains **simulator.py** which is the next most important file and is described in the next section.

#. :doc:`simulator`

   #. The simulator is listed under *gnn/simulator.py*. This file runs simulations and is used for the DAgger loop, benchmarking and many other processes.
   #. *This is where to find the integration of CS-PIBT.*

#. :doc:`new_map_generation`

   #. We created some custom maps and scens for experimenting and testing. This folder creates those maps and new scens that are valid given the inputted map. We created more scens beyond those provided by the MovingAI Benchmark here.

Understanding Data Structure
====================================

The data is organized as multiple npz files. There should be one containing the maps, the bd values for the maps, and then the paths.
Because the paths get very large this had to be split into multiple files to reduce load time. This gets processed by whatever files load the npzs.
There should also lay a folder with the benchmark data and you can toggle between the held and unheld dataset. Place the maps and scens in respectively named
folders under the benchmark folder.

When training happens, the dataloader will load the data from the npz files and then create pt files with information
about each training instance seperately. Beware: this will create several thousands of pt files. These files are indexed and loaded seperately by the model when the index is needed.

.. note::

   This project is under active development and has many moving parts. Please feel free to email me at ajakobss@cmu.edu if you have any questions or would like to contribute.
