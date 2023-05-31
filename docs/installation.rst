Installation
============

An 'official' Python release of DQM (via pip, conda, etc.) is not yet implemented. We've tried to make manual installation as easy as possible.

Please report any issues on the DQM GitHub `Issues <https://github.com/zanderteller/dqm/issues>`_ page. (Or start on the DQM GitHub `Discussions <https://github.com/zanderteller/dqm/discussions>`_ page if you're not sure how to report the issue, whether the issue is really specific to DQM, etc.)

Python Dependencies
-------------------

You'll need to install some python packages that DQM expects to find. (You can install via pip, conda, etc., as you prefer.)

Hard Dependencies
^^^^^^^^^^^^^^^^^

*These packages are required: DQM won't work without them.*

* numpy

Soft Dependencies
^^^^^^^^^^^^^^^^^

*The core functionality of DQM does not need any of these packages, but working with DQM will be much easier with them.*

* jupyterlab (for working with DQM's example Jupyter notebooks)
* plotly (for interactive animated 3-D plotting)
* matplotlib (for basic plotting)

Cloning the DQM Repository
--------------------------

Go to the `DQM GitHub <https://github.com/zanderteller/dqm>`_ page and clone the repository to your machine. (Use the green 'Code' button at top right of the page.)

For right now, you probably want the 'main' branch, which is the default on the GitHub page. (There is a tagged version 0.1.0, but things may change quickly at first...)

Setting Your PYTHONPATH
-----------------------

Wherever you put your local clone on your machine, the clone's top-level main folder (containing the README file), which we'll call ``<your clone main folder>``, will ultimately need to be in your PYTHONPATH.

If you use pip generally to install packages, you can copy the DQM package to your 'standard' site-package location as shown below. (**NOTE:** be sure to do this *AFTER* building the compiled C++ library file.)

.. code-block:: console

   $ cd <your clone main folder>
   $ pip install .

Compiling the C++ Library Code
------------------------------

Important core functions in DQM are implemented in C++, which you'll need to compile on your machine. (*DQM has Python-only implementations of all of these core functions, but the Python versions will be unusably slow for all but the smallest data sets.*)

The C++ code uses the OpenMP library for parallel processing. (*So, be aware that a large job will eat up all of your machine's processing power...*)

For all the platforms below, successful compilation will automatically put a compiled library file in ``<your clone main folder>/dqm/bin``, which is where the DQM Python code expects to find it.

Linux
^^^^^

Compilation from the command line should be simple. (Tested successfully on Ubuntu 22.04.)

.. code-block:: console

   $ cd <your clone main folder>/cpp
   $ make all

To be sure everything worked, check for the compiled library in its final location: ``<your clone main folder>/dqm/bin/dqm_python.so``.

Windows
^^^^^^^

Use `Visual Studio <https://visualstudio.microsoft.com/>`_. In the Visual Studio application:

* Open the DQM Visual Studio solution file: ``<your clone main folder>/cpp/dqm_python.sln``.
* Make sure the Configuration dropdown (in the main toolbar near the top of the window) is set to 'Release' (and *not* 'Debug').
* From the 'Build' menu, run the 'Build Solution' command.

To be sure everything worked, check for the compiled library in its final location: ``<your clone main folder>/dqm/bin/dqm_python.dll``.

Mac
^^^

**OpenMP**

Unfortunately, Apple officially parted ways with OpenMP a while back. Solutions are possible, but it may or may not be easy. And remember that the exact solution may depend on which kind of chip you have: Intel or ARM (M).

The following was a successful solution on an M1 Mac:

This `R-Project for Mac <https://mac.r-project.org/openmp/>`_ page has prebuilt OpenMP binaries (for both Intel and ARM). As described on that page, we downloaded the lastest version and moved the prebuilt files to the following locations:

.. code-block:: console

   /usr/local/lib/libomp.dylib
   /usr/local/include/ompt.h
   /usr/local/include/omp.h
   /usr/local/include/omp-tools.h

Note that the DQM Makefile (``<your clone main folder>/cpp/Makefile``) expects to find the above files in those exact locations.

**g++ Compiler**

The DQM Makefile expects to use the g++ compiler, which you may need to install (via XCode, homebrew, or other means).

You're welcome to try other compilers as well (we certainly didn't test every option), by changing the line ``CXX:=g++`` in the Makefile.

**Compiling**

Once you've cleared those hurdles, compilation from the command line should be simple:

.. code-block:: console

   $ cd <your clone main folder>/cpp
   $ make all

To be sure everything worked, check for the compiled library in its final location: ``<your clone main folder>/dqm/bin/dqm_python.dylib``.

|
