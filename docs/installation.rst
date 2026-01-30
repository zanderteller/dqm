Installation
============

Standard Installation
---------------------

Install DQM using pip:

.. code-block:: console

   $ pip install pydqm

That's it! The package includes pre-compiled binaries, so no compiler is needed.

**Requirements:** Python 3.10 or later. The only required dependency (numpy) is installed automatically.

**Supported platforms:** Linux (x86_64), macOS (Apple Silicon). Intel Mac users should build from source (see Development Installation below).

**macOS users:** You also need OpenMP installed:

.. code-block:: console

   $ brew install libomp

Optional Dependencies
---------------------

The core functionality of DQM does not require these packages, but they make working with DQM much easier:

.. code-block:: console

   $ pip install pydqm[viz]

This installs:

* **jupyterlab** - for working with DQM's example Jupyter notebooks
* **plotly** - for interactive animated 3D plotting
* **matplotlib** - for basic plotting

Verifying Installation
----------------------

To verify that DQM is installed correctly and the compiled library is working:

.. code-block:: python

   import dqm
   print("DQM version:", dqm.__version__)
   print("Compiled library loaded:", dqm.dqm_lib is not None)

You should see ``Compiled library loaded: True``. If you see ``False``, the compiled library failed to load - please report this as an issue.

----

Development Installation
------------------------

This section is for contributors or users who want to build DQM from source.

Cloning the Repository
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: console

   $ git clone https://github.com/zanderteller/dqm.git
   $ cd dqm

Building with CMake (Recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

DQM uses CMake for cross-platform builds. The easiest way to build and install for development:

.. code-block:: console

   $ pip install -e .

This will:

1. Compile the C++ library using CMake
2. Install DQM in "editable" mode (changes to Python code take effect immediately)

**Prerequisites:**

* A C++ compiler (g++, clang++, or MSVC)
* CMake 3.17 or later
* OpenMP (for parallel processing - see platform-specific notes below)

Building with Make (Alternative)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The original Makefile is still available for developers who prefer it:

.. code-block:: console

   $ cd cpp
   $ make all

Successful compilation puts the library in ``dqm/bin/``.

Platform-Specific Notes
^^^^^^^^^^^^^^^^^^^^^^^

**Linux**

OpenMP is typically available by default. If not:

.. code-block:: console

   # Ubuntu/Debian
   $ sudo apt-get install libgomp1

   # CentOS/RHEL
   $ sudo yum install libgomp

**macOS**

Apple does not ship OpenMP by default. Install it via Homebrew:

.. code-block:: console

   $ brew install libomp

The build system automatically detects Homebrew's libomp location on both Intel and Apple Silicon Macs.

**Windows**

Use Visual Studio. Open the solution file ``cpp/dqm_python.sln``, set configuration to "Release", and build.

Alternatively, CMake with MSVC should work (not extensively tested).

----

Troubleshooting
---------------

**"Compiled library loaded: False"**

The compiled library failed to load. This can happen if:

* The wheel doesn't include a binary for your platform
* There's a missing system library (like OpenMP)

Try installing from source (see Development Installation above).

**macOS: "Library not loaded: libomp.dylib" or similar**

DQM uses OpenMP for parallel processing. On macOS, install it via Homebrew:

.. code-block:: console

   $ brew install libomp

Note: Homebrew will show a "keg-only" warning - this is normal and does not affect DQM.

**Reporting Issues**

Please report installation issues on the DQM GitHub `Issues <https://github.com/zanderteller/dqm/issues>`_ page, or start a discussion on the `Discussions <https://github.com/zanderteller/dqm/discussions>`_ page.
