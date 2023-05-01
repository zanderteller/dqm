User Guide
==========

It's a good idea to walk through the :doc:`quick_start` guide first. Seeing DQM at work with a simple example can provide a helpful context for all of the detail offered here.

Overview
--------

Dynamic Quantum Mapping (DQM) is a unique system designed for exploring and understanding the intrinsic structure of high-dimensional numerical data.

DQM works on any given data set, using the mathematical framework of quantum mechanics, by creating a high-dimensional data-density map and then moving data points toward nearby regions of higher data density.

No assumptions are made about the underlying structure of the data.

Visual and numerical analysis of the resulting animated 'evolution' of the data can reveal both clusters and extended structures, leading to a rich understanding of relationships between different subsets of the data.

Among other uses, DQM can help with understanding how many models are needed for a given data set. (See :ref:`Interpreting Results <Interpreting and Using Results>` below for more on this subject.)

.. note::

	In code examples throughout this guide, we'll refer to an instance of the :class:`DQM <dqm.DQM>` class that we'll call 'dqm', created with the default constructor: ``dqm = DQM()``.

Visualization
-------------

Animation
^^^^^^^^^

Animated 3-D scatter plots (using DQM's :func:`plot_frames <dqm.utils.plot_frames>` utility function) are a regular feature of working with DQM.

If DQM is working with, say, 20 dimensions, then the motion created by the DQM evolution is occurring in all 20 dimensions.

**This means, crucially, that the motion observed in an animated 3-D plot is being driven by information from 20 dimensions, even though the visualization is only showing 3 dimensions.**

In this sense, DQM is *not* a 3-dimensional embedding, even though the regular use of 3-D plots may seem to suggest otherwise.

(This issue is revisited in the :ref:`PCA Transformations` section below.)

Visualization Tools
^^^^^^^^^^^^^^^^^^^

DQM's main visualization tool is the :func:`plot_frames <dqm.utils.plot_frames>` function, which uses the `Plotly <https://plotly.com/python/>`_ package.

DQM also has a 'backup' function, :func:`plot_frames_ipv <dqm.utils.plot_frames_ipv>`, which uses the `IPyVolume <https://ipyvolume.readthedocs.io>`_ package. (IPyVolume is less stable/mature than Plotly and may be buggy. However, it may handle large numbers of data points and/or frames much better than Plotly does.)

If you prefer to use your own visualization tools, you can. The core functionality of DQM itself does not rely on the plotting functions above. The final result of a DQM evolution (stored in the ``dqm.frames`` instance variable) is a 3-D array with shape: ``<number of rows/points x number of dimensions x number of frames>``. From there, you can visualize however you wish.

Basic Workflow
--------------

The starting point for DQM is always a 2-D real-valued matrix, with data points (samples) in the rows and dimensions (features) in the columns.

Raw data is stored in a DQM instance in ``dqm.raw_data``.

Data Preprocessing
^^^^^^^^^^^^^^^^^^

**Domain-Specific and Data-Specific Preprocessing**

There are any number of steps you may take to clean, process, and transform your data before exploring it with DQM.

One of many possible examples: when working with biological sequencing data (e.g., RNA-seq), a log2 transform is typically applied to the data before any further analysis.

**Giving Dimensions Equal Weight**

For DQM, every data dimension is simply a dimension, like any other, in a Euclidean space. If some of the dimensions in your data have far greater variance than others, the high-variance dimensions will dominate the structure that you see in DQM. That may be what you want.

If it's not what you want, consider normalizing the variance of each dimension in the data, in order to give all dimensions equal 'weight' in DQM. This can be done with something as simple as a z-score of each dimension (subtracting the mean and dividing by the standard deviation).

PCA Transformations
^^^^^^^^^^^^^^^^^^^

(*See the* `Wikipedia PCA page <https://en.wikipedia.org/wiki/Principal_component_analysis>`_ *for background on Principal Component Analysis.*)

First, note that DQM itself works in any Euclidean coordinate system, with any number of dimensions; using PCA is *not* intrinsic to DQM.

**Whether to Use PCA**

For DQM, PCA is essentially a rotation to a new coordinate system, where the 1st PCA dimension has the greatest variance in the data, the 2nd PCA dimension has the next greatest variance in the data, etc.

Using a PCA transformation as part of your DQM workflow is almost always a good idea, for two reasons:

First, PCA is useful for 'gentle' dimensionality reduction. A typical PCA analysis will only look at the first 2 (maybe 3) PCA dimensions; with DQM, however, dozens of PCA dimensions are often used, or even hundreds. **Using hundreds of PCA dimensions may still count as important dimensionality reduction if you're working with very high-dimensional data.**

Second, visualization of the first 3 dimensions of the PCA coordinate system allows us to pack as much information as possible into a single 3-D plot. (And, as mentioned above, animating this 3-D plot then presents information from the higher dimensions as well.) **For this reason, even though PCA is typically used as a dimensionality-reduction technique, it can and typically should be used with DQM even if you do no dimensionality reduction at all.**

Of course, you can also visualize higher PCA dimensions, not just the first 3. This can be interesting, but observing DQM evolution in the first 3 PCA dimensions is usually good enough.

**How Many PCA Dimensions To Use: The 'Spike' Model and Elbows**

A theory known as the 'spike model' essentially posits that a horizontal plateau in a plot of decreasing PCA eigenvalues represents a floor of noise in the data. This suggests a method for choosing a number of PCA dimensions to work with that will maximize information and minimize noise: namely, by choosing a number of dimensions at the 'elbow' of a PCA eigenvalue plot. (See the `Quick Start example <quick_start.html#run-pca>`_, where the elbow suggests that 4 PCA dimensions is enough to capture the most important structure in the data.)

If you find that the elbow is farther out than your computing resources will allow, it's a good idea to simply use as many dimensions as you can, to maximize the amount of information that you're working with. (*Computational complexity and memory usage for DQM are both essentially linear*, :math:`O(n)`, *in the number of dimensions being used.*)

**Working with PCA in the** :class:`DQM <dqm.DQM>` **Class**

The following code block (following the example in the :doc:`quick_start` guide) demonstrates choosing a number of PCA dimensions to work with:

.. code-block::

	dqm.verbose = True  # default True
	
	# run PCA, store results in instance, and display plots with PCA info
	dqm.run_pca()

	# choose an explicit number of dimensions (takes precedence if not None)
	# dqm.pca_num_dims = 18
	# OR...
	# choose a minimum proportion of total cumulative variance for the PCA dimensions to be used
	dqm.pca_var_threshold = 0.98

	dqm.pca_transform = True  # default True (if false, frame 0 will be a copy of the raw data)
	dqm.create_frame_0()

	print("In the DQM instance, 'frames' (which now stores frame 0) has shape:", dqm.frames.shape)

Creating Frame 0
^^^^^^^^^^^^^^^^

The :meth:`create_frame_0 <dqm.DQM.create_frame_0>` method creates the first 'frame' of the evolution and stores it in ``dqm.frames``. The following code:

.. code-block::

	dqm.create_frame_0()
	print(dqm.frames.shape)

... will print the shape of 'frames', which will be ``<number of rows x number of dimensions x 1>``. Note that 'frames' is 3-D; more frames will be added in the 3rd dimension during evolution.

If you're using a PCA transformation, the number of dimensions will be determined by the instance's PCA-transformation settings (see above).

If you're not using a PCA transformation, frame 0 will simply be a copy of the raw data (stored in ``dqm.raw_data``).

**Excluding Outliers**

If you haven't dealt with outliers already, now is a good time to check for them, in a visualization of frame 0 (by calling ``plot_frames(dqm.frames)``.)

Any extreme outliers in your data will cause the DQM map to become a relatively uninteresting illustration of just how different the outliers are from everything else. Thus, you may want to simply exclude them from the data set.

Choosing a Basis
^^^^^^^^^^^^^^^^

The 'basis' in DQM is a subset of data points that we choose from the data set. These basis points will be used to represent all other data points and will form the core of all DQM calculations. (*The word 'basis' here is referencing the idea from linear algebra; see the technical summary* *Understanding DQM* *for the technical details.*)

The size of the basis (i.e., the number of basis points) sets a 'resolution' for how much detail we can see in the landscape. A large basis is very computationally expensive (building frames is approximately :math:`O(n^3)`), so in order to use DQM efficiently it's a very good idea to follow these guidelines:

* Start with a smaller basis as you begin exploring a data set.
* Increase the basis size later when you need greater resolution.

For the typical computing power available in today's computers, here is a (very approximate) way to think about basis size:

* Small: up to 500 points
* Medium: 500 to 1,000 points
* Large: 1,000 or more points

The following code will choose a basis of size 100:

.. code-block::

	dqm.basis_size = 100
	dqm.choose_basis_by_distance()
	
Choosing the basis by distance means that the method is choosing the basis points to be as far away from each other as possible in the data space. (See :meth:`choose_basis_by_distance <dqm.DQM.choose_basis_by_distance>` for details.)

**Basis Overlap**

For any non-basis point, the 'overlap' of that point in the basis is a measure of how well the basis describes that point. For points far away from any basis point, the overlap will be small, which tells us that the chosen basis will not do a good job in modeling the behavior of that particular point.

Overlap for a given data point is always between 0 and 1, with 1 being a perfect representation. (All basis points have overlap of 1 in the basis.)

*For full technical details on basis overlaps, see the section on "Reconstruction of Wave Functions in the Eigenbasis" in the technical summary Understanding DQM.*

**Low-Overlap Points and Smoothness of Evolution**

How low is too low for basis-overlap values? This question does not have a clear-cut answer, and the 'right' answer may be context-dependent.

However, there is a practical heuristic. If any points 'jump' or 'snap' to a new location at the beginning of the evolution, this is a sign that the jumping points are not well represented in the basis. This problem can be fixed by either:

* increasing the basis size, or
* increasing the value of sigma (see below).

Of course, you can also treat the badly represented point as an outlier and simply exclude it.

Choosing DQM Parameters
^^^^^^^^^^^^^^^^^^^^^^^

**Sigma**

Sigma (:math:`\sigma`), introduced and explained here, is DQM's main tunable parameter.

When DQM builds a data-density map, the first step is to place a multidimensional Gaussian distribution around each data point. Sigma is the width of each Gaussian. There is only a single value for sigma; whatever value is chosen, every Gaussian around every data point has that same width (in every dimension).

The starting point for the overall DQM landscape is then simply all the Gaussians added together.

For any data set, the extremes are always the same:

* for very small sigma, each point has its own 'well' in the landscape, and nothing will move -- there will be no evolution at all.
* for very large sigma, all points will be within a single giant well and will immediately collapse together during evolution.

The values of sigma in between these extremes are where we can learn interesting things about the structure of the data set.

Importantly, note that 'small' and 'large' values of sigma are relative to the overall scale of the data set. (The :meth:`estimate_mean_row_distance <dqm.DQM.estimate_mean_row_distance>` method is a useful starting point for interesting, 'well scaled' values of sigma.)

**Choosing a "Minimum Good Value" of Sigma**

The ability of a set of basis points to describe non-basis points depends on sigma. For a fixed set of basis points and non-basis points, the basis will describe the non-basis points more and more accurately as sigma get bigger. This gives us a way to find a "minimum good value" of sigma that will adequately model the non-basis points in the data set.

As shown in the code block below, the :meth:`choose_sigma_for_basis <dqm.DQM.choose_sigma_for_basis>` method searches for the smallest value of sigma that satisfies the thresholds for minimum and mean overlap values for non-basis points:

.. code-block::

	dqm.overlap_min_threshold = 0.5  # default 0.5
	dqm.overlap_mean_threshold = 0.9  # default 0.9

	dqm.choose_sigma_for_basis()

	print('The DQM instance now has a stored value of sigma:', dqm.sigma)

Note that this method won't work if you're using a 'full' basis (i.e., all data points are in the basis) -- there need to be some non-basis point to work with.

**Mass**

The DQM 'mass' parameter controls the 'transparency' of the DQM landscape for a data point during evolution:

* for a very large mass, a point will get stuck in every local minimum in the landscape.
* for a very small mass, a point will pass through every barrier and head straight for the lowest point in the landscape.

Mass is typically set automatically, by a heuristic designed to make the landscape transparent to random density variations in uniform data. (See the :meth:`default_mass_for_num_dims <dqm.DQM.default_mass_for_num_dims>` method for details.)

The value of mass can be adjusted manually, but it's best to leave this as an 'advanced' technique.

*Note: however many dimensions are being used by DQM, it's always possible that the effective dimensionality of the data cloud could be significantly lower. The current heuristic described above makes no attempt to deal with this issue. DQM has room for improvement here.*

**Step**

The DQM 'step' parameter sets the time step between frames of the evolution. It has a default value of 0.1. (The 'units' of time here are arbitrary and unimportant.)

This parameter essentially never needs to be changed.

*Here's one case where you could be tempted to try, though: if you have an evolution where things are moving very smoothly and very slowly, increasing the time step slightly might be the easiest way to speed up the computation without losing (much) resolution in understanding the structure of the data. Don't say you weren't warned, though. Caveat emptor.*

Building Operators
^^^^^^^^^^^^^^^^^^

A quick recap -- once you've:

* Done any preprocessing of your data
* Chosen whether to use a PCA transformation, and how many PCA dimensions to use (DQM will default to using all PCA dimensions)
* Chosen a basis (DQM will default to a 'full' basis, using all data points)
* Chosen a value of sigma

... then you're ready to build the DQM operators, which will be used during evolution.

This step itself is extremely simple:

.. code-block::

	dqm.build_operators()

That's it. The operators are now stored in the instance, and you'll never need to touch them or change them. (*Note: this step can be slow for large data sets, especially when using a large basis.*)

If you want the gory mathematical details, see the technical summary *Understanding DQM*.

Here, we'll just give an extremely brief description of each operator:

* ``dqm.simt``: this is the transpose of the 'similarity' matrix, which is used to convert state vectors from the 'raw' basis to the eigenbasis.
* ``dqm.xops``: this is a 3-D tensor of position-expectation operators (each slice :math:`i` in the 3rd dimension is the operator matrix for the expected position of a point in the :math:`ith` dimension of the data.)
* ``dqm.exph``: this is the complex-valued 'evolution' operator (that is, the exponentiated Hamiltonian time-evolution operator, which converts a state vector from frame :math:`n` into a new state vector for frame :math:`n+1`)

Building Frames
^^^^^^^^^^^^^^^

Once we've built the operatorWe're now ready to proceed with the DQM evolution.

The :meth:`build_frames <dqm.DQM.build_frames>` method will build a specified number of frames (100 by default):

.. code-block::

	# build and add 50 new frames to the 'frames' instance variable
	dqm.build_frames(50)

The :meth:`build_frames_auto <dqm.DQM.build_frames_auto>` method will call :meth:`build_frames <dqm.DQM.build_frames>` repeatedly (in batches of 100 by default) until all points have stopped moving:

.. code-block::

	# build and add new frames, in batches of 50, until all points stop moving
	dqm.build_frames_auto(50)

:meth:`build_frames_auto <dqm.DQM.build_frames_auto>` uses the ``dqm.stopping_threshold`` parameter to decide when a point has stopped moving. A point is considered to have stopped if it moves less then ``stopping_threshold`` distance from one frame to the next. ``stopping_threshold`` is set automatically to ``dqm.mean_row_distance / 1e6`` but can be adjusted manually.

Iterating through Multiple Values of Sigma
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Coming soon...

Saving and Loading DQM instances
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Coming soon...

The run_simple Method of the DQM class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Coming soon...

Interpreting and Using Results
------------------------------

Coming soon...

Running New Points
------------------

Coming soon...

Additional Topics
-----------------

Coming soon...

Out-of-Distribution Issues
^^^^^^^^^^^^^^^^^^^^^^^^^^

Coming soon...

Working with Large Data Sets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Coming soon...

Working with Other Data Types
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Coming soon...

The Curse of Dimensionality
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Coming soon...

Non-Locality
^^^^^^^^^^^^

Coming soon...

Is DQM a Form of Machine Learning?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Coming soon...

