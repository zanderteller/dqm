User Guide
==========

It's a good idea to walk through the :doc:`quick_start` guide first. Seeing DQM at work with a simple example can provide a helpful context for all of the detail offered here.

*There's also a demo Jupyter notebook, using DQM on real data, in* ``notebooks/demo_real_data_1.ipynb`` *in the DQM repository.*

Overview
--------

Dynamic Quantum Mapping (DQM) is a unique system designed for exploring and understanding the intrinsic structure of high-dimensional numerical data.

DQM works on any given data set, using the mathematical framework of quantum mechanics, by creating a high-dimensional data-density map and then moving data points toward nearby regions of higher data density.

No assumptions are made about the underlying structure of the data.

Visual and numerical analysis of the resulting animated 'evolution' of the data can reveal both clusters and extended structures, leading to a rich understanding of relationships between and within different subsets of the data.

Among other uses, DQM can help with understanding how many models are needed for a given data set. (See the section :ref:`How Many Models for Your Data?` below.)

.. note::

   In code examples throughout this guide, we'll refer to an instance of the :class:`DQM <dqm.DQM>` class that we'll call 'dqm_obj', created with the default constructor: ``dqm_obj = DQM()``.

Why Quantum Mechanics?
----------------------

**First of all: as a user of DQM, you don't need to know anything about quantum mechanics.**

Now, if you're curious about why DQM uses quantum mechanics under the hood...

The very short answer: non-local gradient descent provides an elegant solution to the problem of non-convex gradient descent.

The slightly longer answer: in classical mechanics, any object moving downhill in a landscape (a ball, a pebble, a drop of water) only 'sees' the landscape locally, so it can get stuck in any local minimum (any pocket, divot, dimple) in the terrain. In quantum mechanics, however, a quantum particle 'sees' the entire landscape -- you can think of this as the particle being 'attracted' to a lower part of the landscape, even if there's a hill in between. In fact, quantum 'tunneling' is the process of the particle traveling 'through' the hill to get to the lower place. DQM uses these features of quantum mechanics to ignore the uninteresting little details of the terrain as points move downhill, thus revealing the overall structure of the landscape.

*A full explanation of the underlying mathematics is available in the technical summary* `Understanding DQM <https://github.com/zanderteller/dqm/blob/main/docs/Understanding%20DQM.pdf>`_.)

Visualization
-------------

Animation
^^^^^^^^^

Animated 3D scatter plots (using DQM's :func:`plot_frames <dqm.utils.plot_frames>` utility function) are a regular feature of working with DQM.

If DQM is working with, say, 20 dimensions, then the motion created by the DQM evolution is occurring in all 20 dimensions. **This means, crucially, that the motion observed in an animated 3D plot is being driven by information from 20 dimensions, even though the visualization is only showing 3 dimensions.** In this sense, DQM is *not* a 3-dimensional embedding, even though the regular use of 3D plots may seem to suggest otherwise.

See the :doc:`Value of Animation <value_of_animation>` page for a simple example illustrating this important idea.

Visualization Tools
^^^^^^^^^^^^^^^^^^^

DQM's main visualization tool is the :func:`plot_frames <dqm.utils.plot_frames>` function, which uses the `Plotly <https://plotly.com/python/>`_ package.

DQM also has a 'backup' function, :func:`plot_frames_ipv <dqm.utils.plot_frames_ipv>`, which uses the `IPyVolume <https://ipyvolume.readthedocs.io>`_ package. (IPyVolume is less stable/mature than Plotly and may be buggy. However, it may handle large numbers of data points and/or frames much better than Plotly does.)

If you prefer to use your own visualization tools, you can. The core functionality of DQM itself does not rely on the plotting functions above. The final result of a DQM evolution (stored in the ``dqm_obj.frames`` instance variable) is a 3D array with shape: ``<number of points x number of dimensions x number of frames>``. From there, you can visualize (and analyze) however you wish.

Basic Workflow
--------------

The starting point for DQM is always a 2D real-valued matrix, with data points (samples) in the rows and dimensions (features) in the columns.

Raw data is stored in a DQM instance in ``dqm_obj.raw_data``.

Data Preprocessing
^^^^^^^^^^^^^^^^^^

**Domain-Specific and Data-Specific Preprocessing**

There are any number of steps you may take to clean, process, and transform your data before exploring it with DQM.

One of many possible examples: when working with biological sequencing data (e.g., RNA-seq), a log2 transform is typically applied to the data before any further analysis.

You may also want to exclude extreme outliers from your data. (See the `Excluding Outliers`_ section below.)

**Giving Dimensions Equal Weight**

For DQM, every data dimension is simply a dimension, like any other, in a Euclidean space. If some of the dimensions in your data have far greater variance than others, the high-variance dimensions will dominate the structure that you see in DQM. That may be what you want.

If it's not what you want, consider normalizing the variance of each dimension in the data, in order to give all dimensions equal 'weight' in DQM. This can be done with something as simple as a z-score of each dimension (subtracting the mean and dividing by the standard deviation).

PCA Transformation
^^^^^^^^^^^^^^^^^^

(*See the* `Wikipedia PCA page <https://en.wikipedia.org/wiki/Principal_component_analysis>`_ *for background on Principal Component Analysis.*)

First, note that DQM itself works in any Euclidean coordinate system, with any number of dimensions; using PCA is *not* intrinsic to DQM.

**Whether to Use PCA**

For DQM, PCA is essentially a rotation to a new coordinate system, where the 1st PCA dimension has the greatest variance in the data, the 2nd PCA dimension has the next greatest variance in the data, etc.

Using a PCA transformation as part of your DQM workflow is almost always a good idea, for two reasons:

First, PCA is useful for 'gentle' dimensionality reduction. A typical PCA analysis will only look at the first 2 (maybe 3) PCA dimensions; with DQM, however, dozens of PCA dimensions are often used, or even hundreds. **Using hundreds of PCA dimensions may still count as important dimensionality reduction if you're working with very high-dimensional data.**

Second, visualization of the first 3 dimensions of the PCA coordinate system allows us to pack as much information as possible into a single 3D plot. (And, as mentioned above, animating this 3D plot then presents information from the higher dimensions as well.) **For this reason, even though PCA is typically used as a dimensionality-reduction technique, it can and typically should be used with DQM even if you do no dimensionality reduction at all.**

Of course, you can also visualize higher PCA dimensions, not just the first 3. This can be interesting, but observing DQM evolution in the first 3 PCA dimensions is usually good enough.

**How Many PCA Dimensions To Use: The 'Spike' Model and Elbows**

If a data cloud can be seen to have an 'effective dimensionality' that is lower than the total number of dimensions, projecting into a smaller number of PCA dimensions can be an important source of noise reduction (while also reducing computation time and memory usage).

A theory known as the 'spike model' essentially posits that a horizontal plateau in a plot of decreasing PCA eigenvalues represents a floor of noise in the data. This suggests a method for choosing a number of PCA dimensions to work with that will maximize information and minimize noise: namely, by choosing a number of dimensions at the 'elbow' of a PCA eigenvalue plot. (See the `Quick Start example <quick_start.html#run-pca>`_, where the elbow suggests that 4 PCA dimensions is enough to capture the most important structure in the data.)

If you find that the elbow is farther out than your computing resources will allow, it's a good idea to simply use as many dimensions as you can, to maximize the amount of information that you're working with. (*Computational complexity and memory usage for DQM are both essentially linear*, :math:`O(n)`, *in the number of dimensions being used.*) It's helpful that the ordering of PCA dimensions is based only on variance in the entire data cloud, and is otherwise 'unbiased' (as far as any relationships with metadata or types of structures that may be revealed.)

**Working with PCA in the DQM Class**

The following code block (following the `Quick Start example <quick_start.html#run-pca>`_) demonstrates choosing a number of PCA dimensions to work with:

.. code-block::

    dqm_obj.verbose = True  # default True

    # run PCA, store results in instance, and display plots with PCA info
    dqm_obj.run_pca()

    # choose an explicit number of dimensions (takes precedence if not None)
    # dqm_obj.pca_num_dims = 18
    # OR...
    # choose a minimum proportion of total cumulative variance for the PCA dimensions to be used
    dqm_obj.pca_var_threshold = 0.98

    dqm_obj.pca_transform = True  # default True (if False, frame 0 will be a copy of the raw data)
    dqm_obj.create_frame_0()

    print("In the DQM instance, 'frames' (which now stores frame 0) has shape:", dqm_obj.frames.shape)

Creating Frame 0
^^^^^^^^^^^^^^^^

The :meth:`create_frame_0 <dqm.DQM.create_frame_0>` method creates the first 'frame' of the evolution and stores it in ``dqm_obj.frames``. The following code:

.. code-block::

    dqm_obj.create_frame_0()
    print(dqm_obj.frames.shape)

... will print the shape of 'frames', which will be ``<number of points x number of dimensions x 1>``. Note that 'frames' is 3D; more frames will be added in the 3rd dimension during DQM evolution.

If you're using a PCA transformation, the number of dimensions will be determined by the instance's PCA-transformation settings (see above).

If you're not using a PCA transformation, frame 0 will simply be a copy of the raw data (stored in ``dqm_obj.raw_data``).

.. _Excluding Outliers:

**Excluding Outliers**

If you haven't dealt with outliers already, now is a good time to check for them, in a visualization of frame 0 (by calling ``plot_frames(dqm_obj.frames)``).

Any extreme outliers in your data will cause the DQM map to become a relatively uninteresting illustration of just how different the outliers are from everything else. Thus, you may want to simply exclude them from the data set.

Choosing a Basis
^^^^^^^^^^^^^^^^

The 'basis' in DQM is a subset of data points that we choose from the data set. These basis points will be used to represent all other data points and will form the core of all DQM calculations. (*The word 'basis' here is referencing the idea from linear algebra; see the technical summary* `Understanding DQM <https://github.com/zanderteller/dqm/blob/main/docs/Understanding%20DQM.pdf>`_ *for the technical details.*)

The size of the basis (i.e., the number of basis points) sets a 'resolution' for how much detail we can see in the landscape. A large basis is very computationally expensive (building frames is approximately :math:`O(n^3)`), so in order to use DQM efficiently it's a very good idea to follow these guidelines:

* Start with a smaller basis as you begin exploring a data set.
* Increase the basis size later when you need greater resolution.

For the typical computing power available in today's computers, here is a (very approximate) way to think about basis size:

* Small: up to 500 points
* Medium: 500 to 1,000 points
* Large: 1,000 or more points

The following code will choose a basis of size 100:

.. code-block::

    dqm_obj.basis_size = 100
    dqm_obj.choose_basis_by_distance()
	
Choosing the basis by distance means that the method is choosing the basis points to be as far away from each other as possible in the data space. (See :meth:`choose_basis_by_distance <dqm.DQM.choose_basis_by_distance>` for details.)

**Basis Overlap**

For any non-basis point, the 'overlap' of that point in the basis is a measure of how well the basis describes that point. For points far away from any basis point, the overlap will be small, which tells us that the chosen basis will not do a good job in modeling the behavior of that particular point.

Overlap for a given data point is always between 0 and 1, with 1 being a perfect representation of the point by the basis. (All basis points have overlap of 1 in the basis.)

By default, the :meth:`build_overlaps <dqm.DQM.build_overlaps>` method builds and returns basis overlaps for all non-basis rows.

*For full technical details on basis overlaps, see the section on "Reconstruction of Wave Functions in the Eigenbasis" in the technical summary* `Understanding DQM <https://github.com/zanderteller/dqm/blob/main/docs/Understanding%20DQM.pdf>`_.

**Low-Overlap Points and Smoothness of Evolution**

How low is too low for basis-overlap values? This question does not have a clear-cut answer, and the 'right' answer may be context-dependent.

However, there is a practical heuristic. If any points 'jump' or 'snap' to a new location at the beginning of the evolution, this is a sign that the jumping points are not well represented in the basis. This problem can be fixed by either:

* increasing the basis size, or
* increasing the value of sigma (see below), or
* treating the badly represented point as an outlier and excluding it

A second heuristic is expressed in the current default values of ``dqm_obj.overlap_min_threshold`` and ``dqm_obj.overlap_mean_threshold`` -- see the section below `Choosing a Minimum Good Value of Sigma`_.

Choosing DQM Parameters
^^^^^^^^^^^^^^^^^^^^^^^

**Sigma**

Sigma (:math:`\sigma`), introduced and explained here, is DQM's main tunable parameter (stored in ``dqm_obj.sigma``).

When DQM builds a data-density map, the first step is to place a multidimensional Gaussian distribution around each data point. Sigma is the width of each Gaussian. There is only a single value for sigma; whatever value is chosen, every Gaussian around every data point has that same width (in every dimension).

The starting point for the overall DQM landscape is then simply all the Gaussians added together.

For any data set, the extremes are always the same:

* for very small sigma, each point has its own 'well' in the landscape, and nothing will move -- there will be no evolution at all.
* for very large sigma, all points will be within a single giant well and will immediately collapse together during evolution.

The values of sigma in between these extremes are where we can learn interesting things about the structure of the data set.

Importantly, note that 'small' and 'large' values of sigma are relative to the overall scale of the data set. (The :meth:`estimate_mean_row_distance <dqm.DQM.estimate_mean_row_distance>` method is a useful starting point for interesting, 'well scaled' values of sigma.)

.. _Choosing a Minimum Good Value of Sigma:

**Choosing a "Minimum Good Value" of Sigma**

The ability of a set of basis points to describe non-basis points depends on sigma. For a fixed set of basis points and non-basis points, the basis will describe the non-basis points more and more accurately as sigma get bigger. This gives us a way to find a "minimum good value" of sigma that will adequately model the non-basis points in the data set.

As shown in the code block below, the :meth:`choose_sigma_for_basis <dqm.DQM.choose_sigma_for_basis>` method searches for the smallest value of sigma that satisfies the thresholds for minimum and mean overlap values for non-basis points:

.. code-block::

    dqm_obj.overlap_min_threshold = 0.5  # default 0.5
    dqm_obj.overlap_mean_threshold = 0.9  # default 0.9

    dqm_obj.choose_sigma_for_basis()

    print('The DQM instance now has a stored value of sigma:', dqm_obj.sigma)

Note that this method won't work if you're using a 'full' basis (i.e., all data points are in the basis) -- there need to be some non-basis points to work with.

**Mass**

The DQM mass parameter (stored in ``dqm_obj.mass``) controls the 'transparency' of the DQM landscape for a data point during evolution:

* For a very large mass, a point will get stuck in every local minimum.
* For a very small mass, a point will pass through every barrier and shoot straight toward the global miminum.

Mass is typically set automatically, by a heuristic designed to make the landscape transparent to density variations in uniform random data -- that is, the mass should be just small enough that density variations at that scale are ignored and passed through. (See the :meth:`default_mass_for_num_dims <dqm.DQM.default_mass_for_num_dims>` method for details.)

The value of mass can be adjusted manually, but it's best to leave this as an 'advanced' technique.

.. note::

   However many dimensions are being used by DQM, it's always possible that the effective dimensionality of the data cloud could be significantly lower. The current heuristic described above makes no attempt to deal with this issue. DQM has room for improvement here.

.. warning::

   Using a value of mass that is too small can cause oscillatory behavior -- data points can oscillate around a minimum, because they are overshooting the minimum in each step of the evolution. In this scenario, data points may never stop moving. (The :meth:`build_frames_auto <dqm.DQM.build_frames_auto>` method has a ``max_num_frames`` parameter as a backstop for this problem.)

**Step**

The DQM 'step' parameter (stored in ``dqm_obj.step``) sets the time step between frames of the evolution. It has a default value of 0.1. (The 'units' of time here are arbitrary and unimportant.)

This parameter essentially never needs to be changed.

*Here's one case where you could be tempted to try, though: if you have an evolution where things are moving very smoothly and very slowly, increasing the time step slightly might be the easiest way to speed up the computation without losing (much) resolution in understanding the structure of the data. Don't say you weren't warned, though. Caveat emptor.*

Building Operators
^^^^^^^^^^^^^^^^^^

A quick recap -- once you've:

* Done any preprocessing of your data
* Chosen whether to use a PCA transformation, and how many PCA dimensions to use (DQM uses all PCA dimensions by default)
* Chosen a basis (DQM uses a 'full' basis, using all data points, by default)
* Chosen a value of sigma

... then you're ready to build the DQM operators, which will be used during evolution.

This step itself is extremely simple, using the :meth:`build_operators <dqm.DQM.build_operators>` method:

.. code-block::

    dqm_obj.build_operators()

That's it. The operators are now stored in the instance, and you'll never need to work with them directly. (*Note: this step can be slow for large data sets, especially when using a large basis.*)

**Changing the Operators**

The operators depend on all of the following:

* the raw data
* the choice of basis
* the DQM parameters: sigma, mass, and step

If you change any of those things, you'll need to rebuild the operators.

If the instance already has multiple frames, :meth:`build_operators <dqm.DQM.build_operators>` will raise an error. This is a safety precaution, to make it harder to allow the instance to wind up in an inconsistent state.

You can use the :meth:`clear_frames <dqm.DQM.clear_frames>` method to clear frames (keeping frame 0 by default).

.. warning::

   The onus is currently on the user to make sure that a DQM instance doesn't wind up in an inconsistent state, with mismatches between the stored values for the basis, parameters (sigma, mass, step), operators, and frames. There are a reasonable number of error checks in the code, but it's a complicated system. (DQM undoubtedly has room for improvement here.)

**The Underlying Mathematics for the Operators**

Here, we'll give an extremely brief description of each operator:

``dqm_obj.simt`` is the transpose of the 'similarity' matrix, which is used to convert each data point's current state vector from the 'raw' basis (of basis points) to the eigenbasis (of quantum eigenstates).

``dqm_obj.exph`` is the complex-valued 'evolution' operator matrix (that is, the exponentiated Hamiltonian time-evolution operator matrix). It converts a data point's current eigenbasis state vector at time :math:`t` into a new 'evolved' eigenbasis state vector at time :math:`t + step`.

``dqm_obj.xops`` is a 3D tensor of position-expectation operators. Each slice :math:`i` in the 3rd dimension is the operator matrix that converts the eigenbasis state vector for a data point into the expected position of the data point in the :math:`ith` dimension of the data space.

If you want the full mathematical details, see the section on "Building the Quantum Operators" in the technical summary `Understanding DQM <https://github.com/zanderteller/dqm/blob/main/docs/Understanding%20DQM.pdf>`_.

Building Frames
^^^^^^^^^^^^^^^

We're now ready to proceed with the DQM evolution.

The :meth:`build_frames <dqm.DQM.build_frames>` method will build a specified number of frames (100 by default):

.. code-block::

    # build and add 50 new frames to the 'frames' instance variable
    dqm_obj.build_frames(50)  # default 100

The :meth:`build_frames_auto <dqm.DQM.build_frames_auto>` method will call :meth:`build_frames <dqm.DQM.build_frames>` repeatedly (in batches of 100 by default) until all points have stopped moving:

.. code-block::

    # build and add new frames, in batches of 50, until all points stop moving
    dqm_obj.build_frames_auto(50)  # default batch size 100

:meth:`build_frames_auto <dqm.DQM.build_frames_auto>` uses the ``dqm_obj.stopping_threshold`` parameter to decide when a point has stopped moving. A point is considered to have stopped if it moves less then ``stopping_threshold`` distance from one frame to the next. ``stopping_threshold`` is set automatically to ``dqm_obj.mean_row_distance / 1e6`` but can be adjusted manually.

For large data sets and large basis sizes, building frames can be quite slow. In these cases, it's a very good idea to build a small number of frames first, to begin to understand what the landscape looks like, before committing to building hundreds or even thousands of frames.

The run_simple Method of the DQM class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :meth:`run_simple <dqm.DQM.run_simple>` method conveniently wraps all the steps we've seen so far into a single call -- here's exactly what the method is actually doing:

.. code-block::

    def run_simple(self, dat_raw, sigma):
        self.raw_data = dat_raw
        self.sigma = sigma

        self.create_frame_0()
        self.build_operators()
        self.build_frames_auto()
    # end method run_simple

Calling the method can be this simple:

.. code-block::

    dqm_obj = DQM()
    dqm_obj.run_simple(dat_raw, sigma)

Be aware of DQM's default behaviors (unless you change settings in the instance before you call the method):

* It does a PCA transformation and keeps all PCA dimensions.
* It uses a 'full' basis (all data points are in the basis).

Especially for small data sets, doing multiple simple runs with various values of sigma can be the quickest way to understand the landscape that DQM is revealing.

Saving and Loading DQM instances
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For evolutions that take a long time to build, saving the results of your work can be important. For this purpose, the :class:`DQM <dqm.DQM>` class has these methods:

* :meth:`exists <dqm.DQM.exists>` (class method)
* :meth:`load <dqm.DQM.load>` (class method)
* :meth:`save <dqm.DQM.save>` (instance method)

Each method takes a path to a folder and an optional name of a subfolder.

The main folder stores information that can be common to multiple DQM landscapes (raw data, PCA results).

The subfolder stores landscape-specific information (basis, DQM parameters, operators, frames).

This setup allows you to group multiple results that share the same raw data. (It's up to you to name the subfolders in a way that keeps things organized and decipherable.)

*For large data sets, basis sizes, and numbers of frames, keep in mind that the files on disk can become quite large.*

Interpreting and Using Results
------------------------------

DQM evolutions, or 'maps', are a rich source of nuanced information about the structure inherent in any data set. Interpreting and using results from DQM maps is, accordingly, a multifaceted issue, with plenty of room for exploration and development by the user. DQM is desigend and intended for open-ended exploration, and best results will often be achieved when you approach with an open mind. Learning answers to questions you didn't know you had can be a valuable source of insights and new directions.

DQM has two main tools for interpretation: application of metadata by color, and the :func:`get_clusters <dqm.utils.get_clusters>` utility function. It's easy to imagine other, more sophisticated tools as well; a few are hinted at below, and some will probably make their way into DQM over time. For now, though, it's likely that finding interesting results in your DQM analyses will involve some tool-building on your part.

Application of Metadata
^^^^^^^^^^^^^^^^^^^^^^^

In line with the importance of visualization in the DQM process, metadata is best applied to a data set by coloring of data points. There is potential for plenty of nuance here: the relationship(s) between data and metadata may be simple or complex, and may manifest in all or only in parts of the data set.

In the :doc:`quick_start` guide, coloring the 4 clusters provides a clear (though artificial) example of coloring by metadata. The color syntax demonstrated there is entirely flexible, meaning it can be used to apply continuous metadata as a color map as well. (*Adding wiring to the* :func:`plot_frames <dqm.utils.plot_frames>` *function to make use of Plotly's built-in color maps is an obvious opportunity for improvement.*)

Sets with No Interesting Structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Some data sets will have no 'interesting' structure; the entire set may be a simple spherical cloud, with points arriving at the cluster center from all directions during DQM evolution.

**This result may often seem disappointing, but it's important to remember that a finding of no interesting structure is itself valuable information.** Most conventional modeling algorithms (clustering, regression, and classification) will happily report whatever structure you ask for, whether or not said structure actually exists in the data set.

When this happens, there are a few obvious conclusions to consider:

* You may need a better way to choose the interesting features (dimensions) in your data
* You may need better preprocessing of your data
* You may need better data

**Order of Arrival**

Before despairing, though... The dynamic aspect of DQM can sometimes provide value even in the 'uninteresting' case -- order of arrival at the cluster center can itself contain information. In a very simple hypothetical example: healthy samples may consistently arrive earlier (meaning they're closer to the center of the cloud), with sick samples consistently arriving later. This can be readily apparent in visualization of the evolution.

Clusters
^^^^^^^^

Multiple clusters that have separated during DQM evolution become very easy to tell apart.

DQM's primary tool for numerical separation of clusters is the :func:`get_clusters <dqm.utils.get_clusters>` utility function. You can also use any other conventional clustering algorithm, or even just separate by area of space (by setting thresholds in one or several data dimensions).

Note that different clusters, and different numbers of clusters, can be extracted from different frames within a given DQM evolution; see the Quick Start guide's section on `Using get_clusters <quick_start.html#using-get-clusters>`_ for a clear example.

1-D Extended Structures as Subclusters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1-dimensional extended structures are a regular occurrence in DQM maps -- acting as streambeds in a landscape, with points flowing along the structure to arrive at a final location.

When multiple 1-D structures flow into the same final location from different directions, these structures can be meaningfully treated as subclusters of the main cluster.

These subclusters can be separated by numerical methods (including, as in the `Quick Start <quick_start.html#using-get-clusters>`_ guide, by using :func:`get_clusters <dqm.utils.get_clusters>` on an intermediate frame). In some cases, though, it may be easier to separate them by isolating the main cluster and then building a new DQM map to separate the subclusters. (The Quick Start guide's section on `using run_sumple <quick_start.html#using-run-simple>`_ demonstrates this technique as well.)

You may even see branches in these 1-D structures, like multiple tributaries feeding into a larger river. The relative importance of these sub-subclusters will often be context-dependent (possibly depending on relationships with metadata).

1-D Extended Structures as Regressions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Most generally, a regression is a mathematical relationship between a dependent variable and some number of independent variables.

In DQM, if some continuous metadata variable is seen to vary consistently along a 1-dimensional extended structure, this is clearly evidence of a regression in the above sense.

Unlike conventional regression algorithms, DQM does not provide you with a mathematical formula describing the revealed relationship between the metadata and the data dimensions. On the other hand, DQM makes no assumptions of any kind about the shape underlying the relationship. In fact, you don't even have to know beforehand whether you're going to see a regression relationship or not.

Also, a DQM map can itself be used as a model, bypassing the need for a mathematical formula describing the relationship. (See the section below on :ref:`running new points <Running New Points>`.)

Higher-Dimensional Extended Structures
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

DQM has been seen to reveal 2-dimensional manifolds in real data, and there are no theoretical barriers to seeing even higher-dimensional manifolds as well (though, it would seem, these may be rare).

Interpretation and analysis of these higher-dimensional manifolds may be valuable but will be intrinsically more complex.

One approach to exploring the effective dimensionality of a particular structure is to isolate that structure (using, e.g., :func:`get_clusters <dqm.utils.get_clusters>`) and then re-run PCA, typically on an intermediate frame of the evolution, just for the points in the structure in question.

The utility function :func:`rescale_frames <dqm.utils.rescale_frames>` can also be useful here; it effectively 'zooms in' on a structure that is shrinking as the DQM evolution unfolds, making it much easier to see the nature of the structure later in the evolution. Subselecting data points to see only the structure in question (with no outliers) is important in order for this tool to be useful.

Area-of-Space Relationships
^^^^^^^^^^^^^^^^^^^^^^^^^^^

There may be cases where you notice clear differences between metadata values in different areas of the data space, without useful structures forming in the DQM evolution. This observation can lead back in the direction of applying a traditional classifer to your data.

Outliers
^^^^^^^^

'Outliers' in DQM are points that never move -- or perhaps move just enough to join very small 'outlier clusters'. A point being an outlier is a relative concept in DQM -- increasing sigma can pull outliers into larger structures (which is sometimes the main motivation for increasing sigma).

Outliers should not necessarily just be ignored -- as with the 'order of arrival' observation above, outliers may themselves have a meaningful relationship with the metadata.

How Many Models for Your Data?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

One of the most valuable aspects of DQM analysis can be determining how many models you actually need in order to accurately describe your data set.

To illustrate the idea, consider a simple hypothetical example: suppose you see three clusters in your data set. Two of those clusters are seemingly spherical 'point' clusters, with points arriving at the cluster centers from all directions. The third cluster, however, shows a clear 1-dimensional extended structure, possibly with an interesting relationship to some metadata value. Knowing that a more conventional regression algorithm may be usefully applied, but only to a particular subset of your data, is a vitally important insight.

The **Demo: Real Data #1** Jupyter notebook (in ``notebooks/demo_real_data_1.ipynb`` in the DQM repository) has a good example of this issue.

Feature Selection
^^^^^^^^^^^^^^^^^

Feature selection -- the process of identifying which features (i.e., dimensions) in your data are the important ones -- is an important aspect of data analysis.

**General Feature Selection**

Particularly when using a PCA transformation, you can look at the weights in the first few PCA dimensions. (PCA dimension weights are stored in the columns of the ``dqm_obj.pca_eigvecs`` matrix.) Is there a small number of 'raw' dimensions with much larger PCA weights than all other 'raw' dimensions? If so, those raw dimensions are presumably disproportionately responsible for whatever structure you're seeing in DQM.

You can test that theory: for the given subset of features, if you build a DQM map with just those features, do you see essentially the same structure that you saw in the 'full' map using all features? If so, this is a decent indication that your subset of features contains all of the important information leading to the structure that you're seeing.

**Feature Selection for DQM Clusters**

Simple differential-expression calculations can be applied between clusters to see which features show the strongest differentiation.

**Feature Selection for DQM Extended Structures**

Given an ordering of points along a 1-D structure (paused/frozen at some frame of a DQM evolution), which features are more or less highly correlated with the ordering of points along the structure?

These correlations are clearly connected to the direction along which the 1-D structure extends in the data space. Of course, if the 1-D structure is nowhere close to straight, such correlations will be weak; this is a sign that the structure relies on all (or at least many) of your features, and it's likely to be difficult to retain the structure when subselecting to a smaller feature set.

**DQM Mapping of Features**

By simply transposing your raw-data matrix, you can proceed to build a DQM map where the points on the map are now your original features (dimensions), and the dimensions of the data space are now your samples.

This approach can be complex and nuanced, and may provide insights well outside of what other feature-selection methods even consider.

Note that normalization of your features (the rows in your transposed raw data) is crucially important here. (As a starting point, be aware that L2 normalization is highly preferable to L1 normalization, which can create intriguing but essentially meaningless 'spikes' in a DQM feature map.)

Running New Points
------------------

Any given DQM map can actually be used as a model, in the sense that new 'out-of-sample' points can be evolved in that map, and the points' behavior in the map can lead to conclusions and predictions about the new points.

A DQM map can be used for:

* classification -- based on which cluster (if any) each new point joins
* regression -- based on where along some extended structure (if at all) each new point arrives (at some predetermined 'moment' -- i.e., frame -- in the evolution)

When using an existing map as a model, note that the DQM map is *not* updated to include the effect of the new points on the landscape. The map itself is entirely 'in-sample', based only on the original data.

The process of running new points should be as follows:

* Apply any data preprocessing to new points. For this to make sense, preprocessing of new points needs to be *exactly* the same as the preprocessing of the original data.
* Call the :meth:`run_new_points <dqm.DQM.run_new_points>` method, where each input row is a preprocessed new point.

**Rule of thumb: if you can't run new points one at a time, you must be cheating somehow.** In other words: if you're using any aggregate statistics about your new points, then you're not fully treating them as 'out-of-sample'.

The outputs of :meth:`run_new_points <dqm.DQM.run_new_points>` are:

* a set of frames for the new points (evolved to as many frames as currently exist in ``dqm_obj.frames``)
* a vector of in-sample basis overlaps (for all original non-basis points)
* a vector of out-of-sample basis overlaps (for all new points)
* a vector of in-sample proportional norms (see below)
* a vector of out-of-sample proportional norms (see below)

Out-of-Distribution Issues
^^^^^^^^^^^^^^^^^^^^^^^^^^

Most conventional modeling algorithms will happily model a new point even if that new point is completely outside of the distribution of data used to build the model. This behavior is clearly problematic.

DQM provides a way -- two ways, actually -- to address this out-of-distribution issue.

.. note::

   In both situations below (for both norms and overlaps), thresholds for what values qualify as 'too low' are not well defined, may be context-dependent, and are subjects for further study.

**'Off the Map'**

*The following only applies when using a PCA transformation.*

The proportional-norm vectors mentioned above (both in-sample and out-of-sample) present a 'norm' for each point that is actually the ratio of norm 1 / norm2:

* norm 1: the PCA-transformed (centered, rotated, truncated) L2 norm for the point
* norm 2: the original (centered) L2 norm for the point

A 'perfect' norm has a value of 1 (i.e., no loss of information for the given point).

Any out-of-sample norms that are significantly below the distribution of in-sample norms should be considered to be 'off the map' -- that is, too much information about the new point has been lost in the PCA transformation (more so than for most/all in-sample points).

If the in-sample distribution of norms is itself too low, that may prompt you to reconsider the value of the map you're working with. (Of course, 'too low' here is relative to how much 'loss of information' you believe is either helpful noise reduction or an acceptable cost of dimensionality reduction.)

**'Holes in the Map'** 

*The following applies whether a PCA transformation is used or not.*

Any new points with basis overlaps well below the distribution of in-sample basis overlaps are not being well represented by the basis.

To distinguish how we talk about the two issues: here, rather than being 'off the map', we can think of these low-overlap points as existing in 'holes' or 'empty/blank spots' in the map.

As mentioned in the section on :ref:`Choosing a Basis` above, low-overlap points can 'jump ' or 'snap' closer to nearby basis points at the beginning of evolution. Visualization of the evolution for such a point can be misleading, and it may be better to exlude them from visualization entirely.

Again, a distribution of in-sample basis overlaps that is itself too low should be cause for reconsideration of the quality of the map itself.

Additional Topics
-----------------

Working with Large Data Sets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Large Numbers of Dimensions**

DQM has been successfully used for very large numbers of dimensions (up to ~500,000), with good results.

Keep in mind that a PCA transformation will give a total number of dimensions that is the *minimum* of the number of raw dimensions and the number of data points. For a data set with 1,000 samples and 500,000 dimensions, PCA only needs 1,000 dimensions to fully describe the samples.

If you're dealing with very large numbers of both dimensions *and* samples, the PCA calculations will be... challenging. DQM in its current form does not provide a solution to this problem.

**Large Numbers of Data Points**

DQM has been used successfully on data sets with millions of data points.

Depending particularly on the size of the basis you're using, processing millions of points can consume a whole lot of computing resources. It's a good idea to run timing tests to give yourself an estimate of how long running all points will take.

There are also other strategies that can help you learn about the structure of your data more efficiently than waiting for millions of points to evolve for hundreds or thousands of frames.

Here's one example of a strategic starting point: choose two random subsets of points from your data -- say, 10,000 points each. Build a separate DQM map for each subset. Are you seeing the same structure in both maps or not? If not, work your way up to a sample size that starts to give you a clear picture of what the structure of the entire data set looks like. As a further test at each sample size, you can run some or all of the points from each subset as new, out-of-sample points in the map built with the *other* subset, to get an even more specific sense of how similar or different the two maps are from each other. (Also: in this example, as always, it's efficient to start with relatively small basis sizes and work your way up until you're getting the resolution that you need.)

**Computational-Complexity Notes**

Different parts of the DQM workflow have different computational complexities, but these are general facts to keep in mind:

* **number of DQM dimensions**: complexity is essentially linear, :math:`O(n)`.
* **number of data points**: complexity is essentially linear, :math:`O(n)`. Choosing the basis is the exception: if you want to start from the greatest outlier, complexity there is quadratic, :math:`O(n^2)`.
* **basis size**: the big cost is building frames, where the complexity is approximately cubic, :math:`O(n^3)`. A larger basis gets more expensive very quickly -- so, again, it's best to start with relatively small basis sizes and work your way up to the resolution that you need.

**Memory-Usage Notes**

There are two big considerations for memory usage (in memory and on disk):

The position operators (stored in ``dqm_obj.xops``) are ``<basis size x basis size x number of DQM data dimensions>``. For a basis size of 1,000 and 100 DQM data dimensions, that comes out to 0.8 GB.

The frames are the big one -- they're ``<number of points x number of DQM data dimensions x number of frames>``. For, say, 10,000 points, 100 DQM data dimensions, and 1,000 frames, that comes out to 8 GB. (If you're dealing with millions of data points - well, you do the math...)

**Parallel Processing**

Be warned: DQM will eat up all the CPU resources it can get its hands on. (The compiled C++ code uses the OpenMP library for parallel processing.) Particularly when building frames during evolution, you may see all of your CPUs working at full capacity.

As far as parallel processing across multiple machines is concerned, DQM has that potential but is not currently set up for it. Here are the changes that could be made:

* In building the operators, there's a function in the compiled code (AggregatePotentialContributions) where a map/reduce operation across all data points could be easily applied.
* In building frames, the evolution of each data point is entirely independent of all other data points. So, the evolution of batches of data points could easily be farmed out to multiple machines.

Working with Other Data Types
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

DQM is inherently designed and built to work within a Euclidean data space of continuous, real-valued dimensions.

However, there are various techniques for converting other data types into a Euclidean data space, so that a DQM analysis might be usefully performed. Below are two examples.

**Categorical Data**

Consider a categorical data dimension -- say, hospital name, with 5 different possible values. There is no ordering to the possible values in this dimension.

A simple solution is to replace the dimension with 5 new binary dimensions, each containing a simple 0/1 (yes/no) for each possible hospital name. It's clear how to assign coordinates to a given sample, and every point in these 5 new dimensions is equidistant from every other point, preserving the desired lack of ordering. (By design, a given sample should always have exactly one value of 1 in these 5 dimensions.)

**Graph/Network Data**

For an undirected graph (the situation for a directed graph is harder), a popular metric of distance from one vertex to another is the commute time: that is, the expected time for a random walk from vertex 1 to arrive at vertex 2, plus the expected time for a random walk to go back from vertex 2 to vertex 1. (This definition makes the commute time symmetric, necessary for a distance metric.)

These commute-time distances allow you to construct a Euclidean distance matrix, which is just the symmetric matrix of pairwise distances between vertices. From there, you can construct a set of Euclidean coordinates for each vertex that satisfies all distances in the distance matrix.

Multimodal Analysis
^^^^^^^^^^^^^^^^^^^

It's possible to look for interactions between different data types by combining them into a single DQM analysis.

Consider an example involving hospital patients, where every patient has data in 2 different data sets:

* an EHR (electronic health record) data set, with 100 dimensions (blood pressure, heart rate, etc.)
* a blood-sample RNA-seq data set, with 10,000 dimensions (with expression levels for 10,000 different genes)

You may choose to simply concatenate the data into a single 10,100-dimensional data set, and then run a DQM analysis.

There are two important considerations here:

#. Make sure that the overall variance is not extremely different between the two data sets. Otherwise, the set with larger variance will dominate, and the set with smaller variance will have little or no impact on the DQM landscape. As with individual data sets, you can choose to simply normalize the variance of each dimension within each set.
#. Keep in mind that, in addition to the issue of overall variance, a great difference between the *numbers* of dimensions will also affect which data set predominates in determining structure in DQM. In our current example, one set has 10,000 'votes' while the other set only has 100 'votes' as far as what the DQM landscape will look like. An extreme imbalance here may render this kind of multimodal analysis unhelpful. (*You might be tempted to try to counteract this effect by increasing the relative scale of the data set with fewer dimensions, but this begins to raise tricky questions -- notably, how do you know if you've achieved 'balance between the sets successfully...?*)

The Curse of Dimensionality
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Any very high-dimensional space has an extremely large number of 'corners', and so it's extremely easy for a data set to fail to cover the entire space, even for a very large number of data points. DQM is not in any way immune to this problem.

However, there's the separate question of whether a given data set is, in fact, covering the range of possible combinations of values, in all dimensions, that you're ever likely to see. If so, that implies that all real data in the given domain lives within some lower-dimensional manifold of the high-dimensional space (which is entirely possible).

The crucial question is whether this issue impairs the functioning of DQM in high dimension. The short answer is, 'no'.

DQM is concerned with variations in data density in the space -- in other words, patterns in the relative distances of data points from each other.

If every data point in a high-dimensional space is off in its own unique corner of the space, with every point thus more or less equidistant from every other point, then DQM will see that, in the form of a lack of interesting structure in the data set. (*On an important related note: the heuristic in the* :meth:`default_mass_for_num_dims <dqm.DQM.default_mass_for_num_dims>` *method is designed to make mass just small enough that DQM will ignore -- that is, not treat as interesting structure -- typical density variations in uniform random data. The scale of those variations goes up with the number of dimensions, and thus so does the default mass.*)

If however, we are in the situation where all possible observations lie in some lower-dimensional manifold, and the data set contains some degree of interesting structure within that manifold, then DQM will reveal that structure.

*As a separate matter, entirely distinct from the curse of dimensionality, we can ask how small a data set needs to be before we risk mistaking random variations for 'structure'. DQM is also not immune to issues of statistical significance.*

Non-Locality
^^^^^^^^^^^^

It's a key feature of DQM that every point in a data set effects the entire landscape for that data set, by virtue of the Gaussian distribution placed around it. (The effect of that point is strongest in the immediate vicinity of the point, of course.) This means that removing a subset of points from a data set can noticeably change the relationships between the points that are left.

A notable example involves relative sample sizes. Consider two metadata categories -- for example, healthy and sick. A set of healthy samples and a set of sick samples may form two clearly separate and distinct clusters. However, this may only be true if the relative sample sizes for the two categories are roughly equal. If, on the other hand, there are far more sick samples than healthy samples, then the healthy samples may appear as a subcluster of the sick samples, or possibly may not be distinguishable at all (if the imbalance is sufficiently extreme).

This is a subtlety to be cautious about; there is a learned intuition about DQM landscapes that informs which aspects of a landscape may change under such circumstances.

|
