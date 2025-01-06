import os
import numpy as np
from copy import copy
from math import floor, log10
from time import time
from .utils import pca
import pickle

from .dqm_pure_python import choose_basis_by_distance_python, build_overlaps_python,\
    make_operators_python, build_frames_python

from . import dqm_lib  # compiled-function library

try:
    from matplotlib import pyplot as plt
    HAVE_PLT = True
except ModuleNotFoundError:
    HAVE_PLT = False


class DQM:
    '''
    This is the main class for using DQM (Dynamic Quantum Mapping).

    The code has a reasonable number of error checks, but it's a complicated system.  The onus is currently on
    the user to make sure that the choice of parameter settings makes sense and that a given instance of the
    class doesn't wind up in an inconsistent state.

    Class variables:

    :cvar min_report_time: Report class-level execution times that are this many seconds or longer. Default 10.

    Instance variables (general):

    :ivar verbose: Boolean: whether to report on various operations. Default True.
    :ivar min_report_time: Report instance-level execution times that are this many seconds or longer. Default 10.
    :ivar raw_data: The raw data (a 2-D matrix). Default None.
    :ivar call_c: Boolean: whether to call compiled (C++) code. Default True if the module can find the compiled
        library, otherwise default False (with a printed warning).

    Instance variables (for PCA):

    :ivar pca_transform: Boolean: whether to do PCA rotation/truncation of the raw data when creating frame 0.
        Default True. (If False, all other PCA settings are ignored.)
    :ivar pca_num_dims: Integer number of PCA dimensions to use. Takes precedence over pca_var_threshold if set.
        Default None.
    :ivar pca_var_threshold: Threshold used for choosing number of PCA dimensions to use, representing required
        proportion of total cumulative variance (e.g., 0.9) in the PCA dimensions used. Ignored if pca_num_dims
        is set. Default None.
    :ivar raw_col_means: Stored column means of raw data. (Needed for PCA rotation/truncation of any new
        'out-of-sample' points.) Default None.
    :ivar pca_eigvals: Stored vector of PCA eigenvalues (in descending order). Default None.
    :ivar pca_eigvecs: Stored matrix of PCA eigenvectors (columns correspond to entries in pca_eigvals).
        Default None.
    :ivar pca_cum_var: Stored vector of proportional cumulative variance for the first n PCA dimensions.
        Default None.

    Instance variables (for the basis):

    :ivar basis_size: Integer number of points to use in the basis. (All rows will be used by default for the basis
        if this value is not set.) Default None.
    :ivar basis_num_chunks: Choose the basis by partitioning all rows into this number of 'chunks'. (Multiple chunks
        offers a useful speedup when working with large numbers of rows.) Default 1.
    :ivar basis_rand_seed: Random seed, used to choose a random starting row for the basis. Ignored if
        basis_start_with_outlier is True. Default 1.
    :ivar basis_start_with_outlier: Boolean: whether to use the single greatest outlier row as the starting
        row for the basis. (Will be slow, O(n^2), for large numbers of points.) Default True.
    :ivar basis_row_nums: Stored list of row numbers for rows in the basis. Default None.
    :ivar non_basis_row_nums: Stored list of row numbers for rows not in the basis. Default None.
    :ivar basis_rows: Stored matrix of basis rows (i.e., data for the basis rows, taken from frame 0).
        Default None.

    Instance variables (for choosing sigma -- see method 'choose_sigma_for_basis'):

    :ivar overlap_min_threshold: Minimum overlap for non-basis rows. Default 0.5.
    :ivar overlap_mean_threshold: Minimum mean overlap for non-basis rows. Default 0.9.
    :ivar mean_row_distance: Stored estimated mean pairwise distance between rows in the data set. (See method
        'estimate_mean_row_distance'). Default None.

    Instance variables (main DQM parameters):

    :ivar sigma: Width (standard deviation) of the multidimensional Gaussian placed around every data point.
        Default None.
    :ivar mass: Value of mass assigned to each data point during DQM evolution. Typically set manually by the
        user less often than sigma (see method 'default_mass_for_num_dims'). Default None.
    :ivar step: Time step used during DQM evolution. Typically set manually by the user rarely or never.
        Default 0.1.

    Instance variables (DQM operators):

    :ivar  simt: Stored transpose of the 'similarity' matrix, used to convert state vectors from the 'raw'
        basis to the orthonormal basis of eigenstates. Default None.
    :ivar xops: Stored 3-D array of position-expectation operator matrices. (Each slice in 3rd dim is the
        operator matrix for the corresponding column/dimension in 2nd dim in 'frames'.)
    :ivar exph: Stored 'evolution' operator matrix. (This is the exponentiated time-evolution Hamiltonian
        operator matrix.) Has complex values. Default None.

    Instance variables (frames):

    :ivar stopping_threshold: A given data point is considered to have 'stopped' when it
        moves less than this distance from one frame to the next frame. Typically is automatically set to
        mean_row_distance / 1e6, but can be set manually. Default None.
    :ivar frames: Stored 3-D array of frames: <num rows x num dims x num frames>. First slice in 3rd dim
        contains the original data (possibly PCA rotated/truncated), stored here before evolution has taken
        place. Default None.
    '''

    '''
    Note on Passing Arrays to C Code
    
    Numpy 'C-CONTIGUOUS' arrays are not actually row-major when they're 3-D -- it's the 3rd dimension that varies
    most quickly, not the 2nd dimension, in the underlying memory. So, in order to give the C code the ordering
    that it expects when passing a 3-D array, we need to put the column dimension in the 3rd dimension, like so:
    <num_frames x num_rows x num_cols>. This way, the 2nd dimension (column) is varying most quickly, which is
    what the C code expects (this is what 'row-major order' means). So now, when allocated memory is treated
    as 1-dimensional by the C code, sequential writes fill up frame 0 first, row by row, then frame 1, etc.,
    as desired. Afterward, we permute the dimensions back again here in the Python code, to make the 3-D
    array again the expected <num_rows x num_cols x num_frames>.
    '''


    ## static class variables
    # note: there is also a min_report_time instance variable, but there are some class-level operations to report on
    min_report_time = 10  # report execution times that are 10 seconds or longer


    def __init__(self):
        '''
        Constructor for the DQM class

        Initialize all member variables -- some have defaults, many default to None. Documentation of instance
        variables is in the class docstring (above).
        '''

        self.verbose = True
        self.min_report_time = 10

        self.raw_data = None

        if dqm_lib is None:
            print("## WARNING: in DQM constructor -- compiled-library code not found, setting 'call_c' to false")
            self.call_c = False
        else:
            self.call_c = True
        # end if/else we have compiled-library code or not

        ## for PCA
        # note: if pca_transform is false, all other PCA settings are ignored
        self.pca_transform = True
        self.pca_num_dims = None
        self.pca_var_threshold = None
        self.raw_col_means = None
        self.pca_eigvals = None
        self.pca_eigvecs = None
        self.pca_cum_var = None

        ## for choosing and storing basis
        self.basis_size = None
        self.basis_num_chunks = 1
        self.basis_rand_seed = 1
        self.basis_start_with_outlier = True
        self.basis_row_nums = None
        self.non_basis_row_nums = None
        self.basis_rows = None

        ## for choosing sigma (see choose_sigma_for_basis)
        self.overlap_min_threshold = 0.5
        self.overlap_mean_threshold = 0.9

        ## main dqm parameters
        self.sigma = None
        self.mass = None
        self.step = 0.1

        self.mean_row_distance = None

        ## dqm operators
        self.simt = None
        self.xops = None
        self.exph = None

        ## frames
        self.stopping_threshold = None
        self.frames = None
    # end __init__ constructor


    def default_mass_for_num_dims(self, num_dims=None):
        '''
        Use a simple heuristic formula (derived from random-data experiments) to return a suggested
        default value of mass for a given number of dimensions:

        mass = -1 + 2 * log10(num_dims)

        We set a minimum default mass of 1, which overrides the heuristic for small numbers of
        dimensions (< 10), to avoid oscillation caused by a 'too transparent' landscape.

        Important note: for any given data set, the effective dimensionality of the data cloud might be
        significantly lower than the number of dimensions being used, which could affect the appropriateness
        of the suggested value of mass. We make no attempt to deal with that issue here.

        :param num_dims: Number of dimensions.  if None, we attempt to infer the number of dimensions
            from self.frames (by size of 2nd dim). Default None.
        :return: Suggested default mass value for the given number of dimensions.
        '''

        '''
        2FIX: consider ways to address the issue mentioned above, where we would suggest a default mass
        based on effective dimensionality of the data cloud, not just on the total number of dimensions
        being used.
        '''

        assert num_dims is not None or self.frames is not None,\
            'must have a number of dimensions (passed in or from self.frames) to determine a suggested default mass'
        if num_dims is None:
            num_dims = self.frames.shape[1]

        mass = -1 + 2 * log10(num_dims)
        # for small numbers of dimensions (< 10), make sure mass is positive and big enough to avoid oscillation
        mass = max(mass, 1)

        return mass
    # end method default_mass_for_num_dims


    def run_pca(self):
        '''
        Run PCA on self.raw_data (which must exist) and store results.

        :return: None
        '''

        t0 = time()

        assert type(self.raw_data) is np.ndarray and self.raw_data.ndim == 2, \
            "raw data must be 2-D ndarray in order to run PCA"

        # store raw-data column means (important for out-of-sample operations on new points)
        self.raw_col_means = np.mean(self.raw_data, axis=0)

        # run PCA
        self.pca_eigvals, self.pca_eigvecs = pca(self.raw_data, self.verbose)
        assert min(self.pca_eigvals) >= 0, 'PCA eigenvalues must all be non-negative'

        # calculate cumulative variance of PCA dimensions (variance of each dimension is proportional to
        # the eigenvalue for that dimension)
        self.pca_cum_var = np.cumsum(self.pca_eigvals)
        self.pca_cum_var /= self.pca_cum_var[-1]

        t1 = time()

        if self.verbose:
            if t1 - t0 >= self.min_report_time:
                print("ran PCA in {} seconds".format(round(t1 - t0)))
            if HAVE_PLT:
                self.plot_pca()
            else:
                print('# WARNING: need the matplotlib.pyplot package to do plots')
    # end method run_pca


    def clear_pca(self):
        '''
        Clear all PCA results (self.raw_col_means, self.pca_eigvals, self.pca_eigvecs and self.pca_cum_var).

        :return: None
        '''

        self.raw_col_means = None
        self.pca_eigvals = None
        self.pca_eigvecs = None
        self.pca_cum_var = None
    # end method clear_pca


    def plot_pca(self, num_dims=None):
        '''
        Display 3 PCA plots:

          * normalized eigenvalues (all divided by first eigenvalue)
          * log10 of normalized eigenvalues
          * proportional cumulative variance of data (all divided by total variance of data)

        Note: an assertion will fail if the Matplotlib PyPlot module is not loaded.

        :param num_dims: Number of PCA dimensions to show in the plots. Default None (meaning show all PCA dims).
        :return: None
        '''

        assert HAVE_PLT, "must have loaded  matplotlib.pyplot as plt to use plot_pca"
        assert type(self.pca_eigvals) is np.ndarray, 'must have PCA eigenvalues to do PCA plots'

        if num_dims is None or num_dims > self.pca_eigvals.size:
            num_dims = self.pca_eigvals.size

        plt.figure(figsize=(20, 5))
        ax1 = plt.subplot(1, 3, 1)
        ax2 = plt.subplot(1, 3, 2)
        ax3 = plt.subplot(1, 3, 3)

        plt.axes(ax1)
        # normalized eigenvalues
        norm_eigvals = self.pca_eigvals[:num_dims] / (self.pca_eigvals[0])
        plt.plot(norm_eigvals, '-bo')
        plt.xlabel('dimension number (zero-based)')
        plt.ylabel('normalized eigenvalue')
        plt.title('PCA: Normalized Eigenvalues')

        plt.axes(ax2)
        max_idx = np.where(norm_eigvals > 0)[0][-1]
        log_norm_eigvals = np.log10(norm_eigvals[:max_idx])
        plt.plot(log_norm_eigvals, '-bo')
        plt.xlabel('dimension number (zero-based)')
        plt.ylabel('log10 of normalized eigenvalue')
        plt.title('PCA: Log10 of Normalized Eigenvalues')

        plt.axes(ax3)
        plt.plot(self.pca_cum_var[:num_dims], '-bo')
        plt.xlabel('dimension number (zero-based)')
        plt.ylabel('proportional cumulative variance')
        plt.title('PCA: Cumulative Variance')

        plt.show()
    # end method plot_pca


    def _choose_num_pca_dims(self):
        '''
        Return a number of PCA dimensions to use when creating the rotated/truncated frame-0 matrix.

        Logic:
        * If self.pca_num_dims and self.pca_var_threshold are both None, use all PCA dimensions.
        * Otherwise use self.pca_num_dims if set to a positive value.
        * Otherwise use self.pca_var_threshold.

        :return: Number of PCA dimensions to use
        '''

        assert self.pca_eigvals is not None, "'self.pca_eigvals' must not be None (must have run PCA already)"

        if self.pca_num_dims is None and self.pca_var_threshold is None:
            return self.pca_eigvals.size
        # end if using all PCA dimensions

        if self.pca_num_dims is not None:
            assert self.pca_num_dims > 0 and round(self.pca_num_dims) == self.pca_num_dims,\
                f"'self.pca_num_dims' must be a positive integer (currently set to {self.pca_num_dims})"
            return self.pca_num_dims
        # end if using pca_num_dims

        ## otherwise, use pca_var_threshold

        assert 0 < self.pca_var_threshold <= 1,\
            f"'self.pca_var_threshold' must be in (0, 1] (currently set to {self.pca_var_threshold})"

        if self.pca_var_threshold == 1:
            # make this case explicit to avoid machine-precision corner cases
            return self.pca_eigvals.size
        else:
            # find minimum number of dimensions that satisfies explained-variance threshold
            dim_idx = np.where(self.pca_cum_var >= self.pca_var_threshold)[0][0]
            return dim_idx + 1
        # end if/else pca_var_threshold is exactly 1 or not
    # end method _choose_num_pca_dims


    def create_frame_0(self, dat_raw=None, _num_pca_dims=None):
        '''
        Create frame 0 from raw data.

        If dat_raw is passed in, we return the created frame 0. Otherwise, we use self.raw_data and store the
        created frame 0 in self.frames.

        If self.pca_transform is True, frame 0 will be the PCA-rotated/truncated coordinates of each row.
        Otherwise, frame 0 will simply be the raw data.

        Note: if dat_raw is passed in and pca_transform is True, we apply the 'in-sample' PCA rotation/truncation
        derived originally from self.raw_data. It's important that any new 'out-of-sample' points be transformed
        using the in-sample PCA transformation. (For more detail, see the discussion of running new points in
        the user guide.)

        :param dat_raw: A raw-data matrix.  if None, we use self.raw_data.  default None.
        :param _num_pca_dims: ONLY USED BY INTERNAL CODE. Use self.pca_num_dims or self.pca_var_threshold instead.
            Default None.
        :return: If 'dat_raw' was passed in, we return frame 0.  otherwise, we return None.
        '''

        if dat_raw is None:
            # do 'in-sample' setup based on self.raw_data, then create frame 0 and store it in self.frames

            assert self.raw_data is not None, 'must have raw data to build frame 0'
            assert self.frames is None or self.frames.shape[2] == 1, \
                "must not already have multiple frames when creating frame 0 (use clear_frames to clear frames)"

            if self.pca_transform:
                assert _num_pca_dims is None, "'_num_pca_dims' must not be passed in when building 'in-sample'" \
                                             "version of frame 0 from self.raw_data (use self.pca_num_dims or" \
                                             "self.pca_var_threshold)"

                if self.pca_eigvecs is None:
                    if self.verbose:
                        print('running PCA...')
                    self.run_pca()
                # end if PCA not run/stored yet

                _num_pca_dims = self._choose_num_pca_dims()
            # end if using PCA transformation

            # create frame 0 based on self.raw_data and store in self.frames
            self.frames = self.create_frame_0(self.raw_data, _num_pca_dims)

            return
        # end if using self.raw_data

        t0 = time()

        if self.pca_transform:
            # center the raw data (always using the 'in-sample' column means)
            assert dat_raw.shape[1] == self.raw_col_means.size, "'dat_raw' must have the expected number of columns"
            dat = dat_raw - self.raw_col_means

            # if _num_pca_dims is None, infer it from self.frames
            if _num_pca_dims is None:
                assert self.frames is not None and type(self.frames) is np.ndarray,\
                    "'self.frames' must be an ndarray (when creating frame 0 for raw data passed in)"
                _num_pca_dims = self.frames.shape[1]
            # end if pca_num_dims is None

            # rotate and truncate using the specified number of PCA dimensions
            if _num_pca_dims > self.pca_eigvals.size:
                if self.verbose:
                    print('## WARNING: {} PCA dims requested, but only have {} -- using all {} PCA dimensions...'
                          .format(_num_pca_dims, self.pca_eigvals.size, self.pca_eigvals.size))
                _num_pca_dims = self.pca_eigvals.size
            # end if too many PCA dims requested
            eigvecs = self.pca_eigvecs[:, :_num_pca_dims]
            # NOTE: numpy matrix multiplication seems to be really slow (sometimes?) for no good reason. if we
            # build each column of the final matrix separately and then cat them together, the whole thing goes
            # much faster.
            # 2FIX: IS THE PROBLEM THAT CHANGING DAT 'IN PLACE' IS CONFUSING THE NUMPY CALCULATIONS?
            # dat = dat @ eigvecs  # this is (sometimes?) extremely slow
            new_dat = dat @ eigvecs[:, 0:1]  # 0:1 indexing is to keep the column vector 2-D
            for dim_idx in range(1, eigvecs.shape[1]):
                new_dat = np.concatenate((new_dat, dat @ eigvecs[:, dim_idx:dim_idx + 1]), axis=1)
            dat = new_dat

            if self.verbose:
                print('using {} of {} PCA dimensions ({:.1f}% of total variance)'.\
                      format(_num_pca_dims, self.pca_eigvals.size, 100 * self.pca_cum_var[_num_pca_dims - 1]))
        else:
            # not doing PCA transformation -- just use raw data
            dat = dat_raw
        # end if/else doing PCA transformation or not

        # make frame 0 a 3-D array
        frame0 = dat[:, :, np.newaxis]

        t1 = time()
        if self.verbose and t1 - t0 >= self.min_report_time:
            print("created frame 0 in {} seconds".format(round(t1 - t0)))

        return frame0
    # end method create_frame_0


    def clear_basis(self):
        '''
        Clear instance variables storing information about the basis, INCLUDING self.basis_size.

        Use this method to clear a basis when you want to return to the default behavior of using
        all rows as the basis.

        :return: None
        '''

        self.basis_size = None
        self.basis_row_nums = None
        self.non_basis_row_nums = None
        self.basis_rows = None
    # end method clear_basis


    def _set_basis(self, basis_row_nums=None):
        '''
        Set basis row nums and basis rows

        :param basis_row_nums: List of basis row numbers. If None, we use all rows as the basis.
        :return: None
        '''

        assert type(self.frames) is np.ndarray and self.frames.ndim == 3, 'must have frame 0 to set basis'

        if basis_row_nums is None:
            basis_row_nums = list(range(self.frames.shape[0]))  # use all rows as the basis by default

        assert type(basis_row_nums) is list, 'basis_row_nums must be a list'
        assert len(basis_row_nums) > 0, 'basis_row_nums must have at least 1 row'
        assert min(basis_row_nums) >= 0 and max(basis_row_nums) < self.frames.shape[0] and \
                len(basis_row_nums) <= self.frames.shape[0] and len(set(basis_row_nums)) == len(basis_row_nums), \
                'basis_row_nums must have valid, unique row numbers'

        self.basis_row_nums = copy(basis_row_nums)
        # find non-basis row numbers (using sets -- much faster than list comprehension for large number of rows)
        self.non_basis_row_nums = list(set(range(self.frames.shape[0])).difference(set(basis_row_nums)))
        self.basis_rows = np.copy(self.frames[basis_row_nums, :, 0])  # note: basis_rows is 2-D, not 3-D
    # end method _set_basis


    def build_operators(self, n_potential=None):
        '''
        Build the DQM operators and store them in the instance.

        If basis has not been set, we use all rows as the basis by default. (For large numbers of rows, this
        default will be unusably slow.)

        Note: the relative order of the basis rows is baked into the operators, so the relative ordering of
        the basis rows must not change later (when building frames).

        The operators are:

        * simt: Transpose of 'similarity' matrix, which converts state vectors from 'raw' basis of basis rows
          to orthonormal basis of eigenstates. Dimensions: <num basis vectors x num basis rows>.
        * xops: 3-D array, where slice i in 3rd dimension is the position-expectation operator matrix for data
          dimension i (in 2nd dimension in frames). Dimensions: <num basis vectors x num basis vectors x num dims>.
        * exph: Complex-valued 'evolution' operator matrix (the exponentiated time-evolution Hamiltonian operator
          matrix). Dimensions: <num basis vectors x num basis vectors>.

        :param n_potential: USED MAINLY FOR DEBUGGING AND SPEED TESTING. Use this number of rows to build
            the potential, starting from the first row. If None, we use all rows (not just the basis rows).
            Default None.
        :return: None
        '''

        assert type(self.frames) is np.ndarray and self.frames.ndim == 3, 'must have frame 0 to build operators'
        assert self.frames.shape[2] == 1, \
            "must not already have multiple frames when building operators (use clear_frames to clear frames)"

        if self.mass is None:
            self.mass = self.default_mass_for_num_dims()
            if self.verbose:
                print('mass was not set -- setting mass to {:.3f} for {} dimensions'
                      .format(self.mass, self.frames.shape[1]))
            # end if verbose
        # envd if mass was None

        assert self.sigma > 0 and self.step > 0 and self.mass > 0, \
            'all parameters (self.sigma, self.step, self.mass) must be positive to build operators'

        if self.basis_row_nums is None:
            if self.verbose:
                print(f'basis was not set -- using full basis (all {self.frames.shape[0]} rows)')
            self._set_basis()

        if n_potential is None:
            n_potential = self.frames.shape[0]  # use all rows to build the potential

        num_basis_rows = self.basis_rows.shape[0]

        if self.verbose:
            print('building operators for {:,} basis rows and {:,} potential rows...'.\
                  format(num_basis_rows, n_potential))

        if num_basis_rows < self.frames.shape[0]:
            # MakeOperatorsC expects the basis rows to be first in the matrix of rows, so reorder the rows
            shuffled_row_nums = self.basis_row_nums + self.non_basis_row_nums
            # note: shuffling the rows causes numpy to make a copy, which is what we want (since we need
            # mat's underlying memory to be contiguous)
            mat = self.frames[shuffled_row_nums, :, 0]
        else:
            mat = self.frames[:, :, 0]

        mat = np.ascontiguousarray(mat)

        t0 = time()

        if self.call_c:
            simt = np.zeros((num_basis_rows, num_basis_rows), dtype=np.float64)

            # set up xops so that the C code sees the allocated memory in C-friendly order (see note on 3-D
            # arrays at the top of this file).  xops is supposed to be <num_basis_rows x num_basis_rows x num_cols>
            # (and will be, see below)
            xops = np.zeros((self.frames.shape[1], num_basis_rows, num_basis_rows), dtype=np.float64)

            exph = np.zeros((num_basis_rows, num_basis_rows), dtype=np.complex128)

            if dqm_lib is not None:
                num_basis_vecs = dqm_lib.MakeOperatorsC(mat, self.frames.shape[0], self.frames.shape[1],
                                                        num_basis_rows, n_potential, self.sigma, self.step,
                                                        self.mass, simt, xops, exph)
            else:
                raise RuntimeError("in DQM instance, 'call_c' is True but compiled-library code not found")

            # reorder the dimensions to make xops <num_basis_rows x num_basis_rows x num_cols>
            # (note: we make the array contiguous here because it will be contiguous if we save it to disk and then
            # load it again.  making it contiguous here keeps things consistent for later operations.)
            xops = np.ascontiguousarray(np.transpose(xops, (1, 2, 0)))

            # if the number of basis eigenstates is less than the number of basis rows, we need to subselect the
            # output arrays appropriately.  (note: we need to make copies when subselecting to keep the underlying
            # memory contiguous.)
            if num_basis_vecs < num_basis_rows:
                if self.verbose:
                    print('number of eigenstates ({}) is less than number of basis rows ({}) -- \
                            subselecting operators...'.format(num_basis_vecs, num_basis_rows))
                # simt should be <num_basis_vecs x num_basis_rows>
                simt = np.copy(simt[:num_basis_vecs, :])
                # xops should be <num_basis_vecs x num_basis_vecs x num_cols>
                xops = np.copy(xops[:num_basis_vecs, :num_basis_vecs, :])
                # exph should be <num_basis_vecs x num_basis_vecs>
                exph = np.copy(exph[:num_basis_vecs, :num_basis_vecs])
            # end if subselecting for num_basis_vecs
        else:
            simt, xops, exph = make_operators_python(mat, num_basis_rows, n_potential,
                                                     self.sigma, self.step, self.mass)
        # end if/else calling C or Python

        t1 = time()
        if self.verbose and t1 - t0 >= self.min_report_time:
            print("built operators in {} seconds".format(round(t1 - t0)))

        self.simt, self.xops, self.exph = simt, xops, exph
    # end method build_operators


    def choose_basis_by_distance(self):
        '''
        Choose and store a set of basis rows, based on Euclidean distance, and store the results.

        self.basis_size must be set to a positive number less than the number of rows in self.frames (which
        must exist).

        First basis row: if self.basis_start_with_outlier is True, we use the largest outlier (with the farthest
        nearest neighbor) as the first basis row. Otherwise, we choose a random row as the first basis row.

        Subsequent basis rows: we choose the non-basis row whose closest distance to any current basis row is
        largest, until the desired basis size is reached.

        If self.basis_num_chunks is > 1, we partition all rows into 'chunks' and choose basis rows separately for
        each chunk. (This is faster but less 'accurate', since 2 basis rows in 2 different chunks may be arbitrarily
        close to each other.)

        :return: None
        '''

        # if using multiple chunks, shuffle the rows to remove any bias in row ordering
        shuffle_rows = self.basis_num_chunks > 1
        # if shuffling the rows or choosing a random start row, we'll need a random generator
        randomizing = shuffle_rows or not self.basis_start_with_outlier

        assert type(self.frames) is np.ndarray and self.frames.ndim == 3, 'must have frame 0 to choose basis'
        frame0 = self.frames[:, :, 0]

        basis_row_nums = []
        num_rows = self.frames.shape[0]
        num_chunks = self.basis_num_chunks
        basis_size = self.basis_size

        assert basis_size is not None and round(basis_size) == basis_size,\
            "'self.basis_size' must be an integer value"
        basis_size = int(basis_size)
        assert self.basis_size > 0 and self.basis_size < num_rows,\
            'desired basis size must be positive and less than number of rows to choose basis'

        if self.verbose:
            print('choosing {:,} basis rows by distance...'.format(basis_size))

        rng = None
        if randomizing:
            rng = np.random.default_rng(self.basis_rand_seed)

        if shuffle_rows:
            shuffled_row_idxs = rng.permutation(num_rows)
            frame0 = frame0[shuffled_row_idxs, :]

        # set up chunk row numbers
        num_per_chunk = int(np.ceil(num_rows / num_chunks))
        start_idxs = [num_per_chunk * i for i in range(num_chunks)]
        end_idxs = [min(num_per_chunk * (i + 1), num_rows) for i in range(num_chunks)]

        # set up chunk basis sizes
        chunk_basis_size = int(np.ceil(self.basis_size / num_chunks))
        chunk_basis_sizes = [chunk_basis_size for i in range(num_chunks)]
        if num_chunks > 1:
            # tweak basis size for last chunk to make total come out right
            chunk_basis_sizes[-1] = basis_size - sum(chunk_basis_sizes[:-1])

        t0 = time()

        # choose basis rows for each chunk
        for chunk_idx in range(num_chunks):
            chunk_basis_size = chunk_basis_sizes[chunk_idx]
            chunk_row_nums = list(range(start_idxs[chunk_idx], end_idxs[chunk_idx]))
            chunk_rows = frame0[chunk_row_nums, :]
            chunk_basis_idxs = self._choose_basis_by_distance_single_chunk(chunk_rows, chunk_basis_size, rng)
            # convert back to original (possibly shuffled) row numbers
            chunk_basis_row_nums = [chunk_row_nums[i] for i in chunk_basis_idxs]
            basis_row_nums += chunk_basis_row_nums
        # end for each chunk

        if shuffle_rows:
            # return to original row numbers
            basis_row_nums = shuffled_row_idxs[basis_row_nums].tolist()

        t1 = time()
        if self.verbose and t1 - t0 >= self.min_report_time:
            if num_chunks > 1:
                print('chose {} basis rows (in {} chunks) in {} seconds'.\
                      format(basis_size, num_chunks, round(t1 - t0)))
            else:
                print('chose {} basis rows in {} seconds'.format(basis_size, round(t1 - t0)))

        self._set_basis(basis_row_nums)
    # end method choose_basis_by_distance


    def _choose_basis_by_distance_single_chunk(self, rows, basis_size, rng):
        '''
        Choose basis_size number of rows from rows to act as a basis.

        See comments for choose_basis_by_distance for more details.

        :param rows: Matrix of rows
        :param basis_size: Number of rows to select for the basis
        :param rng: Numpy random-number generator
        :return: List of selected basis row numbers
        '''

        num_rows, num_cols = rows.shape
        assert basis_size < num_rows, 'desired basis size must be smaller than number of rows in chunk'

        if self.basis_start_with_outlier:
            # sentinel value, which tells later code to start with the largest outlier
            first_basis_row_num = -1
        else:
            # choose random row number as the first basis row
            assert rng is not None, 'must have random generator to start basis chunk with random row'
            first_basis_row_num = rng.integers(num_rows)

        if self.call_c:
            basis_row_nums = np.zeros(basis_size, dtype=np.int32)
            if dqm_lib is not None:
                dqm_lib.ChooseBasisByDistanceC(rows, num_rows, num_cols, basis_size, basis_row_nums,
                                               first_basis_row_num)
            else:
                raise RuntimeError("in DQM instance, 'call_c' is True but compiled-library code not found")
            # end if/else have compiled-library instance or not
        else:
            basis_row_nums = choose_basis_by_distance_python(rows, basis_size, first_basis_row_num)
        # end if/else calling C or Python

        return basis_row_nums.tolist()
    # end method _choose_basis_by_distance_single_chunk


    def build_overlaps(self, rows=None, row_nums=None, sigma=None, batch_size=int(100e3), verbose=None):
        '''
        Build basis overlaps for a given set of rows.

        If 'rows' is passed in, we build overlaps for those rows. Otherwise, if row_nums is passed in,
        we build overlaps for those rows. Otherwise, we build overlaps for all non-basis rows.

        'Overlap' is a measure of how well a given data point is represented by the basis. Basis points will
        all have an overlap of 1, meaning perfect representation. (For technical details, see the section on
        "Reconstruction of Wave Functions in the Eigenbasis" in the technical-summary document "Understanding
        DQM".)

        :param rows: 2-D array of data rows. Takes precedence if not None.  Default None.
        :param row_nums: List of row numbers, used if rows is None. Default None.
        :param sigma: Value of sigma. If None, we use self.sigma. Default None.
        :param batch_size: Number of rows in a batch. (For very large numbers of rows, memory management can
            become an issue.) Default 100,000.
        :param verbose: Boolean: if not None, overrides self.verbose. Default None.
        :return: Vector containing scalar overlap value for each row.
        '''

        t0 = time()

        if verbose is None:
            verbose = self.verbose

        assert self.basis_row_nums is not None and self.basis_rows is not None, \
            'must have basis to build overlaps'

        if rows is None:
            if row_nums is None:
                # build overlaps for all non-basis rows by default
                assert len(self.non_basis_row_nums) > 0, \
                    "must have some non-basis rows in order to build overlaps for them"
                row_nums = self.non_basis_row_nums
            assert min(row_nums) >= 0 and max(row_nums) < self.frames.shape[0], \
                'must have valid row numbers to build overlaps'
            rows = self.frames[row_nums, :, 0]
        # end if/else (rows, row nums, or neither passed in)

        assert type(rows) is np.ndarray and (rows.ndim == 2 or (rows.ndim == 3 and rows.shape[2] == 1)), \
            "'rows' must be 2-D ndarray (or 3-D with 1 slice in dim 3)"
        if rows.ndim == 3:
            rows = rows[:, :, 0]  # make it 2-D

        if sigma is None:
            sigma = self.sigma
        assert sigma is not None and sigma > 0, 'must have positive value of sigma to build overlaps'

        num_rows = rows.shape[0]
        num_basis_rows = self.basis_rows.shape[0]

        assert self.frames.shape[1] == rows.shape[1], \
            "'rows' must have correct number of columns to build overlaps"

        # a very large number of rows (more than a few million) causes Windows errors -- not sure why.
        # so, we run in batches.
        batch_size = int(batch_size)
        num_batches = int(np.ceil(num_rows / batch_size))

        if num_batches == 1:
            if self.call_c:
                overlaps = np.zeros(num_rows, dtype=np.float64)
                if dqm_lib is not None:
                    dqm_lib.BuildOverlapsC(sigma, self.basis_rows, rows, num_basis_rows, num_rows,
                                           self.frames.shape[1], overlaps)
                else:
                    raise RuntimeError("in DQM instance, 'call_c' is True but compiled-library code not found")
                # end if/else have compiled-library instance or not
            else:
                overlaps = build_overlaps_python(sigma, self.basis_rows, rows)
            # end if/else calling C or Python
        else:
            overlaps = np.zeros(0)  # empty vector
            for batch_idx in range(num_batches):
                start_idx = batch_size * batch_idx
                end_idx = min(num_rows, batch_size * (batch_idx + 1))
                batch_overlaps = self.build_overlaps(rows=rows[start_idx:end_idx, :], sigma=sigma,
                                                     batch_size=batch_size, verbose=verbose)
                overlaps = np.concatenate((overlaps, batch_overlaps))
            # end for each batch
        # end if/else multiple batches or not

        t1 = time()
        if verbose and t1 - t0 >= self.min_report_time:
            print("built {:,} overlaps in {} seconds".format(num_rows, round(t1 - t0)))

        return overlaps
    # end method build_overlaps


    def estimate_mean_row_distance(self, rel_err_threshold=0.01, rand_seed=500):
        '''
        Estimate the mean pairwise distance between rows in frame 0. (self.frames must exist.)

        Use a successively larger number of row pairs to estimate the overall mean distance, until the
        'relative error' (standard error of the mean divided by the mean) drops below rel_err_threshold.

        The final result is stored in self.mean_row_distance.

        :param rel_err_threshold: Threshold for 'relative error' (standard error of mean divided by mean).
            Must be positive. Default 0.01.
        :param rand_seed: Random seed for choosing row pairs for distance calculations. Default 500.
        :return: None
        '''

        assert type(self.frames) is np.ndarray, 'frame 0 must exist to estimate mean distance between rows'
        assert rel_err_threshold > 0, f"'rel_err_threshold must be positive', is currently {rel_err_threshold}"

        rows = self.frames[:, :, 0]
        num_rows = rows.shape[0]

        rng = np.random.default_rng(rand_seed)
        shuffled_row_nums = rng.permutation(num_rows)

        # dists array will grow as needed (see below)
        dists_array_size = num_rows
        dists = np.zeros(dists_array_size)

        done = False
        row_idx1 = 0
        row_idx2 = 1
        num_pairs = 0
        while not done:
            # calculate store row-pair distance
            num_pairs += 1
            dists[num_pairs - 1] = np.linalg.norm(rows[shuffled_row_nums[row_idx1], :] -
                                                  rows[shuffled_row_nums[row_idx2], :])

            # update row-pair indices
            row_idx2 += 1
            if row_idx2 == num_rows:
                row_idx1 += 1
                if row_idx1 == num_rows - 1:
                    break  # we've run out of row pairs
                else:
                    row_idx2 = row_idx1 + 1
            # end if reached the end of the shuffled list of rows for row_idx2

            # calculate current relative error
            if num_pairs > 1:
                mu = np.mean(dists[:num_pairs])
                std = np.std(dists[:num_pairs])
                rel_err = std / np.sqrt(num_pairs) / mu
                done = rel_err <= rel_err_threshold
            # end if have multiple row-pair distances for calculations

            if num_pairs == dists.size and not done:
                # grow dists array as needed
                dists_array_size += num_rows
                new_dists = np.zeros(dists_array_size)
                new_dists[:num_pairs] = dists
                dists = new_dists
            # end if growing dists array
        # end while not done

        if self.verbose:
            report_precision = floor(log10(mu)) - 2
            print('estimated mean distance between rows is {:.{}f} (relative error {:.1f}%, from {:,} row pairs)'.
                  format(mu, max(0, -report_precision), 100 * rel_err, num_pairs))

        self.mean_row_distance = mu

        return None
    # end method estimate_mean_row_distance


    def choose_sigma_for_basis(self, batch_size=None, num_batches_to_test=10, rand_seed=11):
        '''
        Choose the smallest value of sigma that meets overlap-threshold requirements (self.overlap_min_threshold
        and self.overlap_mean_threshold) for non-basis rows.

        self must already have frame 0 (in self.frames) and a selected basis that is smaller than the total number
        of rows (i.e., not a 'full' basis).

        We set self.sigma to the final resulting value of sigma.

        Default values for the arguments are good enough in most cases. (The 'batch' mode is for handling large
        numbers of non-basis rows more efficiently.)

        :param batch_size: Number of rows in a single batch.  if None, test all non-basis rows at once.
            Default None.
        :param num_batches_to_test: Number of batches that must return the same value of sigma before
            we're done (if batch_size is not None). Default 10.
        :param rand_seed: random seed for shuffling the order of non-basis row numbers. Default 11.
        :return: None
        '''

        '''
        Batch logic

        * Using rand_seed, shuffle all non-basis rows into a random order.
        * If batch_size is None, pass all non-basis rows to _choose_sigma_for_rows in a single batch.
        * If batch_size is not None, call _choose_sigma_for_rows 1 batch at a time, continuing as long as
          the returned value of sigma is the same for every batch. When a mismatch occurs, increase the
          batch size by 25% and start over.
        * When num_batches_to_test batches all return the same value of sigma, we have our final selected value.
        '''

        assert self.basis_rows.shape[0] < self.frames.shape[0],\
            'must have some non-basis rows to choose sigma for basis'

        # shuffle all non-basis row numbers
        row_nums = np.array(self.non_basis_row_nums)
        num_rows = row_nums.size
        rng = np.random.default_rng(rand_seed)
        shuffled_row_nums = row_nums[rng.permutation(num_rows)]

        if batch_size is None:
            # if no batch size given, test all non-basis rows at once
            self.sigma = self._choose_sigma_for_rows(shuffled_row_nums)
            return
        # end if no batch size (testing all non-basis rows at once)

        done = False
        while not done:
            batch_size = int(batch_size)
            num_batches = int(np.ceil(num_rows / batch_size))
            num_batches_to_test = min(num_batches_to_test, num_batches)
            mismatch = False
            sigma = None

            if self.verbose:
                print('##### to choose sigma, testing {:,} batches (batch size {:,}) #####'.
                      format(num_batches_to_test, batch_size))

            for batch_idx in range(num_batches_to_test):
                start_row_idx = batch_size * batch_idx
                end_row_idx = min(num_rows, batch_size * (batch_idx + 1))
                batch_row_nums = shuffled_row_nums[start_row_idx:end_row_idx]

                batch_sigma = self._choose_sigma_for_rows(batch_row_nums, verbose=False)
                if self.verbose:
                    print('batch {}: sigma is {}'.format(batch_idx, batch_sigma))
                if sigma is None:
                    sigma = batch_sigma
                else:
                    if sigma != batch_sigma:
                        if self.verbose:
                            print("sigma values don't agree -- increasing batch size...")
                        mismatch = True
                        batch_size *= 1.25
                        break
                # end if/else first batch or not
            # end for each test batch

            done = not mismatch
        # end while not done

        self.sigma = sigma
    # end method choose_sigma_for_basis


    def _choose_sigma_for_rows(self, row_nums, verbose=None):
        '''
        For the given row_nums, use a binary search to find the smallest value of sigma that meets
        overlap-threshold requirements (self.overlap_min_threshold and self.overlap_mean_threshold).

        self must already have frame 0 (in self.frames) and a selected basis.

        We determine precision of the search as follows:

        * Estimate mean distance between rows by calling self.estimate_mean_row_distance (if needed).
        * Set precision at least 2 orders of magnitude below the mean distance:
          precision = 10 ** (floor(log10(mean_distance)) - 2).
          For example, if mean distance is 20, precision will be 0.1.
        * Precision determines the smallest allowed step from one value of sigma to the next.

        The first search value for sigma is 10 * precision. If that first value of sigma is good, we divide
        sigma by 2 until we find a bad value. (If the largest bad value of sigma is below the current
        precision level, we increase precision as needed.) If that first value of sigma is bad, we multiply
        sigma by 2 until we find a good value. Once we have a bad value and a good value, we proceed by binary
        search, until we have a bad value and a good value within one precision step of each other. The larger,
        good, value is the final selected value of sigma.

        :param row_nums: List/array of row numbers to use for testing overlaps (must not include any basis
            row numbers).
        :param verbose: Boolean: if not None, overrides self.verbose. Default None.
        :return: Final value of sigma.
        '''

        if verbose is None:
            verbose = self.verbose

        # intersect row_nums and self.basis_row_nums, make sure they're disjoint
        basis_row_nums = np.array(list(set(row_nums).intersection(set(self.basis_row_nums))))
        assert basis_row_nums.size == 0, "must not have any basis row numbers in 'row_nums'"

        rows = self.frames[row_nums, :, 0]

        if self.mean_row_distance is None:
            self.estimate_mean_row_distance()
        # precision is at least 2 orders of magnitude below mean distance
        precision = 10 ** (floor(log10(self.mean_row_distance)) - 2)

        # use 'epsilon' to avoid machine-precision issues in comparisons
        eps = precision / 1e4

        # don't allow precision to shrink more than another 3 orders of magnitude (so, still 10 times
        # bigger than epsilon)
        min_precision = (precision / 1e3) - eps

        # starting value of sigma
        sigma = 10 * precision

        # do binary search within this range, once both of these values are non-zero
        sigma_range = [0, 0]

        if verbose:
            print('choosing sigma to precision of {:.1e} for basis of size {:,}...'.
                  format(precision, self.basis_rows.shape[0]))

        done_searching = False
        while not done_searching:
            # test current value of sigma
            overlaps = self.build_overlaps(rows=rows, sigma=sigma)
            min_overlap = np.min(overlaps)
            mean_overlap = np.mean(overlaps)
            sigma_is_good = min_overlap >= self.overlap_min_threshold and mean_overlap >= self.overlap_mean_threshold
            if verbose:
                print('for sigma = {:.{}f}: min overlap {:.3f}, mean overlap {:.3f}{}'.
                      format(sigma, max(0, round(-log10(precision))), min_overlap, mean_overlap,
                             ' (GOOD)' if sigma_is_good else ''))

            if sigma_is_good:
                # search for smaller sigma (so, current sigma now defines the top of the search range)
                sigma_range[1] = sigma
                doing_binary_search = sigma_range[0] > 0
                if not doing_binary_search:
                    # if we're at minimum sigma for current precision, we need to increase precision
                    if sigma < precision + eps:
                        precision /= 10
                        assert precision >= min_precision, \
                            'precision must not go below {:.1e}'.format(min_precision)
                        if verbose:
                            print('increasing precision to {:.1e}...'.format(precision))
                    # end if increasing precision
                    sigma = round(sigma / 2, round(-log10(precision)))  # search for smaller sigma
                    assert sigma > 0, 'sigma must always be positive'
            else:
                # search for larger sigma (so, current sigma now defines the bottom of the search range)
                sigma_range[0] = sigma
                doing_binary_search = sigma_range[1] > 0
                if not doing_binary_search:
                    sigma = round(sigma * 2, round(-log10(precision)))  # search for larger sigma
            # end if/else searching for smaller or larger sigma

            if doing_binary_search:
                sigma = round(np.mean(sigma_range), round(-log10(precision)))
                assert sigma > 0, 'sigma must always be positive'

            # to be done, top and bottom of search range must both be positive, and search range must be
            # at or below current precision
            done_searching = sigma_range[0] > 0 and sigma_range[1] > 0 and \
                             sigma_range[1] - sigma_range[0] < precision + eps
        # end while not done searching

        sigma = sigma_range[1]
        if verbose:
            print('final sigma is {:.{}f}'.format(sigma, max(0, round(-log10(precision)))))

        return sigma
    # end method _choose_sigma_for_rows


    def _stopped_row_nums(self, frames=None):
        '''
        For the given set of frames, return a list of row numbers for rows that have stopped (according to
        self.stopping_threshold).

        :param frames: A 3-D array of frames. If None, we use self.frames (which must exist). Default None.
        :return: List of stopped row numbers.
        '''

        if frames is None:
            assert self.frames is not None, 'must have frames in order to determine stopped row numbers'
            return self._stopped_row_nums(self.frames)
        # end if frames not passed in

        assert type(frames) is np.ndarray and frames.ndim in [2, 3], "'frames' must be a 2-D or 3-D ndarray"

        assert self.stopping_threshold is not None, 'must have stopping_threshold to determine stopped rows'

        if frames.ndim < 3 or frames.shape[2] <= 1:
            return []  # need 2 frames to determine stopping

        last_deltas = frames[:, :, -1] - frames[:, :, -2]
        dists = np.linalg.norm(last_deltas, axis=1)
        num_rows = frames.shape[0]
        stopped_row_nums = [i for i in range(num_rows) if dists[i] < self.stopping_threshold]
        return stopped_row_nums
    # end method _stopped_row_nums


    def set_stopping_threshold(self):
        '''
        Set self.stopping_threshold to self.mean_row_distance / 1e6. (First call self.mean_row_distance, if needed.)

        :return: None.
        '''

        if self.mean_row_distance is None:
            self.estimate_mean_row_distance()
        self.stopping_threshold = self.mean_row_distance / 1e6
        if self.verbose:
            print('set stopping threshold to {:.2e}'.format(self.stopping_threshold))
    # end method set_stopping_threshold


    def build_frames(self, num_frames_to_build=100, frames=None, pare_frames=True, verbose=None):
        '''
        Build new frames in a DQM evolution and concatenate them with existing frames.

        Instance must have basis rows, operators, and positive sigma.

        If 'frames' is passed in, we return all frames (old and new together).  otherwise, we set
        self.frames to be all frames (old and new together) and return None.

        Stopped rows are not evolved further. (A row is 'stopped' when it fails to move at least
        self.stopping_threshold distance from one frame to the next.)

        :param num_frames_to_build: Number of new frames to build. Default 100.
        :param frames: 3-D array of existing frames (<num rows x num dims x num frames>). If None, we use
            self.frames. Default None.
        :param pare_frames: Boolean: if True, we delete any final frames where nothing is changing. Default True.
        :param verbose: Boolean: if not None, overrides self.verbose. Default None.
        :return: If 'frames' was passed in, we return all frames (old and new together). Otherwise,
            we return None.
        '''

        if verbose is None:
            verbose = self.verbose

        if frames is None:
            self.frames = self.build_frames(num_frames_to_build, self.frames, pare_frames, verbose=verbose)
            return
        # end if using self.frames

        assert type(frames) is np.ndarray and frames.ndim == 3, "'frames' must be a 3-D ndarray"
        assert num_frames_to_build > 0, "'num_frames_to_build' must be positive"
        assert self.sigma is not None and self.sigma > 0, 'sigma must be positive'
        assert self.basis_rows is not None, 'must have basis rows to build frames'
        assert self.simt is not None and self.xops is not None and self.exph is not None, \
            'must have operators to build frames'

        num_rows, num_cols = frames.shape[:2]
        current_frame = np.copy(frames[:, :, -1])  # make a copy to be sure memory is contiguous

        assert num_cols == self.frames.shape[1], "'frames' must have correct number of columns"

        if self.stopping_threshold is None:
            self.set_stopping_threshold()

        # deal with stopped rows
        stopped_row_nums = self._stopped_row_nums(frames)
        not_stopped_row_nums = list(set(list(range(num_rows))).difference(set(stopped_row_nums)))
        if len(not_stopped_row_nums) == 0:
            if verbose:
                print('all rows have stopped -- no frames added')
            if pare_frames:
                frames = self.pare_frames(frames)
            return frames
        # end if all rows stopped
        num_evolving_rows = len(not_stopped_row_nums)

        have_stopped_rows = len(stopped_row_nums) > 0
        if have_stopped_rows:
            current_frame = current_frame[not_stopped_row_nums, :]

        t0 = time()

        if self.call_c:
            # set up new_frames so that the C code sees the allocated memory in C-friendly order (just as we
            # did with xops in build_operators, which see).  new_frames is supposed to be
            # <num_evolving_rows x num_cols x num_frames_to_build> (and will be, see below)
            new_frames = np.zeros((num_frames_to_build, num_evolving_rows, num_cols), dtype=np.float64)

            # shuffle the xops data into C-friendly order (see note at top of file)
            xops = np.ascontiguousarray(np.transpose(self.xops, (2, 0, 1)))

            num_basis_vecs = self.exph.shape[0]
            if dqm_lib is not None:
                dqm_lib.BuildFramesAutoC(new_frames, num_evolving_rows, num_cols, num_frames_to_build, current_frame,
                                         self.basis_rows, self.basis_rows.shape[0], self.simt, num_basis_vecs,
                                         xops, self.exph, self.sigma, self.stopping_threshold)
            else:
                raise RuntimeError("in DQM instance, 'call_c' is True but compiled-library code not found")
            # end if/else have compiled-library instance or not

            # make new_frames <num_evolving_rows x num_cols x num_frames_to_build>
            # (note: new_frames will now not be C_CONTIGUOUS, but everything else we're going to do with
            # new_frames is here in Python, so we don't care about the underlying memory order anymore,
            # we just let numpy handle it.)
            new_frames = np.transpose(new_frames, (1, 2, 0))
        else:
            # call the Python-only version
            new_frames = build_frames_python(num_frames_to_build, current_frame, self.basis_rows, self.simt,
                                             self.xops, self.exph, self.sigma, self.stopping_threshold)
        # end if/else calling C or Python

        t1 = time()
        if verbose and t1 - t0 >= self.min_report_time:
            print("built {} frames in {} seconds".format(num_frames_to_build, round(t1 - t0)))

        if have_stopped_rows:
            # fill the stopped rows forward
            new_frames_all = frames[:, :, -1][:, :, np.newaxis]  # current frame (unsubselected, 3-D)
            new_frames_all = np.repeat(new_frames_all, num_frames_to_build, axis=2)
            # overwrite where we have new data for evolving rows
            new_frames_all[not_stopped_row_nums, :, :] = new_frames
        else:
            new_frames_all = new_frames
        # end if/else any stopped rows or not

        frames = np.concatenate((frames, new_frames_all), axis=2)

        if pare_frames:
            frames = self.pare_frames(frames)

        return frames
    # end method build_frames


    def build_frames_auto(self, batch_size=100, frames=None, pare_frames=True, max_num_frames=int(1e4)):
        '''
        Add new frames in batches (by calling build_frames) until all rows have stopped.

        If 'frames' is passed in, we return all frames (old and new together). Otherwise, we set
        self.frames to be all frames (old and new together) and return None.

        :param batch_size: Number of new frames to add in each batch. Default 100.
        :param frames: 3-D array of frames (<num rows x num dims x num frames>). If None, we use self.frames.
            Default None.
        :param pare_frames: Boolean: if True, we delete any final frames where nothing is changing. Default True.
        :param max_num_frames: Maximum number of frames, including any initial frames. Default 10,000. (This
            parameter is important because a too small value of mass can cause data points to oscillate around a
            minimum -- overshooting the minimum at each step -- meaning that they will never stop moving.)
        :return: If 'frames' was passed in, we return all frames (old and new together). Otherwise,
            we return None.
        '''

        if frames is None:
            self.frames = self.build_frames_auto(batch_size, self.frames, pare_frames, max_num_frames)
            return
        # end if using self.frames

        assert type(frames) is np.ndarray and frames.ndim == 3, "'frames' must be a 3-D ndarray"
        assert self.sigma is not None and self.sigma > 0, 'sigma must be positive'

        if self.stopping_threshold is None:
            self.set_stopping_threshold()
        assert self.stopping_threshold >= 1e-10 * self.sigma, \
            'stopping_threshold must be >= 1e-10 * sigma (to prevent build_frames_auto from running forever)'

        num_frames_start = frames.shape[2]

        t0 = time()

        num_frames1 = -1
        num_frames2 = 0
        while num_frames2 > num_frames1:
            if self.verbose:
                print(f'adding {batch_size} frames...')
            num_frames1 = frames.shape[2]
            frames = self.build_frames(batch_size, frames, pare_frames=False, verbose=False)
            num_frames2 = frames.shape[2]
            if num_frames2 >= max_num_frames:
                if self.verbose:
                    print(f'WARNING: have reached or exceeded max num frames of {max_num_frames}\
                          (current num frames is {num_frames2})')
                break
        # end while still adding frames

        if pare_frames:
            frames = self.pare_frames(frames)

        t1 = time()

        if self.verbose:
            num_frames_end = num_frames2
            print(f'added a total of {num_frames_end - num_frames_start} frames in {round(t1 - t0)} seconds')

        return frames
    # end method build_frames_auto


    def pare_frames(self, frames):
        '''
        Drop any duplicate frames (where frame n + 1 is identical to frame n) at the end of an evolution.

        :param frames: A 3-D array of frames. No default.
        :return: A pared 3-D array of frames (which will be a reference to the frames passed in if no
            frames were dropped).
        '''

        assert type(frames) is np.ndarray and frames.ndim == 3, "'frames' must be a 3-D array"
        num_frames = frames.shape[2]

        if num_frames <= 1:
            return frames

        # use binary search to find the last time a pair of consecutive frames differs
        start_idx = 0
        end_idx = num_frames - 1
        done = False
        while not done:
            if end_idx - start_idx == 1:
                done = True
                start_same_as_end = np.array_equal(frames[:, :, start_idx], frames[:, :, end_idx])
                if start_same_as_end:
                    keep_idx = start_idx
                else:
                    keep_idx = end_idx
            else:
                # indices are more than 1 apart
                mid_idx = round((start_idx + end_idx) / 2)
                mid_same_as_end = np.array_equal(frames[:, :, mid_idx], frames[:, :, end_idx])
                if mid_same_as_end:
                    end_idx = mid_idx  # search downward (all frames in top half assumed to be the same)
                else:
                    start_idx = mid_idx  # search upward (all frames in bottom half assumed to be different)
            # end if/else indices only 1 apart or not
        # end while doing binary search

        return frames[:, :, :keep_idx + 1]
    # end method pare_frames


    def clear_frames(self, keep_frame_0=True):
        '''
        Keep frame 0 and clear all frames in self.frames after frame 0.

        (Note that create_frame_0 and build_operators will both fail if self.frames has multiple frames in 3rd dim.
        This is to prevent accidental loss of information, particularly since building frames can be slow.)

        :param keep_frame_0: Boolean: if False, we set self.frames to None. default True.
        :return: None
        '''

        if self.frames is None:
            if self.verbose:
                print("'self.frames' is not an array -- no frames to clear")
        elif not keep_frame_0:
            self.frames = None
        else:
            # 'reset' to frame 0
            self.frames = self.frames[:, :, 0:1]  # 0:1 indexing is to keep frame 0 as 3-D array
    # end method clear_frames


    def pca_projection(self, dat_raw=None, num_pca_dims=None):
        '''
        Apply PCA 'projection' (centering + rotation + truncation) to a raw data matrix, as follows:

        * Center the columns by subtracting self.raw_col_means (which must exist).
        * Create a rotated and truncated matrix by applying the combination of self.pca_eigvecs (rotation) and
          the number of PCA dimensions being used (truncation). (Number of PCA dimensions can be specified via
          the num_pca_dims parameter. If num_pca_dims is None, we infer the number of PCA dimensions being used
          from the 2nd dimension of self.frames, which must then exist.)
        * Calculate the proportional norms for each row in the raw data -- meaning, the centered/rotated/truncated
          L2 norm divided by the centered-only L2 norm.

        The instance must have stored PCA results.

        Importantly, if dat_raw is passed in, we apply the 'in-sample' PCA projection based on the original
        self.raw_data, *not* based on this new raw data. (For more detail, see the discussion of running new
        points in the user guide.)

        :param dat_raw: 2-D raw-data matrix. If None, we use self.raw_data. Default None.
        :param num_pca_dims: Number of PCA dimensions to use in the projection. If None, we infer from the
            number of columns in 2nd dimension of self.frames (which must then exist). default None.
        :return: A vector of proportional norms (projected / original) for each row.
        '''

        '''
        2FIX: add checks/warnings for this (probably very unlikely?) corner case

        if, after centering the data cloud, a data point is exactly at (or within machine precision of) the
        origin, then calculations here wil fail: either the original L2 norm will actually be zero (producing
        a divide-by-zero error), or the proportion of norms for this point will be dominated by noise.
        
        it's a little easier to imagine this case coming up in a scenario involving discrete data (e.g., many
        binary dimensions)...
        '''

        if dat_raw is None:
            dat_raw = self.raw_data
        assert type(dat_raw) is np.ndarray and dat_raw.ndim == 2, "'dat_raw' must be a 2-D ndarray"

        # always use in-sample column means
        assert self.raw_col_means is not None, 'must have raw column means'
        assert self.raw_col_means.size == dat_raw.shape[1], "'dat_raw' must have correct number of columns"

        if num_pca_dims is None:
            assert self.frames is not None and type(self.frames) is np.ndarray and self.frames.ndim == 3,\
                "must have 'self.frames' to infer number of PCA dimensions being used"
            num_pca_dims = self.frames.shape[1]
        # end if num_pca_dims is None

        t0 = time()

        dat = dat_raw - self.raw_col_means

        # get original row norms (after centering)
        norms_orig = np.linalg.norm(dat, axis=1)

        # get rotated/truncated row norms
        dat_rotated = dat @ self.pca_eigvecs[:, :num_pca_dims]
        norms_rotated = np.linalg.norm(dat_rotated, axis=1)

        norm_props = norms_rotated / norms_orig

        # error check: there should be no loss of information in the 'full' rotation (with no subspace
        # truncation/projection)
        #
        # NOTE: this actually isn't always true.  if the number of points/rows is less than the number of raw
        # dimensions when PCA is first run -- for example, if there are 20 points in 30 raw dimensions -- then
        # there will only be 20 PCA dimensions in total. these 20 PCA dimensions are enough to fully describe
        # the initial 20 points, but they are not enough to fully describe any arbitrary new point in the 30
        # raw dimensions (that would, of course, require 30 dimensions).
        dat_rotated_full = dat @ self.pca_eigvecs
        norms_rotated_full = np.linalg.norm(dat_rotated_full, axis=1)
        norm_props_full = norms_rotated_full / norms_orig
        if self.verbose and np.min(norm_props_full) < 0.999:
            print('WARNING: minimum norm proportion in full PCA rotation (with no subspace projection) is {:.4f}'.
                  format(np.min(norm_props_full)))

        t1 = time()
        if self.verbose and t1 - t0 >= self.min_report_time:
            print("calculated PCA-projection proportions in {} seconds".format(round(t1 - t0)))

        return norm_props
    # end method pca_projection


    def run_new_points(self, dat_raw_oos):
        '''
        Given dat_raw_oos, which is a raw-data matrix of new ('out-of-sample') points:

        * Apply the 'in-sample' PCA projection (subtract in-sample column means, apply in-sample
          PCA rotation, and truncate to in-sample number of PCA dimensions being used).
        * Build basis overlaps for the new points.
        * Evolve the new out-of-sample points, using the in-sample map (that is, the stored DQM operators),
          to as many frames as currently exist in self.frames.

        Important note: for running new out-of-sample points to make sense, the new raw data must be preprocessed
        in exactly the same way that the original raw data was.

        :param dat_raw_oos: A 2-D raw-data matrix of new 'out-of-sample' points. Must have the same number of
            columns as self.raw_data.
        :return: A tuple of:

            * frames_oos: 3-D array of out-of-sample evolved frames
            * overlaps_is: vector of in-sample basis overlaps (for all non-basis rows)
            * overlaps_oos: vector of out-of-sample basis overlaps
            * norm_props_is: vector of in-sample proportional norms (projected L2 norms divided by
              original L2 norms)
            * norm_props_oos: vector of out-of-sample proportional norms (projected L2 norms divided by
              original L2 norms)
        '''

        '''
        2FIX: create parameter to specify number of frames to build for new points?
        * as many as in self.frames (current default)
        * as many frames as needed for new points to stop
        * explicitly specified number of frames

        2FIX: add option where new points below specified thresholds for PCA-transformation proportional norms
        ('off the map') or basis overlaps ('in a blank spot on the map') are not evolved at all? (the fact that
        a low-overlap point ‘snaps’ closer to the basis points at the beginning of evolution is confusing and
        misleading. [ADDRESS THIS ISSUE MORE GENERALLY SOMEHOW?])
        '''

        assert type(dat_raw_oos) is np.ndarray and dat_raw_oos.ndim == 2, \
            "'dat_raw_oos' must be a 2-D ndarray"
        assert dat_raw_oos.shape[1] == self.raw_data.shape[1], \
            "'dat_raw_oos must have the same number of columns as self.raw_data"

        if self.pca_transform:
            assert type(self.raw_col_means) is np.ndarray and self.raw_col_means.ndim == 1, \
                "must have raw column means to run new out-of-sample points"

            # do the in-sample PCA projection for both in-sample points and out-of-sample points
            norm_props_is = self.pca_projection()
            norm_props_oos = self.pca_projection(dat_raw_oos)

            if self.verbose and HAVE_PLT:
                # plot histograms of in-sample and out-of-sample subspace proportions
                plt.figure(figsize=(22, 8))
                ax1 = plt.subplot(1, 2, 1)
                ax2 = plt.subplot(1, 2, 2)
                label_font = {'size': 15}
                title_font = {'size': 17}
                num_bins = 50
                plt.axes(ax1)
                plt.hist(norm_props_is, bins=num_bins)
                plt.xlabel('subspace norm as proportion of original norm', fontdict=label_font)
                plt.ylabel('count', fontdict=label_font)
                plt.title('In-Sample Proportion of L2 Norms for PCA Subspace Projection', fontdict=title_font)
                plt.axes(ax2)
                plt.hist(norm_props_oos, bins=num_bins)
                plt.xlabel('subspace norm as proportion of original norm', fontdict=label_font)
                plt.ylabel('count', fontdict=label_font)
                plt.title('Out-of-Sample Proportion of L2 Norms for PCA Subspace Projection', fontdict=title_font)
                plt.show()
            # end if verbose
        else:
            norm_props_is = None
            norm_props_oos = None
        # end if using pca transformation

        frame0_oos = self.create_frame_0(dat_raw_oos)

        # build in-sample and out-of-sample basis overlaps
        full_basis = self.basis_rows.shape[0] == self.frames.shape[0]
        if full_basis:
            overlaps_is = np.ones(self.frames.shape[0], dtype=np.float64)
        else:
            overlaps_is = self.build_overlaps()
        overlaps_oos = self.build_overlaps(rows=frame0_oos)

        if self.verbose and HAVE_PLT:
            # plot histograms of in-sample and out-of-sample overlaps
            if full_basis:
                print('NOTE: full basis, no in-sample non-basis rows to evaluate -- all in-sample overlaps are 1')
            plt.figure(figsize=(22, 8))
            ax1 = plt.subplot(1, 2, 1)
            ax2 = plt.subplot(1, 2, 2)
            label_font = {'size': 15}
            title_font = {'size': 17}
            num_bins = 50
            plt.axes(ax1)
            plt.hist(overlaps_is, bins=num_bins)
            plt.xlabel('basis overlap', fontdict=label_font)
            plt.ylabel('count', fontdict=label_font)
            plt.title('Basis Overlaps for In-Sample Non-Basis Rows', fontdict=title_font)
            plt.axes(ax2)
            plt.hist(overlaps_oos, bins=num_bins)
            plt.xlabel('basis overlap', fontdict=label_font)
            plt.ylabel('count', fontdict=label_font)
            plt.title('Basis Overlaps for Out-of-Sample Rows', fontdict=title_font)
            plt.show()
        # end if verbose

        # run dqm evolution of out-of-sample points, using in-sample operators and parameter values
        num_frames_to_build = self.frames.shape[2] - 1
        frames_oos = self.build_frames(num_frames_to_build, frames=frame0_oos)

        return frames_oos, overlaps_is, overlaps_oos, norm_props_is, norm_props_oos
    # end method run_new_points


    def run_simple(self, dat_raw, sigma):
        '''
        Do a simplified full DQM 'run', as follows -- given dat_raw (raw-data matrix) and a value of sigma:

        * Store raw data and sigma in the instance.
        * Create and store frame 0 (using all PCA dimensions)
        * Build and store operators (using a full basis and default value of mass)
        * Build and store frames (using build_frames_auto) until all points stop moving

        Note: default behaviors can be overridden by setting relevant instance parameters before calling this method.

        For small data sets, doing simple runs with various values of sigma can be the quickest way to understand
        the landscape that DQM is revealing.

        :param dat_raw: Raw data (2-D matrix).
        :param sigma: Positive value for sigma.
        :return: None
        '''

        assert type(dat_raw) is np.ndarray and dat_raw.ndim == 2,\
            "'dat_raw' must be 2-D ndarray"
        assert sigma > 0, "'sigma' must be positive"

        self.raw_data = dat_raw
        self.sigma = sigma

        self.create_frame_0()
        self.build_operators()
        self.build_frames_auto()
    # end method run_simple


    @classmethod
    def exists(cls, main_dir, sub_dir=None):
        '''
        Check whether main_dir contains a saved DQM instance. If sub_dir is not None, also check
        whether sub_dir contains saved DQM info.

        :param main_dir: Relative or absolute path to a folder. No default.
        :param sub_dir: Name of subdirectory (inside the 'main_dir' folder) for landscape-specific saved data.
            Default None.
        :return: Boolean: True if main_dir contains a saved DQM instance. (If sub_dir is not None,
            sub_dir must also have saved landscape-specific DQM info in order for us to return True.)
        '''

        member_path = os.path.join(main_dir, 'dqm_members')
        if not os.path.exists(member_path):
            return False
        elif sub_dir is None:
            return True
        else:
            sub_member_path = os.path.join(main_dir, sub_dir, 'dqm_members')
            return os.path.exists(sub_member_path)
    # end class method exists


    # 2FIX: IT'S BRITTLE THAT ANY CHANGES IN MEMBER VARIABLES MUST BE MADE MANUALLY HERE AS WELL
    # 2FIX: IF A MEMBER VARIABLE IS CLEARED (SET TO NONE) IN MEMORY AND THEN THE INSTANCE IS SAVED
    # TO DISK, AN OLD SAVED VERSION OF THAT MEMBER VARIABLE COULD STILL EXIST ON DISK, AND WOULD
    # THUS BE LOADED NEXT TIME, PUTTING THE INSTANCE IN AN INCONSISTENT STATE.  ADD LOGIC TO DELETE
    # FILES ON DISK IF THE MEMBER IS NONE.  (EXCEPT FOR RAW DATA, WHICH MIGHT NOT BE LOADED, JUST
    # FOR SPEED.  THINK THIS LOGIC THROUGH MORE CAREFULLY...)
    def save(self, main_dir, sub_dir=None):
        '''
        Save an instance of the DQM class:

        * Save numpy arrays separately.
        * Pickle everything else in the instance.

        Things that are common to multiple landscapes (raw data, PCA results) are saved in main_dir, which can
        be an absolute or relative path to a folder.

        Things that are specific to a given landscape (basis, DQM parameters, operators, evolved frames) are
        saved in sub_dir, which is relative to main_dir (so, typically sub_dir is just a folder name)

        Both main_dir and sub_dir (if not None) are created if they do not exist.

        :param main_dir: Relative or absolute path to a folder. No default.
        :param sub_dir: Name of subdirectory (inside the 'main_dir' folder) for basis-specific saved data.
            Default None.
        :return: None
        '''

        t0 = time()

        if not os.path.exists(main_dir):
            os.makedirs(main_dir)
        if sub_dir:
            sub_dir = os.path.join(main_dir, sub_dir)
            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir)

        ## things that are common to all landscapes (based on different bases, parameter values, and operators)
        if self.raw_data is not None:
            np.save(os.path.join(main_dir, 'raw_data.npy'), self.raw_data)
        if self.raw_col_means is not None:
            np.save(os.path.join(main_dir, 'raw_col_means.npy'), self.raw_col_means)
        if self.pca_eigvals is not None:
            np.save(os.path.join(main_dir, 'pca_eigvals.npy'), self.pca_eigvals)
        if self.pca_eigvecs is not None:
            np.save(os.path.join(main_dir, 'pca_eigvecs.npy'), self.pca_eigvecs)
        if self.pca_cum_var is not None:
            np.save(os.path.join(main_dir, 'pca_cum_var.npy'), self.pca_cum_var)
        if self.frames is not None:
            # save frame 0 only (still as 3-D array)
            np.save(os.path.join(main_dir, 'frame_0.npy'), self.frames[:, :, :1])
        members = {
            'pca_transform': self.pca_transform,
            'pca_num_dims': self.pca_num_dims,
            'pca_var_threshold': self.pca_var_threshold,
            'verbose': self.verbose,
            'min_report_time': self.min_report_time,
            'call_c': self.call_c,
            'mean_row_distance': self.mean_row_distance
        }
        with open(os.path.join(main_dir, 'dqm_members'), 'wb') as pickle_file:
            pickle.dump(members, pickle_file)

        ## things that are specific to a given landscape (basis, parameter values, and operators)
        if sub_dir:
            if self.basis_rows is not None:
                np.save(os.path.join(sub_dir, 'basis_rows.npy'), self.basis_rows)
            if self.simt is not None:
                np.save(os.path.join(sub_dir, 'simt.npy'), self.simt)
            if self.xops is not None:
                np.save(os.path.join(sub_dir, 'xops.npy'), self.xops)
            if self.exph is not None:
                np.save(os.path.join(sub_dir, 'exph.npy'), self.exph)
            if self.frames is not None:
                np.save(os.path.join(sub_dir, 'frames.npy'), self.frames)
            members = {
                'basis_num_chunks': self.basis_num_chunks,
                'basis_rand_seed': self.basis_rand_seed,
                'basis_row_nums': self.basis_row_nums,
                'non_basis_row_nums': self.non_basis_row_nums,
                'basis_size': self.basis_size,
                'basis_start_with_outlier': self.basis_start_with_outlier,
                'sigma': self.sigma,
                'step': self.step,
                'mass': self.mass,
                'overlap_mean_threshold': self.overlap_mean_threshold,
                'overlap_min_threshold': self.overlap_min_threshold,
                'stopping_threshold': self.stopping_threshold
            }
            with open(os.path.join(sub_dir, 'dqm_members'), 'wb') as pickle_file:
                pickle.dump(members, pickle_file)
        # end if sub_dir is not None

        t1 = time()
        if self.verbose and t1 - t0 >= self.min_report_time:
            print("saved dqm instance in {} seconds".format(round(t1 - t0)))
    # end method save


    # 2FIX: IT'S BRITTLE THAT ANY CHANGES IN MEMBER VARIABLES MUST BE MADE MANUALLY HERE AS WELL
    @classmethod
    def load(cls, main_dir, sub_dir=None, load_raw_data=True, verbose=True):
        '''
        Load an instance of the DQM class from disk and return it.

        :param main_dir: Relative or absolute path to folder. No default.
        :param sub_dir: Name of subdirectory (inside the 'main_dir' folder) for landscape-specific saved data.
            Default None.
        :param load_raw_data: Boolean: if True, we load raw data. Set to False to save time if raw
            data is very large. Default True.
        :param verbose: Boolean: whether to report on various operations. Default True.
        :return: a DQM instance with data loaded from 'main_dir' (and from sub_dir, if not None).
        '''

        t0 = time()

        assert cls.exists(main_dir), f"dir '{main_dir}' must be a saved dqm instance"
        if sub_dir:
            sub_dir_name = sub_dir
            sub_dir = os.path.join(main_dir, sub_dir)
            assert os.path.exists(os.path.join(sub_dir, 'dqm_members')), \
                f"sub dir '{sub_dir_name}' must exist and have saved dqm data"
        # end if sub_dir is not None

        dqm_obj = DQM()

        ## things that are common to all landscapes (raw data and PCA info)
        if load_raw_data:
            pth = os.path.join(main_dir, 'raw_data.npy')
            if os.path.exists(pth):
                dqm_obj.raw_data = np.load(pth, allow_pickle=True)
        pth = os.path.join(main_dir, 'raw_col_means.npy')
        if os.path.exists(pth):
            dqm_obj.raw_col_means = np.load(pth, allow_pickle=True)
        pth = os.path.join(main_dir, 'pca_eigvals.npy')
        if os.path.exists(pth):
            dqm_obj.pca_eigvals = np.load(pth, allow_pickle=True)
        pth = os.path.join(main_dir, 'pca_eigvecs.npy')
        if os.path.exists(pth):
            dqm_obj.pca_eigvecs = np.load(pth, allow_pickle=True)
        pth = os.path.join(main_dir, 'pca_cum_var.npy')
        if os.path.exists(pth):
            dqm_obj.pca_cum_var = np.load(pth, allow_pickle=True)
        pth = os.path.join(main_dir, 'frame_0.npy')
        if os.path.exists(pth):
            dqm_obj.frames = np.load(pth, allow_pickle=True)
        pth = os.path.join(main_dir, 'dqm_members')
        if os.path.exists(pth):
            with open(pth, 'rb') as pickle_file:
                members = pickle.load(pickle_file)
                dqm_obj.pca_transform = members['pca_transform']
                dqm_obj.pca_num_dims = members['pca_num_dims']
                dqm_obj.pca_var_threshold = members['pca_var_threshold']
                dqm_obj.verbose = members['verbose']
                dqm_obj.min_report_time = members['min_report_time']
                dqm_obj.call_c =  members['call_c']
                dqm_obj.mean_row_distance = members['mean_row_distance']
            # end with pickle file
        # end if members saved

        ## things that are specific to a given landscape (basis, parameter values, operators, evolved frames)
        if sub_dir:
            pth = os.path.join(sub_dir, 'basis_rows.npy')
            if os.path.exists(pth):
                dqm_obj.basis_rows = np.load(pth, allow_pickle=True)
            pth = os.path.join(sub_dir, 'simt.npy')
            if os.path.exists(pth):
                dqm_obj.simt = np.load(pth, allow_pickle=True)
            pth = os.path.join(sub_dir, 'xops.npy')
            if os.path.exists(pth):
                dqm_obj.xops = np.load(pth, allow_pickle=True)
            pth = os.path.join(sub_dir, 'exph.npy')
            if os.path.exists(pth):
                dqm_obj.exph = np.load(pth, allow_pickle=True)
            pth = os.path.join(sub_dir, 'frames.npy')
            if os.path.exists(pth):
                dqm_obj.frames = np.load(pth, allow_pickle=True)
            pth = os.path.join(sub_dir, 'dqm_members')
            if os.path.exists(pth):
                with open(pth, 'rb') as pickle_file:
                    members = pickle.load(pickle_file)
                    dqm_obj.basis_num_chunks = members['basis_num_chunks']
                    dqm_obj.basis_rand_seed = members['basis_rand_seed']
                    dqm_obj.basis_row_nums = members['basis_row_nums']
                    dqm_obj.non_basis_row_nums = members['non_basis_row_nums']
                    dqm_obj.basis_size = members['basis_size']
                    dqm_obj.basis_start_with_outlier = members['basis_start_with_outlier']
                    dqm_obj.sigma = members['sigma']
                    dqm_obj.step = members['step']
                    dqm_obj.mass = members['mass']
                    dqm_obj.overlap_mean_threshold = members['overlap_mean_threshold']
                    dqm_obj.overlap_min_threshold = members['overlap_min_threshold']
                    dqm_obj.stopping_threshold = members['stopping_threshold']
                # end with pickle file
            # end if members saved
        # end if sub_dir is not None

        t1 = time()
        if verbose and t1 - t0 >= cls.min_report_time:
            print("loaded dqm instance in {} seconds".format(round(t1 - t0)))

        return dqm_obj
    # end class method load

# end class DQM

