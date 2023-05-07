import numpy as np

from . import dqm_lib  # compiled-function library

try:
    import plotly.graph_objects as go
    HAVE_PTY = True
except ModuleNotFoundError:
    HAVE_PTY = False

try:
    import ipyvolume as ipv
    HAVE_IPV = True
except ModuleNotFoundError:
    HAVE_IPV = False


def pca(mat, verbose=True):
    '''
    Given a matrix, compute and return the PCA eigenvalues and associated eigenvectors (principal components).

    :param mat: A 2-D real-valued matrix.
    :param verbose: Boolean: whether to report on various operations. Default True.
    :return: A tuple of:

        * Vector of eigenvalues, in descending order of magnitude (all are non-negative).
        * Matrix with corresponding eigenvectors (principal components) in the columns.
    '''

    '''
    Handle 2 cases differently:
    * If 'mat' has more rows than columns, compute the covariance matrix for the columns and find the
      eigenvectors of the covariance matrix, which are the principal components.
    * If 'mat' has more columns than rows, compute the covariance matrix for the rows, and find the
      eigenvectors for that covariance matrix. These eigenvectors are the left singular vectors (U) of
        the SVD decomposition of 'mat' (after centering). Then use the definition of SVD (M = U @ S @ VT)
        to compute VT, whose rows are the right singular vectors, which are the PCA eigenvectors that we
        want. (In this case, for numerical stability, we drop eigenvalues that are less than 1e-10 times
        the largest eigenvalue. We set these too-small eigenvalues and their associated eigenvectors equal
        to zero.)

    Treating the 2 cases above differently allows us to handle matrices with a large number of rows or
    a large number of columns more efficiently. This approach can handle matrices where numpy.linalg.svd
    quickly runs out of memory. (Matrices with a large number of rows *and* a large number of columns
    will still be very memory-intensive.)

    In both cases, we can use numpy.linalg.eigh instead of numpy.linalg.eig because we know the covariance
    matrix is symmetric. (eigh is faster and guarantees the eigenvalues are returned in ascending order of
    magnitude.)
    '''

    assert type(mat) is np.ndarray and mat.ndim == 2 and mat.dtype == np.float64, \
        "'mat' must be a 2-D float-64 ndarray"

    num_rows, num_cols = mat.shape

    if verbose and min(num_rows, num_cols) > 1e4:
        print('### WARNING: running PCA for large matrices may run out of memory')

    # subtract mean from each column
    mat = mat - np.mean(mat, axis=0)

    if num_rows >= num_cols:
        # compute covariance matrix for the columns
        mat_cov = mat.T @ mat / (num_rows - 1)

        # do eigen decomposition
        eigvals, eigvecs = np.linalg.eigh(mat_cov)
    else:
        # compute covariance matrix for the rows
        # (note: for consistency when matching eigenvalues with singular values from SVD, we still
        # use num_rows in the denominator here, not num_cols)
        mat_cov = mat @ mat.T / (num_rows - 1)

        # do eigen decomposition
        eigvals, eigvecs = np.linalg.eigh(mat_cov)

        # eigvecs gives us U, the left singular vectors, in the SVD decomposition of mat.  using the
        # definition of SVD, recover the eigenvectors we actually want, which are the right singular
        # vectors, which are the rows of VT:
        #   M = U @ S @ VT, which gives
        #   VT = S_R @ UT @ M (where UT is the transpose of U, and S_R is the diagonal matrix with the
        #   reciprocals of the singular values)
        s_r = np.zeros((eigvals.size, eigvals.size))
        min_eigval = 1e-10 * eigvals[-1]  # largest eigenvalue is last
        for i in range(eigvals.size):
            if eigvals[i] >= min_eigval:
                s_r[i, i] = (eigvals[i] * (num_rows - 1)) ** -0.5
        vt = s_r @ eigvecs.T @ mat
        eigvecs = vt.T
    # end if/else more rows or more columns

    # set any non-positive eigenvalues to zero, along with the corresponding eigenvectors
    nonpos_eigval_idxs = np.where(eigvals <= 0)[0]
    nonpos_n = nonpos_eigval_idxs.size
    if nonpos_n > 0:
        assert nonpos_eigval_idxs.tolist() == list(range(nonpos_n)), \
            "non-positive eigenvalues must come first in ascending sorting of eigenvalues"
        if verbose:
            print('### WARNING: PCA has {} negative eigenvalue{} (due to noise at the machine-precision level):'.\
                  format(nonpos_n, 's' if nonpos_n > 1 else ''))
            print('setting those eigvenvalues and associated eigenvectors to zero ###')
        eigvals[nonpos_eigval_idxs] = 0
        eigvecs[:, nonpos_eigval_idxs] = 0
    # end if any negative eigenvalues

    # set any unnormalized eigenvectors to zero, along with the corresponding eigenvalues
    eigvec_norms = np.linalg.norm(eigvecs, axis=0)
    bad_vec_idxs = np.where(abs(eigvec_norms - 1) > 1e-6)[0]
    bad_n = bad_vec_idxs.size
    if bad_n > 0:
        assert bad_vec_idxs.tolist() == list(range(bad_n)), \
            'unnormalized eigenvectors must come first in ascending sorting of eigenvalues'
        if verbose:
            print('### WARNING: PCA has {} unnormalized eigenvector{} (due to noise at the machine-precision level):'.\
                  format(bad_n, 's' if bad_n > 1 else ''))
            print('setting those eigvenvectors and associated eigenvalues to zero ###')
        eigvals[bad_vec_idxs] = 0
        eigvecs[:, bad_vec_idxs] = 0
    # end if any unnormalized eigenvectors

    # return eigenvalues and associated eigenvectors in descending order of eigenvalue magnitude
    eigvals = np.flip(eigvals)
    eigvecs = np.flip(eigvecs, axis=1)

    return eigvals, eigvecs
# end function pca


def get_clusters(mat, max_dist):
    '''
    Get clusters from a matrix.

    We here define a cluster as a group of rows such that every row is within max_dist Euclidean distance
    of at least one other row in the cluster. The largest distance between 2 rows in a cluster may be
    much larger than max_dist. ('Cluster' is defined here in a way that will include extended structures.)

    :param mat: A 2-D real-valued matrix. No default.
    :param max_dist: A row being within max_dist Euclidean distance of any other row in a cluster
        will make that row part of that same cluster. No default.
    :return: A tuple of:

        * a list of lists of row numbers for the clusters. the list of clusters is sorted by cluster
          size (number of rows), in descending order.
        * a parallel list of cluster sizes
    '''

    '''
    This function calls compiled (C++) code. This function is slower than the python-only version in
    some cases, but it's generally much faster for large numbers of rows, meaning more than 100k or so
    (except maybe in certain corner cases).
    '''

    assert type(mat) is np.ndarray and mat.ndim == 2, "'mat' must be a 2-dimensional ndarray"
    assert max_dist > 0, "'max_dist' must be positive"

    num_rows, num_cols = mat.shape

    clusters = []

    man_idxs = np.ndarray(num_rows, dtype=np.int32)

    if not mat.flags['C_CONTIGUOUS']:
        mat = np.copy(mat)

    if dqm_lib is None:
        print('## WARNING: in get_clusters -- compiled-library code not found, calling Python code...')
        clusters, cluster_sizes = _get_clusters_python(mat, max_dist)
        return clusters, cluster_sizes
    # end if don't have compiled-library code

    dqm_lib.GetClustersC(mat, num_rows, num_cols, max_dist, man_idxs)

    # get clusters from man_idxs (see GetClustersC for details)
    # notes
    #  * row indices in man_idxs are a linked list for each cluster
    #  * each entry in man_idxs points to the next row in that cluster
    #  * a value of num_rows in man_idxs means the end of a given linked list (either the given row is its
    #    own cluster, or it's the last row in a cluster)
    #  * the lists should always point forward, meaning we should always have man_idxs[row_idx] > row_idx
    #  * as we add rows to the clusters we're constructing, we flag them with a -1 value and ignore them
    #    from then on
    assert np.min(man_idxs) >= 0, "'man_idxs' must not have any negative values"
    for row_idx in range(num_rows):
        if man_idxs[row_idx] == -1:
            continue  # this row has already been added to a cluster

        cluster = [row_idx]
        idx = row_idx
        while man_idxs[idx] < num_rows:
            assert man_idxs[idx] > idx, 'linked lists should always point forward in the overall list'
            new_idx = man_idxs[idx]
            man_idxs[idx] = -1
            idx = new_idx
            cluster.append(idx)
        # end while following linked list for current cluster
        man_idxs[idx] = -1

        clusters.append(cluster)
    # end for each row

    # sort clusters by size, descending
    clusters.sort(reverse=True, key=lambda x: len(x))

    cluster_sizes = [len(cluster) for cluster in clusters]

    return clusters, cluster_sizes
# end function get_clusters


def _get_clusters_python(mat, max_dist):
    '''
    Get clusters from a matrix.

    We here define a cluster as a group of rows such that every row is within max_dist Euclidean distance
    of at least one other row in the cluster. The largest distance between 2 rows in a cluster may be
    much larger than max_dist. ('Cluster' is defined here in a way that will include extended structures.)

    This is the version of get_clusters that does *not* call the compiled C++ code. This python-only
    version is faster in some cases, but it's unusably slow for large numbers of rows, meaning more than 100k
    or so (except maybe in certain corner cases).

    :param mat: A 2-D real-valued matrix. No default.
    :param max_dist: A row being within max_dist Euclidean distance of any other row in a cluster
        will make that row part of that same cluster. No default.
    :return: A tuple of:

        * a list of lists of row numbers for the cluster. the list of clusters is sorted by cluster
          size (number of rows), in descending order.
        * a parallel list of cluster sizes
    '''

    assert type(mat) is np.ndarray and mat.ndim == 2, "'mat' must be a 2-dimensional ndarray"
    assert max_dist > 0, "'max_dist' must be positive"

    num_rows = mat.shape[0]

    clusters = []

    row_nums_left = list(range(num_rows))

    while np.any(row_nums_left):
        seed_row_num = row_nums_left[0]
        row_nums_left = row_nums_left[1:]
        new_cluster = _get_cluster_python(mat, max_dist, seed_row_num, row_nums_left)
        clusters.append(new_cluster)
        row_nums_left = list(set(row_nums_left).difference(set(new_cluster)))
    # end while any rows left

    # sort clusters by size, descending
    clusters.sort(reverse=True, key=lambda x: len(x))

    cluster_sizes = [len(cluster) for cluster in clusters]

    return clusters, cluster_sizes
# end function _get_clusters_python


def _get_cluster_python(mat, max_dist, seed_row_num, row_nums=None):
    '''
    Get a single cluster from a matrix. (See comments for _get_clusters_python for details.)

    :param mat: A 2-D real-valued matrix. No default.
    :param max_dist: A row being within max_dist Euclidean distance of any other row in a cluster
        will make that row part of that same cluster. No default.
    :param seed_row_num: Row number for the seed row for the cluster.
    :param row_nums: List of row numbers to consider (may or may not contain seed_row_num). If None, we use
        all row numbers. Default None.
    :return: List of row numbers for the cluster, starting with seed_row_num.
    '''

    assert type(mat) is np.ndarray and mat.ndim == 2, "'mat' must be a 2-dimensional ndarray"

    num_rows = mat.shape[0]
    assert 0 <= seed_row_num < num_rows, "'seed_row_num' must be a valid row number"

    assert max_dist > 0, "'max_dist' must be positive"

    # set up list of candidate row numbers
    if row_nums is None:
        row_nums = list(range(num_rows))
    cand_row_nums = np.array(list(set(row_nums).difference({seed_row_num})))

    man_row_nums = np.array([seed_row_num])  # row numbers in the cluster
    seed_row_nums = np.array([seed_row_num])  # current seed row numbers

    num_cols = mat.shape[1]

    # continue while there are both seed row numbers and candidate row numbers
    while seed_row_nums.size > 0  and cand_row_nums.size > 0:
        # find the distance from each candidate row to each seed row
        # (reshape matrices to enable broadcasting)
        c = mat[cand_row_nums, :]
        c = c.reshape((c.shape[0], 1, num_cols))
        s = mat[seed_row_nums, :]
        s = s.reshape((1, s.shape[0], num_cols))
        diff_vecs = c - s
        dists = np.linalg.norm(diff_vecs, axis=2)

        # find the minimum distance to any seed row for each candidate row
        min_dists = np.min(dists, axis=1)

        # select the candidate rows that are close
        close_row_nums = cand_row_nums[min_dists <= max_dist]

        # rows that are not close remain candidates
        cand_row_nums = cand_row_nums[min_dists > max_dist]

        # add close rows to cluster
        man_row_nums = np.concatenate((man_row_nums, close_row_nums))

        # set up new seeds
        seed_row_nums = close_row_nums
    # end while seeds and candidates left

    return man_row_nums.tolist()
# end function _get_cluster_python


def nearest_neighbors(mat):
    '''
    Return nearest-neighbor row number and distance for each row in mat.

    :param mat: A 2-D real-valued matrix. No default.
    :return: A tuple of:

        * Vector of nearest-neighbor row number for each row.
        * Vector of nearest-neighbor distance for each row.
    '''

    assert type(mat) is np.ndarray and mat.ndim == 2, "'mat' must be a 2-dimensional array"

    num_rows, num_cols = mat.shape

    nn_row_nums = np.zeros(num_rows, dtype=np.int32)
    nn_dists = np.zeros(num_rows, dtype=np.float64)
    if dqm_lib is not None:
        dqm_lib.NearestNeighborsC(mat, num_rows, num_cols, nn_row_nums, nn_dists)
    else:
        print('## WARNING: in nearest_neighbors -- compiled-library code not found, calling Python code...')
        nn_row_nums, nn_dists = _nearest_neighbors_python(mat)
    # end if /else have compiled-library code or not

    return nn_row_nums, nn_dists
# end function nearest_neighbors


def _nearest_neighbors_python(mat):
    '''
    Return nearest-neighbor row number and distance for each row in mat.

    This is the version of nearest_neighbors that does *not* call the compiled C++ code. It will generally be
    much slower.

    :param mat: A 2-D real-valued matrix. No default.
    :return: A tuple of:

        * Vector of nearest-neighbor row number for each row.
        * Vector of nearest-neighbor distance for each row.
    '''

    assert type(mat) is np.ndarray and mat.ndim == 2, "'mat' must be a 2-dimensional array"

    num_rows, num_cols = mat.shape

    nn_row_nums = np.zeros(num_rows, dtype=np.int64)
    nn_dists = np.zeros(num_rows, dtype=np.float64)

    for row_idx in range(num_rows):
        distances = np.linalg.norm(mat - mat[row_idx, :], axis=1)
        distances[row_idx] = np.max(distances)  # don't choose row as its own nearest neighbor
        min_idx = np.argmin(distances)
        nn_row_nums[row_idx] = min_idx
        nn_dists[row_idx] = distances[min_idx]
    # end for each row

    return nn_row_nums, nn_dists
# end function _nearest_neighbors_python


def plot_frames(frames, color='blue', size=5, skip_frames=1, fps=10, title='', labels=['X', 'Y', 'Z'],
                width=800, height=800, show_as_cube=True, show_gridlines=True, show_ticklabels=True):
    '''
    Display interactive animated 3-D plot of a set of frames, using Plotly.

    WARNING -- WHEN PLOTTING LARGE NUMBERS OF POINTS AND/OR FRAMES:

        * File sizes can get very big for saved plots. (This may be a problem if, for example, you're working
          in a Jupyter notebook with autosave turned on.)
        * Specifying different colors and/or different sizes for each frame will make rendering take *much* longer.

    :param frames: 2-D or 3-D array of frames. If there are more than 3 columns in 2nd dim, we use the first
        3 columns for plotting. Animation is only enabled if there are multiple frames in 3rd dim. No default.
    :param color: Color information for points -- can be:

        * String (must be a color understood by Plotly), applied to all points. Default 'blue'.
        * Array/list/tuple of length 3 (RGB) or 4 (RGBA), with float values in [0, 1], applied to all points.
        * 2-D array, <num_rows x 3 (RGB) or 4 (RGBA)>, with float values in [0, 1], applied to each point in
          all frames.
        * 3-D array, <num_rows x 3 (RGB) or 4 (RGBA) x num_frames>, with float values in [0, 1], applied to each
          point in each frame. (If using skip_frames > 1, frames and colors should line up as passed in -- both
          will be subselected.)

    :param size: Size information for points -- can be:

        * Scalar, applied to all points. default 5.
        * 1-D array, <num_rows>, applied to each point in all frames.
        * 2-D array, <num_rows x num_frames>, applied to each point in each frame. (If using skip_frames > 1,
          frames and sizes should line up as passed in -- both will be subselected.)

    :param skip_frames: Positive integer: if > 1, plot every nth frame (so, for a value of 2, plot every other
        frame). Default 1. (This feature is useful because plotting a large number of frames can be slow and
        memory-intensive.)
    :param fps: Frames per second. Default 10. (Plotly may not be able to handle a frame rate much larger than
        10 or 15 fps. You can use skip_frames to help with this.)
    :param title: Title for the plot. Default "".
    :param labels: Array/list/tuple of axis labels, or None. Default ['X', 'Y', 'Z'].
    :param width: Figure width in pixels. Default 800.
    :param height: Figure height in pixels. Default 800.
    :param show_as_cube: Boolean: if True show plot as a cube, if False show plot with correct axis-range
        proportions. Default True.
    :param show_gridlines: Boolean: whether to show gridlines. Default True.
    :param show_ticklabels: Boolean: whether to show axis tick labels. Default True.
    :return: None
    '''

    '''
    2FIX: add wiring to enable use of Plotly's built-in color maps?
    '''

    assert HAVE_PTY, "must have loaded plotly.graph_objects module to use plot_frames"

    assert type(frames) is np.ndarray and frames.ndim in [2, 3] and frames.shape[1] >= 3, \
        "'frames' must be a 2-D or 3-D ndarray with at least 3 columns in 2nd dimension"
    if frames.ndim == 2:
        frames = frames[:, :, np.newaxis]

    assert skip_frames > 0 and round(skip_frames) == skip_frames, "'skip_frames' must be a positive integer"

    if labels is None:
        labels = ['', '', '']
    assert len(labels) == 3, "'labels' must have 3 elements"

    num_rows = frames.shape[0]

    ## deal with color info
    assert type(color) is str or len(color) in [3, 4] or type(color) is np.ndarray, \
        "'color' must be a string, a list/array/tuple of length 3 or 4, or an ndarray"
    if type(color) is str:
        pass
    elif len(color) in [3, 4]:
        # single color for all points and all frames
        color = np.array(color)
        # plotly handles 2-D color well, but not 1-D -- so, replicate this color in 1st dimension:
        color = np.repeat(color[np.newaxis, :], num_rows, axis=0)
    # end switch
    if type(color) is np.ndarray:
        assert color.shape[0] == num_rows, "'color' array must have same number of rows as 'frames'"
        assert color.shape[1] in [3, 4],\
            "each color specified in 'color' array must have 3 values (RGB) or 4 values (RGBA)"
        assert color.ndim in [2, 3], "'color' array must be 2-D or 3-D"
    # end if color is ndarray
    animating_colors = type(color) is np.ndarray and color.ndim == 3

    ## deal with size info
    if type(size) is np.ndarray:
        assert size.shape[0] == num_rows, "'size' array must have same number of rows as 'frames'"
        assert size.ndim in [1, 2], "'size' array must be 1-D or 2-D"
    # end if size is ndarray
    animating_sizes = type(size) is np.ndarray and size.ndim == 2

    if skip_frames > 1:
        real_frame_idxs = list(range(0, frames.shape[2], skip_frames))
        frames = frames[:, :, real_frame_idxs]
        if animating_colors:
            color = color[:, :, real_frame_idxs]
        if animating_sizes:
            size = size[:, real_frame_idxs]
    else:
        real_frame_idxs = list(range(frames.shape[2]))
    # end if skipping frames
    num_frames = frames.shape[2]
    animate = num_frames > 1

    x_all = frames[:, 0, :]
    y_all = frames[:, 1, :]
    z_all = frames[:, 2, :]

    x_min, x_max = np.min(x_all), np.max(x_all)
    y_min, y_max = np.min(y_all), np.max(y_all)
    z_min, z_max = np.min(z_all), np.max(z_all)

    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min
    max_range = max(x_range, y_range, z_range)

    if show_as_cube:
        aspect_mode = dict(aspectmode='cube')
    else:
        aspect_mode = dict(aspectmode='manual',
                           aspectratio=dict(x=x_range / max_range, y=y_range / max_range, z=z_range / max_range))

    gridcolor = 'gray' if show_gridlines else 'white'

    plot_layout = {
        'scene': dict(xaxis=dict(range=[x_min, x_max], autorange=False, backgroundcolor='white', gridcolor=gridcolor,
                                 showticklabels=show_ticklabels, title=labels[0]),
                      yaxis=dict(range=[y_min, y_max], autorange=False, backgroundcolor='white', gridcolor=gridcolor,
                                 showticklabels=show_ticklabels, title=labels[1]),
                      zaxis=dict(range=[z_min, z_max], autorange=False, backgroundcolor='white', gridcolor=gridcolor,
                                 showticklabels=show_ticklabels, title=labels[2]),
                      bgcolor='white',
                      **aspect_mode
                      ),
        'width': width,
        'height': height
        }

    if not animate:
        # only one frame
        fig = go.Figure(go.Scatter3d(x=x_all[:, 0], y=y_all[:, 0], z=z_all[:, 0],
                                     mode="markers",
                                     marker=dict(color=color, size=size)
                                     ),
                        layout=go.Layout(**plot_layout)
                        )
    else:
        # create plot frames
        go_frames = []
        for frame_idx in range(num_frames):
            marker_dict = {}
            if animating_colors:
                marker_dict['color'] = color[:, :, frame_idx]
            if animating_sizes:
                marker_dict['size'] = size[:, frame_idx]
            go_frames.append(
                go.Frame(data=[go.Scatter3d(x=x_all[:, frame_idx], y=y_all[:, frame_idx], z=z_all[:, frame_idx],
                                            mode="markers",
                                            marker=marker_dict
                                            )
                               ],
                         traces=[0],
                         name=f'frame{frame_idx}'
                         )
                )
        # end for creating each frame

        def frame_args(duration):
            return {
                "frame": {"duration": duration},
                "mode": "immediate",
                "fromcurrent": True,
                "duration": duration,
                "transition": {"duration": 0}
            }
        # end function frame_args

        sliders = [
            {"pad": {"b": 10, "t": 60},
             "len": 0.9,
             "x": 0.1,
             "y": 0,
             "currentvalue": {"prefix": "Frame "},
             "steps": [
                 {"args": [[f.name], frame_args(0)],
                  "label": '{:,}'.format(real_frame_idxs[k]),
                  "method": "animate"
                  } for k, f in enumerate(go_frames)
             ]
             }
        ]

        # set up initial scatter
        first_scatter = go_frames[0].data[0]
        if not animating_colors or not animating_sizes:
            # if not animating size or color, include them here to be inherited by frames
            # (note: this makes rendering *much* faster for large numbers of points and/or frames)
            first_scatter.mode = 'markers'
            if not animating_colors:
                first_scatter.marker['color'] = color
            if not animating_sizes:
                first_scatter.marker['size'] = size
        # end if including color and/or size in initial scatter

        fig = go.Figure(first_scatter, frames=go_frames, layout=go.Layout(**plot_layout))

        fig.update_layout(
            updatemenus=[{"buttons": [
                {
                    "args": [None, frame_args(1000 / fps)],
                    "label": "Play",
                    "method": "animate"
                },
                {
                    "args": [[None], frame_args(0)],
                    "label": "Pause",
                    "method": "animate"
                }],
                "direction": "left",
                "pad": {"r": 10, "t": 70},
                "type": "buttons",
                "x": 0.1,
                "y": 0
            }
            ],
            sliders=sliders
        )
    # end if animating multiple frames or not

    if title is not None and len(title) > 0:
        # add figure title
        fig.update_layout(
            title={
                'text': title,
                'y': 0.94,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'})

    fig.show()
# end function plot_frames


def plot_frames_ipv(frames, color='blue', size=4, skip_frames=1, fps=10, labels=['X', 'Y', 'Z'], show_axes=True):
    '''
    Display interactive animated 3-D plot of a set of frames, using IPyVolume.

    WARNING: IPyVolume is not a fully mature/stable package -- it may be buggy. (However, it may handle large
    numbers of data points and/or frames much better than Plotly will in plot_frames.)

    :param frames: 2-D or 3-D array of frames. If there are more than 3 columns in 2nd dim, we use the first
        3 columns for plotting. Animation is only enabled if there are multiple frames in 3rd dim. No default.
    :param color: Color information for points -- can be:

        * String (must be a color understood by Plotly), applied to all points. Default 'blue'.
        * Array/list/tuple of length 3 (RGB) or 4 (RGBA), with float values in [0, 1], applied to all points.
        * 2-D array, <num_rows x 3 (RGB) or 4 (RGBA)>, with float values in [0, 1], applied to each point in
          all frames.
        * 3-D array, <num_rows x 3 (RGB) or 4 (RGBA) x num_frames>, with float values in [0, 1], applied to each
          point in each frame. (If using skip_frames > 1, frames and colors should line up as passed in -- both
          will be subselected.)
    :param size: Size information for points -- can be:

        * Scalar, applied to all points. default 4.
        * 1-D array, <num_rows>, applied to each point in all frames.
        * 2-D array, <num_rows x num_frames>, applied to each point in each frame. (If using skip_frames > 1,
          frames and sizes should line up as passed in -- both will be subselected.)
    :param skip_frames: Positive integer: if > 1, plot every nth frame (so, for a value of 2, plot every other
        frame). Default 1.
    :param fps: Frames per second. default 10.
    :param labels: Array/list/tuple of axis labels. default ['X', 'Y', 'Z'].
    :param show_axes: Boolean: whether to show axes (axis label and tick labels). 'labels' is ignored if this is
        False. default True.
    :return: None
    '''

    assert HAVE_IPV, "must have loaded ipyvolume module to use plot_frames_ipv"

    assert type(frames) is np.ndarray and frames.ndim in [2, 3] and frames.shape[1] >= 3, \
        "'frames' must be a 2-D or 3-D ndarray with at least 3 columns in 2nd dimension"
    if frames.ndim == 2:
        frames = frames[:, :, np.newaxis]

    assert skip_frames > 0 and round(skip_frames) == skip_frames, "'skip_frames' must be a positive integer"

    num_rows = frames.shape[0]

    ## deal with color info
    assert type(color) is str or len(color) in [3, 4] or type(color) is np.ndarray, \
        "'color' must be a string, a list/array/tuple of length 3 or 4, or an ndarray"
    if type(color) is str:
        pass
    elif len(color) in [3, 4]:
        # single color for all points and all frames
        color = np.array(color)
        # ipyvolume handles 2-D color well, but not 1-D -- so, replicate this color in 1st dimension:
        color = np.repeat(color[np.newaxis, :], num_rows, axis=0)
    elif color.ndim == 3:
        color = color.transpose((2, 0, 1))  # ipyvolume wants <num frames x num rows x RBG(A)> for colors
    # end switch
    if type(color) is np.ndarray:
        assert color.ndim == 1 or color.shape[-2] == num_rows, \
            "'color' array must have same number of rows as 'frames'"
        assert color.shape[-1] in [3, 4], \
            "each color specified in 'color' array must have 3 values (RGB) or 4 values (RGBA)"
    # end if color is ndarray
    animating_colors = type(color) is np.ndarray and color.ndim == 3

    ## deal with size info
    if type(size) is np.ndarray:
        # unless size is a scalar, ipyvolume expects it as <num_frames x num_points> or as a 1-D <num_points> vector
        if size.ndim == 2:
            size = size.transpose()  ## put frames in 1st dim
        assert size.shape[-1] == num_rows, "'size' must be <num_points> or <num_points x num_frames> array"
    # end if size is ndarray
    animating_sizes = type(size) is np.ndarray and size.ndim == 2

    if skip_frames > 1:
        real_frame_idxs = list(range(0, frames.shape[2], skip_frames))
        frames = frames[:, :, real_frame_idxs]
        if animating_colors:
            color = color[real_frame_idxs, :, :]
        if animating_sizes:
            size = size[real_frame_idxs, :]
    else:
        real_frame_idxs = list(range(frames.shape[2]))
    # end if skipping frames
    num_frames = frames.shape[2]
    animate = num_frames > 1

    x = frames[:, 0, :].T
    y = frames[:, 1, :].T
    z = frames[:, 2, :].T

    ipv.figure()
    ipv.style.box_on()

    sctr = ipv.scatter(x, y, z, marker='sphere', size=size, color=color)
    ipv.xlim(np.min(x), np.max(x))
    ipv.ylim(np.min(y), np.max(y))
    ipv.zlim(np.min(z), np.max(z))

    if show_axes:
        assert len(labels) == 3, "'labels' must have 3 elements"
        label_lengths = [len(lbl) for lbl in labels]
        assert min(label_lengths) > 0, 'ipyvolume does not support empty axis labels'
        ipv.xlabel(labels[0])
        ipv.ylabel(labels[1])
        ipv.zlabel(labels[2])
        ipv.style.axes_on()
    else:
        ipv.style.axes_off()
    # end if showing axes or not

    if animate:
        ipv.animation_control(sctr, interval=1000 / fps)

    ipv.show()
# end function plot_frames_ipv


def cat_frames(frames1, frames2=None):
    '''
    Given multiple sets of frames, concatenate them in 1st dim.

    All sets of frames must have the same number of columns in 2nd dim.

    If the numbers of frames in 3rd dim don't match in each set of frames, we replicate the last frame in each
    set as needed.

    :param frames1: 3-D array of frames, or list of 3-D arrays of frames. If it's a list, we cat together
        all sets of frames in the list and return the result. No default.
    :param frames2: 3-D ndarray of frames. Must be None if frames1 is a list. Default None.
    :return: New combined 3-D array of frames.
    '''

    if type(frames1) is list:
        assert len(frames1) > 1, "list of frames arrays must have multiple elements to cat together"
        assert frames2 is None, "'frames2' must be None if 'frames1' is a list of frames"
        frames = frames1[0]
        for idx in range(1, len(frames1)):
            frames = cat_frames(frames, frames1[idx])
        return frames
    # end if frames1 is a list

    assert type(frames1) is np.ndarray and frames1.ndim == 3, "'frames1' must be a 3-D ndarray"
    assert type(frames2) is np.ndarray and frames2.ndim == 3, "'frames2' must be a 3-D ndarray"
    assert frames1.shape[1] == frames2.shape[1], \
        "'frames1' and 'frames2' must have the same number of columns in dim 1"

    num_frames1 = frames1.shape[2]
    num_frames2 = frames2.shape[2]

    if num_frames1 > num_frames2:
        frames2_new = np.zeros((frames2.shape[0], frames2.shape[1], num_frames1))
        frames2_new[:, :, :num_frames2] = frames2
        frames2_new[:, :, num_frames2:] = frames2[:, :, -1:]
        frames2 = frames2_new
    elif num_frames1 < num_frames2:
        frames1_new = np.zeros((frames1.shape[0], frames1.shape[1], num_frames2))
        frames1_new[:, :, :num_frames1] = frames1
        frames1_new[:, :, num_frames1:] = frames1[:, :, -1:]
        frames1 = frames1_new

    return np.concatenate((frames1, frames2), axis=0)
# end function cat_frames


def add_bookend_frames(frames, num_bookend_frames):
    '''
    Add 'bookend' frames at beginning and end of a set of frames -- bookend frames are just duplicates of the
    first and last frames. (Bookend frames can make an animation playing on repeat much easier to follow.)

    :param frames: 3-D array of frames: <num rows x num dims x num_frames>.
    :param num_bookend_frames: number of bookend frames to add at beginning and end of frames.
    :return: New array of frames with bookend frames added.
    '''

    assert type(frames) == np.ndarray and frames.ndim == 3, "'frames' must be 3-dimensional ndarray"
    assert num_bookend_frames == round(num_bookend_frames) and num_bookend_frames >= 0,\
        "'num_bookend_frames must be nonnegative integer'"

    if num_bookend_frames == 0:
        return frames

    new_frames = np.copy(frames)
    bookend_start_frames = np.repeat(new_frames[:, :, :1], num_bookend_frames, axis=2)
    bookend_end_frames = np.repeat(new_frames[:, :, -1:], num_bookend_frames, axis=2)
    new_frames = np.concatenate((bookend_start_frames, new_frames, bookend_end_frames), axis=2)

    return new_frames
# end function add_bookend_frames


def rescale_frames(frames):
    '''
    Rescale each frame so that the first column always has the same scale.

    This function is useful for 'zooming in' on structures that shrink as DQM evolution proceeds.

    We do the following separately for each frame:

        * center all columns (by subtracting the mean from each)
        * divide all columns by the range (max value minus min value) of the first column

    :param frames: 3-D array of frames. No default.
    :return: Rescaled frames.
    '''

    frames = frames - np.mean(frames, axis=0)  # center (subtract mean from every column)

    # get mins and maxes for first column for each frame
    maxes = np.max(frames[:, :1, :], axis=0)
    mins = np.min(frames[:, :1, :], axis=0)

    # divide all columns by column-1 range for each frame
    frames = frames / (maxes - mins)

    return frames
# end function rescale_frames


def smooth_frames(frames, num_new_frames=100, acc_mult=1, verbose=True):
    '''
    Given a 3-D array of frames, interpolate new frames based on a target average speed between frames.

    This function is useful for highlighting the most interesting parts of a DQM evolution. (A common
    'problem' is waiting hundreds of frames for the last few points to stop moving.)

    :param frames: 3-D array of frames. No default.
    :param num_new_frames: Number of frames to create in the output. Default 100.
    :param acc_mult: Acceleration multiplier. Must be positive. If starting average speed of points (between first
        2 output frames) is S, final average speed (between last 2 output frames) will be acc_mult * S. Default 1
        (constant average speed for moving points).
    :param verbose: Boolean: whether to report progress. Default True.
    :return: New 3-D array of smoothed frames.
    '''

    assert num_new_frames > 0, "'num_new_frames' must be positive"
    assert acc_mult > 0, "'acc_mult' must be positive"

    num_frames = frames.shape[2]
    speeds = np.zeros(num_frames)
    dists = np.zeros(num_frames)

    if verbose:
        print('calculating speeds and distances for original frames...')

    for frame_idx in range(1, num_frames):
        deltas = frames[:, :, frame_idx] - frames[:, :, frame_idx - 1]
        frame_speeds = np.linalg.norm(deltas, axis=1)

        # don't include very slow-moving or stopped rows in the mean speed
        mu = np.mean(frame_speeds)
        moving_row_idxs = np.where(frame_speeds > mu / 100)[0]
        speeds[frame_idx] = np.mean(frame_speeds[moving_row_idxs])
        dists[frame_idx] = dists[frame_idx - 1] + speeds[frame_idx]
    # end for each frame (except first)

    # 'target' speed is the average speed across all new frames
    tgt_speed = dists[-1] / (num_new_frames - 1)

    # figure out new speeds between each new frame (first new speed is between new frame 0 and new frame 1)
    first_speed = 2 * tgt_speed / (1 + acc_mult)
    last_speed = acc_mult * first_speed
    speed_diff = last_speed - first_speed
    speed_delta = speed_diff / (num_new_frames - 1)
    new_speeds = [(tgt_speed - (speed_diff / 2)) + speed_delta * idx for idx in range(num_new_frames)]
    assert np.min(new_speeds) > 0, 'all resulting speeds must be positive'

    if verbose:
        print('building new frames...')
    new_frames = np.zeros((frames.shape[0], frames.shape[1], num_new_frames))
    # copy first and last frames
    new_frames[:, :, 0] = frames[:, :, 0]
    new_frames[:, :, -1] = frames[:, :, -1]
    # pointer to 'current' old frame
    old_frame_num = 1
    # track total 'distance covered' by new frames
    new_frame_dist = 0
    for new_frame_idx in range(1, num_new_frames - 1):
        new_frame_dist = new_frame_dist + new_speeds[new_frame_idx - 1]

        # find next old frame with distance greater than or equal to target new distance
        while old_frame_num < num_frames and new_frame_dist > dists[old_frame_num]:
            old_frame_num += 1

        if old_frame_num == num_frames:
            # use last frame
            new_frames[:, :, new_frame_idx] = frames[: , :, -1]
        else:
            # interpolate new frame from previous and current old frame
            prop = (new_frame_dist - dists[old_frame_num - 1]) / (dists[old_frame_num] - dists[old_frame_num - 1])
            new_frame = (1 - prop) * frames[:, :, old_frame_num - 1] + prop * frames[:, :, old_frame_num]
            new_frames[:, :, new_frame_idx] = new_frame
        # end if/else past last original frame or not
    # end for each new frame (except the first)

    return new_frames
# end function smooth_frames

