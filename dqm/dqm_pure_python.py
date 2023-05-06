'''
functions here that are called from the DQM class replicate functionality from the C code (e.g., make_operators_python
instead of MakeOperatorsC), but with everything written in Python.

these python-only functions exist for 2 reasons:
* playing with changes in the code is much easier in Python.
* implementing the logic twice independently acts as a sanity check that the logic is correct.

yes, there is the downside that every change needs to be made in the C code and also here -- this is fragile.

these functions are, of course, much slower than the compiled versions.

'''


import numpy as np


def biggest_outlier(rows):
    '''
    find the row in the input matrix that is the biggest outlier (defined as the row with the nearest
    neighbor that is farthest away).  we do this the simple, brute-force way, comparing every row to
    every other row.

    :param rows: Matrix of rows
    :return: Row number of biggest outlier
    '''

    num_rows, num_cols = rows.shape

    # reshape matrices to enable broadcasting
    # (NOTE: this will work out badly for a large number of rows...)
    rows_1 = rows.reshape((num_rows, 1, num_cols))
    rows_2 = rows.reshape((1, num_rows, num_cols))

    # calculate distances
    deltas = rows_1 - rows_2
    sq_dists = np.sum(deltas ** 2, axis=2)

    # make sure no row is its own nearest neighbor
    max_sq_dist = np.max(sq_dists)
    for i in range(num_rows):
        sq_dists[i, i] = max_sq_dist + 1

    # find nearest-neighbor distance for each row
    nn_dists = np.min(sq_dists, axis=1)

    # find row with largest nearest-neighbor distance
    outlier_idx = np.argmax(nn_dists)

    return outlier_idx
# end function biggest_outliers


def choose_basis_by_distance_python(rows, basis_size, first_basis_row_num):
    '''
    same functionality as ChooseBasisByDistanceC, but all in Python

    :param rows: Matrix of rows to choose from
    :param basis_size: Number of basis rows to choose
    :param first_basis_row_num: First basis row (if < 0, we start with the biggest outlier)
    :return: Array of selected basis_row_nums
    '''

    num_rows, num_cols = rows.shape
    assert basis_size < num_rows, 'basis_size must be less than num_rows'

    all_row_nums = np.array(list(range(num_rows)))

    basis_row_nums = np.zeros(basis_size, dtype=np.int32)
    n_basis = 1
    if first_basis_row_num >= 0:
        basis_row_nums[0] = first_basis_row_num
    else:
        basis_row_nums[0] = biggest_outlier(rows)
    # end if/else starting from specified first row or biggest outlier

    while n_basis < basis_size:
        # get basis rows
        basis_rows = rows[basis_row_nums[:n_basis], :]
        basis_rows = basis_rows.reshape((1, n_basis, num_cols))  # reshape for broadcasting

        # get non-basis 'other' rows
        other_row_nums = np.sort(np.array(list(set(all_row_nums).difference(set(basis_row_nums[:n_basis])))))
        other_rows = rows[other_row_nums, :]
        other_rows = other_rows.reshape((other_rows.shape[0], 1, num_cols))  # reshape for broadcasting

        # find cartesian product of distances between basis rows and other rows
        # note: other rows are the rows of this matrix, basis rows are the columns
        dists = np.linalg.norm(other_rows - basis_rows, axis=2)
        min_dists = np.min(dists, axis=1)  # find distance to closest basis row for each other row
        idx = np.argmax(min_dists)  # find row which is farthest from any current basis row
        new_basis_row_num = other_row_nums[idx]

        n_basis += 1
        basis_row_nums[n_basis - 1] = new_basis_row_num
    # end while building basis

    return basis_row_nums
# end function choose_basis_by_distance


def build_overlaps_python(sigma, basis_rows, rows):
    '''
    same functionality as BuildOverlapsC, but all in Python

    :param sigma: Value for sigma
    :param basis_rows: Matrix of basis rows
    :param rows: Matrix of other rows
    :return: Vector of basis overlap for each 'other' row
    '''

    basis_overlaps = build_overlaps(basis_rows, basis_rows, sigma)
    simt = build_simt(basis_overlaps)

    row_overlaps = build_overlaps(basis_rows, rows, sigma)

    # convert the 'other' overlaps to the orthonormal basis of eigenstates
    ortho_overlaps = simt @ row_overlaps

    # calculate L2-norm of total overlaps for the rows (which are in the columns of ortho_overlaps)
    overlaps = np.linalg.norm(ortho_overlaps, axis=0)

    return overlaps
# end function build_overlaps_python


def make_operators_python(mat, n_basis, n_potential, sigma, step, mass):
    '''
    same functionality as MakeOperatorsC, but all in Python

    :param mat: The frame-0 matrix
    :param n_basis: Use this number of rows for the basis, starting from the first row.  if None, we use all rows.
    :param n_potential: Use this number of rows to use to build the potential, starting from the first row.
        if None, we use all rows.
    :param sigma: Value for sigma
    :param step: Value for step
    :param mass: Value for mass
    :return: A tuple of (simt, xops, exph)
    '''

    num_rows = mat.shape[0]
    assert n_basis <= num_rows and n_potential <= num_rows, \
        "'n_basis' and 'n_potential' must not be greater than number of rows"

    basis_rows = mat[:n_basis, :]
    overlaps = build_overlaps(basis_rows, basis_rows, sigma)
    simt = build_simt(overlaps)
    xops = build_position_operators(basis_rows, overlaps, simt)
    v0 = build_raw_potential(mat[:n_potential, :], basis_rows, overlaps, sigma)
    exph = build_hamiltonian(basis_rows, overlaps, simt, v0, sigma, step, mass)

    return simt, xops, exph
# end function make_operators_python


def build_overlaps(rows_1, rows_2, sigma):
    '''
    Return a <num_rows_1 x num_rows_2> matrix with the pairwise Gaussian overlap (inner product) of each row
    from rows_1 with each row from rows_2.

    :param rows_1: A matrix of rows.
    :param rows_2: A matrix of rows.
    :param sigma: Value of sigma.
    :return: A <num_rows_1 x num_rows_2> matrix with the pairwise Gaussian overlap (inner product) of each row
        from rows_1 with each row from rows_2.
    '''

    num_cols = rows_1.shape[1]
    assert rows_2.shape[1] == num_cols, 'number of columns must be the same'

    # reshape matrices to enable broadcasting
    rows_1 = rows_1.reshape((rows_1.shape[0], 1, num_cols))
    rows_2 = rows_2.reshape((1, rows_2.shape[0], num_cols))

    deltas = rows_1 - rows_2
    sq_dists = np.sum(deltas ** 2, axis=2)

    overlaps = np.exp(-sq_dists / (4 * sigma ** 2))
    overlaps = np.where(overlaps < 1e-12, 0, overlaps)

    return overlaps
# end method build_overlaps


def build_simt(overlaps):
    '''
    Build and return the transpose of the 'similarity' matrix (used to convert state vectors from the 'raw' basis
    to the eigenbasis).

    Note that the returned matrix may not be square, because the number of eigenbasis states may be smaller than
    the number of basis rows.

    :param overlaps: Square symmetric matrix of basis-basis overlaps.
    :return: A <num eigenbasis states x num basis rows> matrix.
    '''

    # note: use eigh instead of eig because we know the basis-overlap matrix is symmetric.
    # eigh is faster and guarantees the eigenvalues are returned in ascending order of magnitude
    eigenvals, eigenvecs = np.linalg.eigh(overlaps)

    # verify that eigenvectors are all normalized and that eigenvalues are in ascending order
    assert np.allclose(np.linalg.norm(eigenvecs, axis=0), np.ones((1, eigenvals.size))), \
        'overlap eigenvectors must be normalized'
    assert np.allclose(eigenvals, np.sort(eigenvals)), \
        'overlap eigenvalues must be in ascending order'

    # throw away very small eigenvalues
    keep_idxs = eigenvals > 1e-5
    eigenvals = eigenvals[keep_idxs]
    eigenvecs = eigenvecs[:, keep_idxs]

    # compute normalized eigenstates from the eigenvectors
    #
    # we want the eigenstates to have L2 norms of 1 in the eigenbasis space.  with the current eigenvectors,
    # converting the overlap matrix to the eigenbasis space looks like this:
    #        eigenvecs.T @ overlaps @ eigenvecs = L,
    #  ... where L is a diagonal matrix with the eigenvalues along the diagonal.
    #
    # if we create normalized eigenstates by dividing each eigenvector by the square root of it associated
    # eigenvalue, then the conversion to the eigenbasis space looks like this:
    #        eigenvecs_norm.T @ overlaps @ eigenvecs_norm = I,
    # ... where I is the identity matrix, which is what we want.
    inverse_sqrt_eigenvals = np.diag(eigenvals ** -0.5)  # convert to diagonal matrix
    eigenvecs_norm = eigenvecs @ inverse_sqrt_eigenvals
    eigenvecs_norm_t = eigenvecs_norm.T

    # TEMP/DEBUGGING DOUBLE-CHECK:
    mat = eigenvecs_norm_t @ overlaps @ eigenvecs_norm
    assert np.allclose(mat, np.diag(np.ones(mat.shape[1]))), 'mat must be the identity matrix'

    return eigenvecs_norm_t
# end function build_simt


def build_position_operators(basis_rows, overlaps, simt):
    '''
    Build and return the 3-D position-operator tensor.

    :param basis_rows: Matrix of the basis rows.
    :param overlaps: Square symmetric matrix of the basis-basis overlaps.
    :param simt: Transpose of the 'similarity' matrix.
    :return: A 3-D array, where slice i in 3rd dimension is the position-expectation operator matrix for data
        dimension i. Dimensions: <num basis vectors x num basis vectors x num dims>.
    '''

    num_rows, num_cols = basis_rows.shape

    # calculate coordinates of all pairwise basis-basis midpoints (<n x n x num_dims>)
    # reshape 2 copies of the basis rows to broadcast them across each other
    rows1 = basis_rows.reshape((num_rows, 1, num_cols))
    rows2 = basis_rows.reshape((1, num_rows, num_cols))
    midpoints = (rows1 + rows2) / 2

    # raw_operators is <num_basis_rows x num_basis_rows x num_dims>
    raw_operators = np.expand_dims(overlaps, axis=2) * midpoints

    # note: NumPy matrix multiplication will broadcast, but the matching dimensions for
    # matrix multiplication must be *last*
    raw_operators = np.moveaxis(raw_operators, -1, 0)  # put the last dimension first

    # convert to basis eigenstate representation:
    operators = simt @ raw_operators @ simt.T
    operators = np.moveaxis(operators, 0, -1)  # put the first dimension last again

    return operators
# end function build_position_operators


def build_raw_potential(rows, basis_rows, overlaps, sigma):
    '''
    Build and return the 'raw' potential matrix.

    :param rows: Matrix of all rows.
    :param basis_rows: Matrix of basis rows.
    :param overlaps: Square symmetric matrix of basis-basis overlaps.
    :param sigma: Value of sigma.
    :return: a <num basis rows x num basis rows> matrix of the potential calculated at every basis-basis midpoint.
    '''

    # calculate coordinates of all pairwise basis-basis midpoints (<n x n x num_dims>)
    # reshape 2 copies of the basis rows to broadcast them across each other
    num_basis_rows, num_cols = basis_rows.shape
    rows1 = basis_rows.reshape((num_basis_rows, 1, num_cols))
    rows2 = basis_rows.reshape((1, num_basis_rows, num_cols))
    midpoints = (rows1 + rows2) / 2

    # each row will make a contribution to the wave function at each basis/basis midpoint
    psi = np.zeros((num_basis_rows, num_basis_rows))

    # each row will make a contribution to the second derivative of the wave function (the Laplacian)
    # at each basis/basis midpoint
    lpl = np.zeros((num_basis_rows, num_basis_rows))

    for i in range(rows.shape[0]):
        # for each row...
        row = rows[i, :]
        row = row.reshape((1, 1, row.size))

        # get squared distance from the row to each basis/basis midpoint
        vecs_to_midpoints = midpoints - row
        sq_dists = np.sum(vecs_to_midpoints ** 2, axis=2)

        # calculate this row's contribution to the wave function at each basis/basis midpoint
        row_psi = np.exp(-sq_dists / (2 * sigma ** 2))

        # calculate each row's contribution to the Laplacian at each basis/basis midpoint
        row_lpl = sq_dists * row_psi

        # aggregate the wave function and the Laplacian of the wave function
        psi += row_psi
        lpl += row_lpl
    # end for each row

    # calculate the potential at each basis/basis midpoint and multiply by the basis-basis overlaps.
    pot = lpl / psi
    pot *= overlaps

    return pot
# end function build_raw_potential


def build_hamiltonian(basis_rows, overlaps, simt, v0, sigma, step, mass):
    '''
    Build and return the 'evolution' (that is, the exponentiated Hamiltonian time-evolution) operator matrix.

    :param basis_rows: Matrix of basis rows.
    :param overlaps: Square symmetric matrix of basis-basis overlaps.
    :param simt: Transpose of the 'similarity' matrix.
    :param v0: The initial-potential matrix.
    :param sigma: Value of sigma.
    :param step: Value of step.
    :param mass: Value of mass.
    :return: A complex-valued <num basis vectors x num basis vectors> matrix.
    '''

    # calculate all pairwise basis-basis squared distances (<n x n>)
    num_rows, num_cols = basis_rows.shape
    # reshape 2 copies of the basis rows to broadcast them across each other
    rows1 = basis_rows.reshape((num_rows, 1, num_cols))
    rows2 = basis_rows.reshape((1, num_rows, num_cols))
    deltas = rows1 - rows2
    sq_dists = np.sum(deltas ** 2, axis=2)

    h0 = -overlaps * sq_dists

    # note: the 1/sigma^2 factor below makes the whole system scale-invariant: if raw data and sigma are both
    # multiplied by an arbitrary scalar factor, the dynamics will not change.) [2FIX: using sigma as the
    # overall scale factor here causes sigma and time step to be coupled: a fixed step value becomes
    # effectively bigger as the scale of the initial Hamiltonian gets smaller. sigma and step could be
    # decoupled here by using something else as the overall scale factor -- say, the mean pairwise distance
    # between all points in the data set. (this would still have the unfortunate effect of making the dynamics
    # very slightly different as points are added to or removed from the data set -- but this is true more
    # generally in any case.) doing this in the compiled code would require implementing a version of the
    # DQM class method estimate_mean_row_distance in the C++ code.]

    h = v0 + (h0 / mass)
    h /= sigma ** 2

    # convert raw hamiltonian to the eigenbasis
    h_trunc = simt @ h @ simt.T

    ### build the exponentiation/iteration operator
    exph = build_hamiltonian_exp_iter(h_trunc, step)

    return exph
# end method build_hamiltonian


def build_hamiltonian_exp_iter(h, step):
    '''
    Exponentiate the Hamiltonian from time 0 to time 'step'.

    To simplify exponentiation of the Hamiltonian matrix, we find the eigenvalues and eigenvectors, exponentiate
    the eigenvalues along the diagonal of a matrix (with zeros elsewhere), then use the eigenvectors to convert
    the matrix back to the original representation.

    :param h: The initial Hamiltonian matrix.
    :param step: Value of step.
    :return: A complex-valued <num basis vectors x num basis vectors> matrix.
    '''

    # note: use eigh instead of eig because we know the matrix is symmetric.
    # eigh is faster and guarantees the eigenvalues are returned in ascending order of magnitude
    h_eigenvals, h_eigenvecs = np.linalg.eigh(h)

    # the eigenvalues of the Hamiltonian represent different energy levels, but differences in energy
    # levels are the only things we actually care about.  if the overall energy is large, all eigenvalues will be
    # large, creating the possibility of losing information about differences between energy levels at
    # the machine-precision level.  subtracting the first (smallest) energy level from every energy level
    # avoids this potential problem.
    h_eigenvals -= h_eigenvals[0]

    diag_elems = np.exp(-1j * step * h_eigenvals)  # exponentiate the eigenvalues
    diag_elems = np.diag(diag_elems)  # convert from vector to diagonal matrix

    # convert back to original representation (mathematically, this is the inverse of the conversion process
    # described in the comments in build_basis (eigenvecs.T @ overlaps @ eigenvecs = L)
    return h_eigenvecs @ diag_elems @ h_eigenvecs.T
# end method build_hamiltonian_exp_iter


def build_frames_python(num_frames_to_build, current_frame, basis_rows, simt, xops, exph, sigma,
                        stopping_threshold):
    '''
    same functionality as BuildFramesAutoC, but all in Python

    :param num_frames_to_build: Number of new frames to build.
    :param current_frame: Current last frame, from which we build new frames.
    :param basis_rows: Matrix of basis rows.
    :param simt: Transpose of 'similarity' matrix.
    :param xops: 3-D tensor of position-operator matrices.
    :param exph: Hamiltonian 'evolution' operator matrix.
    :param sigma: Value of sigma.
    :param stopping_threshold: Value of stopping threshold.

    :return: 3-D array of new_frames.
    '''

    num_rows, num_cols = current_frame.shape

    assert num_cols == basis_rows.shape[1], 'number of columns must be consistent'
    assert sigma > 0, 'sigma must be positive'

    # create combined operators
    combo_op = exph @ simt

    stopped_row_idxs = []  # initialize variable (just for lint)

    evolving_row_idxs = np.array(list(range(num_rows)))
    new_frames = np.empty((num_rows, num_cols, 0))
    for frame_idx in range(num_frames_to_build):
        new_frame = np.zeros((num_rows, num_cols))
        if evolving_row_idxs.size > 0:
            # get basis-overlap vectors for each evolving row
            overlap_vecs = build_overlaps(basis_rows, current_frame[evolving_row_idxs, :], sigma)

            # apply combined operators to get evolved states
            new_states = combo_op @ overlap_vecs

            # normalize new states
            new_states /= np.linalg.norm(new_states, axis=0)

            # get new expected position for each evolved state
            new_positions = expected_positions(new_states, xops)

            # put new positions into new frame
            new_frame[evolving_row_idxs, :] = new_positions
        # end if any rows still evolving

        if evolving_row_idxs.size < num_rows:
            # copy last positions forward for stopped rows
            new_frame[stopped_row_idxs, :] = current_frame[stopped_row_idxs, :]

        # update lists of evolving and stopped rows
        dists = np.linalg.norm(new_frame - current_frame, axis=1)
        stopped_row_idxs = np.where(dists < stopping_threshold)[0]
        evolving_row_idxs = np.where(dists >= stopping_threshold)[0]

        new_frames = np.concatenate((new_frames, new_frame[:, :, np.newaxis]), axis=2)
        current_frame = new_frame
    # end for each new frame

    return new_frames
# end function build_frames_python


def expected_positions(states, xops):
    '''
    Calculate and return new expected positions for a set of evolved states.

    :param states: A <num basis vectors x num rows> matrix of evolved states in the eigenbasis.
    :param xops: 3-D tensor of position-operator matrices.
    :return: A <num rows x num dims> matrix of expected positions for the evolved states.
    '''

    num_states = states.shape[1]
    num_dims = xops.shape[2]
    positions = np.zeros((num_states, num_dims))

    # calculate expected position for each state
    for idx in range(num_states):
        state = states[:, [idx]]  # note: indexing with a 1-item list keeps dimension info
        state_real = state.real
        state_imag = state.imag

        # note: this is +, not a -, because we take the complex conjugate of 1 of them
        real_mat = state_real @ state_real.T  # cartesian product of real values for the state
        imag_mat = state_imag @ state_imag.T  # cartesian product of imaginary values for the state
        norm_mat = real_mat + imag_mat

        # pos will be <num_basis_eigenstates x num_basis_eigenstates x num_dims>
        pos = np.expand_dims(norm_mat, axis=2) * xops

        # collapse both basis-eigenstate dimensions
        pos = np.sum(pos, axis=1)
        pos = np.sum(pos, axis=0)

        positions[idx, :] = pos
    # end for each state

    return positions
# end method expected_positions

