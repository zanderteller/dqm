from . import utils
from .DQM import DQM
import numpy as np
import unittest


'''
Use the following command to run these tests from the command line:

python -m unittest dqm.UtilTests
'''


class UtilTests(unittest.TestCase):


    def test_pca(self):
        # test 1: num_rows > num_cols
        mat = np.random.rand(2000, 500)
        eigvals, eigvecs = utils.pca(mat, False)
        mat = mat - np.mean(mat, axis=0)
        u, s, vt = np.linalg.svd(mat, full_matrices=False, compute_uv=True)
        self.compare_pca_results(eigvals, eigvecs, s, vt, mat.shape[0])

        # test 2: num_rows < num_cols
        mat = np.random.rand(500, 2000)
        eigvals, eigvecs = utils.pca(mat, False)
        mat = mat - np.mean(mat, axis=0)
        u, s, vt = np.linalg.svd(mat, full_matrices=False, compute_uv=True)
        self.compare_pca_results(eigvals, eigvecs, s, vt, mat.shape[0])
    # end method test_pca


    def compare_pca_results(self, eigvals, eigvecs, s, vt, num_rows):
        '''
        helper method for test_pca method: test that PCA eigenvalues/eigenvectors match S and VT from
        the SVD decomposition.

        :param eigvals: Vector of PCA eigenvalues.
        :param eigvecs: Matrix with PCA eigenvectors in the columns.
        :param s: Vector of SVD singular values.
        :param vt: Transposed matrix of SVD right singular values.
        :param num_rows: Number of rows in original matrix.
        :return: None
        '''

        # for each positive eigenvalue:
        #  * test that eigenvector matches the corresponding right-singular vector, up to sign
        #  * test that that eigenvalue has the right relationship to the corresponding singular value:
        #    e = s^2 / (num_rows - 1)
        for vec_idx in range(eigvecs.shape[1]):
            if eigvals[vec_idx] > 0:
                self.assertTrue(np.allclose(eigvecs[:, vec_idx], vt[vec_idx, :]) or
                                np.allclose(eigvecs[:, vec_idx], -vt[vec_idx, :]), \
                                f'eigvenvector {vec_idx} must match right-singular vector {vec_idx} (up to sign)')
                self.assertTrue(np.isclose(eigvals[vec_idx], s[vec_idx] ** 2 / (num_rows - 1)), \
                                f'eigvenvalue {vec_idx} must have correct relationship to singular value {vec_idx}')
    # end method compare_pca_results


    def test_get_clusters(self):
        rng = np.random.default_rng(119)
        mat = rng.random((100, 15))

        num_rows = mat.shape[0]

        # test case where all rows are singletons
        clusters, cluster_sizes = utils.get_clusters(mat, 0.1)
        self.assertTrue(len(cluster_sizes) == num_rows, 'must return all singletons for small max_dist')

        # test case where all rows are in a single cluster
        clusters, cluster_sizes = utils.get_clusters(mat, 2)
        self.assertTrue(len(cluster_sizes) == 1 and cluster_sizes[0] == num_rows,
                        'must return single cluster for large max_dist')

        # test case where results with multiple clusters come out as expected
        clusters, cluster_sizes = utils.get_clusters(mat, 1)
        self.assertTrue(len(clusters) == 50 and
                        cluster_sizes[:3] == [17, 14, 11] and
                        clusters[0] == [2, 4, 5, 8, 12, 17, 40, 46, 47, 49, 57, 63, 65, 75, 84, 85, 91],
                        'returned clusters for max_dist of 1 must be as expected')
    # end method test_get_clusters


    def test_get_clusters_python(self):
        rng = np.random.default_rng(6709)
        mat = rng.random((100, 12))

        # test that C and Python versions return the same results
        max_dist = 0.8
        clusters, cluster_sizes = utils.get_clusters(mat, max_dist)
        clusters_python, cluster_sizes_python = utils._get_clusters_python(mat, max_dist)

        # sort all clusters
        clusters = [sorted(cluster) for cluster in clusters]
        clusters_python = [sorted(cluster) for cluster in clusters_python]

        # order of singletons is different -- we won't worry about that
        all_found_1 = all([cluster in clusters for cluster in clusters_python])
        all_found_2 = all([cluster in clusters_python for cluster in clusters])
        self.assertTrue(all_found_1 and all_found_2, 'C and Python versions must return same results')
    # end method test_get_clusters_python


    def test_nearest_neighbors(self):
        rng = np.random.default_rng(119)
        mat = rng.random((50, 17))

        nn_row_nums, nn_distances = utils.nearest_neighbors(mat)

        # test that results for first row are correct
        distances = np.linalg.norm(mat - mat[0, :], axis=1)
        distances[0] = np.max(distances)  # don't choose row as its own nearest neighbor
        nn_row_num = np.argmin(distances)
        nn_distance = distances[nn_row_num]
        self.assertTrue(nn_row_num == nn_row_nums[0] and np.isclose(nn_distance, nn_distances[0]),
                        'nearest-neighbor results must be correct for first row')

        # test that results for last row are correct
        distances = np.linalg.norm(mat - mat[-1, :], axis=1)
        distances[-1] = np.max(distances)  # don't choose row as its own nearest neighbor
        nn_row_num = np.argmin(distances)
        nn_distance = distances[nn_row_num]
        self.assertTrue(nn_row_num == nn_row_nums[-1] and np.isclose(nn_distance, nn_distances[-1]),
                        'nearest-neighbor results must be correct for last row')
    # end method test_nearest_neighbors


    # 2FIX: WHAT'S THE BEST WAY TO RUN HEADLESS TESTS ON A PLOTTING METHOD?
    def test_plot_frames(self):
        pass
    # end method test_plot_frames


    # 2FIX: WHAT'S THE BEST WAY TO RUN HEADLESS TESTS ON A PLOTTING METHOD?
    def test_plot_frames_ipv(self):
        pass
    # end method test_plot_frames_ipv


    def test_cat_frames(self):
        rng = np.random.default_rng(8838)

        # test that mismatch in number of columns raises an error
        frames1 = rng.random((20, 13, 10))
        frames2 = rng.random((30, 14, 20))
        try:
            utils.cat_frames(frames1, frames2)
            success = False
        except AssertionError:
            success = True
        self.assertTrue(success, "must raise AssertionError when numbers of columns don't match")

        # test that catting works for equal numbers of frames
        frames1 = rng.random((20, 12, 10))
        frames2 = rng.random((30, 12, 10))
        frames = utils.cat_frames(frames1, frames2)
        self.assertTrue(frames.shape[0] == frames1.shape[0] + frames2.shape[0] and
                        frames.shape[1] == frames1.shape[1] == frames2.shape[1] and
                        frames.shape[2] == frames1.shape[2] == frames2.shape[2],
                        "cat_frames must work as expected for equal numbers of frames")

        # test that catting works when frames1 has more frames
        frames1 = rng.random((20, 18, 30))
        frames2 = rng.random((30, 18, 15))
        frames = utils.cat_frames(frames1, frames2)
        self.assertTrue(frames.shape[2] == max(frames1.shape[2], frames2.shape[2]),
                        "cat_frames must work as expected when frames1 has more frames")

        # test that catting works when frames2 has more frames
        frames1 = rng.random((20, 18, 35))
        frames2 = rng.random((30, 18, 50))
        frames = utils.cat_frames(frames1, frames2)
        self.assertTrue(frames.shape[2] == max(frames1.shape[2], frames2.shape[2]),
                        "cat_frames must work as expected when frames2 has more frames")

        # test that passing [frames1, frames2] as a list produces the same result
        frames_new = utils.cat_frames([frames1, frames2])
        self.assertTrue(np.array_equal(frames, frames_new),
                        "cat_frames must produce same results when inputs are passed in as list")

        # test that passing a list with more than 2 elements works as expected
        frames3 = rng.random((47, 18, 63))
        frames = utils.cat_frames([frames1, frames2, frames3])
        self.assertTrue(frames.shape[2] == max(frames1.shape[2], frames2.shape[2], frames3.shape[2]),
                        "cat_frames must work as expected with more than 2 inputs passed in as list")
    # end method test_cat_frames


    def test_add_bookend_frames(self):
        rng = np.random.default_rng(19298)
        frames = rng.random((100, 20, 50))

        # test that adding zero bookend frames leaves frames unchanged
        new_frames = utils.add_bookend_frames(frames, 0)
        self.assertTrue(np.array_equal(frames, new_frames),
                        'adding zero bookend frames must leave original frames unchanged')

        # test that adding bookend frames has the expected result
        num_bookend_frames = 5
        new_frames = utils.add_bookend_frames(frames, num_bookend_frames)
        self.assertTrue(np.array_equal(new_frames[:, :, 0], new_frames[:, :, num_bookend_frames - 1]),
                        'starting bookend frames must be identical')
        self.assertTrue(np.array_equal(new_frames[:, :, -num_bookend_frames], new_frames[:, :, -1]),
                        'ending bookend frames must be identical')
    # end method test_add_bookend_frames


    def test_rescale_frames(self):
        rng = np.random.default_rng(10019)
        frames = rng.random((100, 20, 50))

        # test that undoing rescaling returns original frames
        new_frames = utils.rescale_frames(frames)
        # multiply all columns by original ranges
        maxes = np.max(frames[:, :1, :], axis=0)
        mins = np.min(frames[:, :1, :], axis=0)
        new_frames *= (maxes - mins)
        # add original column means
        new_frames += np.mean(frames, axis=0)
        self.assertTrue(np.allclose(frames, new_frames), 'frames must match after undoing rescaling')
    # end method test_rescale_frames


    def test_smooth_frames(self):
        rng = np.random.default_rng(33322)
        dat = rng.random((100, 20))

        sigma = 1

        dqm_obj = DQM()
        dqm_obj.verbose = False
        dqm_obj.run_simple(dat, sigma)

        num_new_frames = 87

        # note: speed ratios below don't match acc_mult values because the function doesn't include
        # very slow or stopped rows in mean speed calculations

        # test with positive acceleration (> 1)
        acc_mult = 2
        new_frames = utils.smooth_frames(dqm_obj.frames, num_new_frames, acc_mult=acc_mult, verbose=False)
        mean_first_speed = np.mean(np.linalg.norm(new_frames[:, :, 1] - new_frames[:, :, 0], axis=1))
        mean_last_speed = np.mean(np.linalg.norm(new_frames[:, :, -1] - new_frames[:, :, -2], axis=1))
        self.assertTrue(new_frames.shape[2] == num_new_frames and
                        2.45 < mean_last_speed / mean_first_speed < 2.46,
                        'smoothevolution must produce expected number of frames with faster speed at the end')

        # test with negative acceleration (< 1)
        acc_mult = 0.5
        new_frames = utils.smooth_frames(dqm_obj.frames, num_new_frames, acc_mult=acc_mult, verbose=False)
        mean_first_speed = np.mean(np.linalg.norm(new_frames[:, :, 1] - new_frames[:, :, 0], axis=1))
        mean_last_speed = np.mean(np.linalg.norm(new_frames[:, :, -1] - new_frames[:, :, -2], axis=1))
        self.assertTrue(new_frames.shape[2] == num_new_frames and
                        0.25 < mean_last_speed / mean_first_speed < 0.26,
                        'smoothevolution must produce expected number of frames with slower speed at the end')
    # end method test_smooth_frames


# end class UtilTests


if __name__ == '__main__':
    unittest.main()

