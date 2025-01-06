from . import DQM, get_clusters
import numpy as np
from copy import copy
import unittest


'''
Use the following command to run these tests from the command line:

python -m unittest dqm.DQMTests
'''


class DQMTests(unittest.TestCase):
    def test_instance_variables(self):
        dqm_obj = DQM()
        inst_vars = list(vars(dqm_obj).keys())

        # sanity check: if instance variables are changed, they must be changed here as well
        checklist = ['basis_num_chunks', 'basis_rand_seed', 'basis_row_nums', 'basis_rows', 'basis_size',
                     'basis_start_with_outlier', 'call_c', 'exph', 'frames', 'mass', 'mean_row_distance',
                     'min_report_time', 'non_basis_row_nums', 'overlap_mean_threshold', 'overlap_min_threshold',
                     'pca_cum_var', 'pca_eigvals', 'pca_eigvecs', 'pca_num_dims', 'pca_transform', 'pca_var_threshold',
                     'raw_col_means', 'raw_data', 'sigma', 'simt', 'step', 'stopping_threshold', 'verbose', 'xops']
        self.assertTrue(sorted(inst_vars) == sorted(checklist), 'DQM instance variables must match the checklist')
    # end method test_instance_variables

    '''
    2FIX: there are several places where we could/should test the quality/coherence of results at a deeper level
    * test_choose_basis_by_distance: test the quality/coherence of basis rows (are they far apart?)
    * test_build_frames – test the quality/coherence of built frames (e.g., test for oscillation?)
    * test_run_new_points – test the quality/coherence of built frames for new points
    '''

    def test_default_mass_for_num_dims(self):
        dqm_obj = DQM()
        dqm_obj.verbose = False

        # test that method raises an error when no number of dimensions is available
        try:
            mass = dqm_obj.default_mass_for_num_dims()
            success = False
        except AssertionError:
            success = True
        self.assertTrue(success, 'must raise AssertionError when no number of dimensions is available')

        # test that method does not raise an error when number of dimensions is available from dqm_obj.frames
        dqm_obj.frames = np.zeros((10, 6, 5))
        try:
            mass = dqm_obj.default_mass_for_num_dims()
            success = True
        except AssertionError:
            success = False
        self.assertTrue(success,
                        'must not raise AssertionError when number of dimensions is available from dqm_obj.frames')

        # test that suggested mass for 10 dimensions is 1
        mass = dqm_obj.default_mass_for_num_dims(10)
        success = np.isclose(mass, 1)
        self.assertTrue(success, 'suggested mass for 10 dimensions must be ~1')

        # test that suggested mass for 100 dimensions is 3
        mass = dqm_obj.default_mass_for_num_dims(100)
        success = np.isclose(mass, 3)
        self.assertTrue(success, 'suggested mass for 100 dimensions must be ~3')
    # end method test_default_mass_for_num_dims


    def test_run_pca(self):
        dqm_obj = DQM()
        dqm_obj.verbose = False

        # test that method raises an error if instance doesn't have raw data
        try:
            dqm_obj.run_pca()
            success = False
        except AssertionError:
            success = True
        self.assertTrue(success, "must raise AssertionError when instance doesn't have raw data")

        rng = np.random.default_rng(119)
        dqm_obj.raw_data = rng.random((100, 15))

        # test that all relevant instance member variables are set
        dqm_obj.run_pca()
        not_none = [x is not None for x in [dqm_obj.raw_col_means, dqm_obj.pca_eigvecs, dqm_obj.pca_eigvals, dqm_obj.pca_cum_var]]
        success = np.all(not_none)
        self.assertTrue(success, 'method must set all relevant member variables')

        # test that pca_cum_var sums to 1
        success = np.isclose(dqm_obj.pca_cum_var[-1], 1)
        self.assertTrue(success, 'pca_cum_var[-1] must be ~1')
    # end method test_run_pca


    def test_clear_pca(self):
        dqm_obj = DQM()
        dqm_obj.verbose = False

        rng = np.random.default_rng(273)
        dqm_obj.raw_data = rng.random((123, 21))
        dqm_obj.run_pca()
        dqm_obj.clear_pca()

        # test that all relevant instance member variables are reset
        is_none = [x is None for x in
                   [dqm_obj.pca_num_dims, dqm_obj.raw_col_means, dqm_obj.pca_eigvecs, dqm_obj.pca_eigvals, dqm_obj.pca_cum_var]
                   ]
        success = np.all(is_none)
        self.assertTrue(success, 'method must reset all relevant member variables')
    # end method test_clear_pca


    # 2FIX: WHAT'S THE BEST WAY TO RUN HEADLESS TESTS ON A PLOTTING METHOD?
    def test_plot_pca(self):
        pass
    # end method test_plot_pca


    def test_choose_num_pca_dims(self):
        dqm_obj = DQM()
        dqm_obj.verbose = False

        rng = np.random.default_rng(433)
        dqm_obj.raw_data = rng.random((200, 37))
        dqm_obj.run_pca()

        # test that method defaults to using all PCA dimensions
        dqm_obj.pca_num_dims = None
        dqm_obj.pca_var_threshold = None
        num_pca_dims = dqm_obj._choose_num_pca_dims()
        success = num_pca_dims == dqm_obj.pca_eigvals.size
        self.assertTrue(success, "must default to using all PCA dimensions")

        # test pca_var_threshold
        dqm_obj.pca_var_threshold = 0.75
        num_pca_dims = dqm_obj._choose_num_pca_dims()
        # note: the correct value of 22 is specific to the random seed used above
        success = num_pca_dims == 22
        self.assertTrue(success, 'pca_var_threshold must work correctly')

        # test pca_num_dims (including that it overrides pca_var_threshold)
        dqm_obj.pca_num_dims = 17
        num_pca_dims = dqm_obj._choose_num_pca_dims()
        success = num_pca_dims == 17
        self.assertTrue(success, 'pca_num_dims must override pca_var_threshold and must work correctly')
    # end method test_choose_num_pca_dims


    def test_create_frame_0(self):
        dqm_obj = DQM()
        dqm_obj.verbose = False

        # test that method raises an error when raw data is not set or passed in
        try:
            dqm_obj.create_frame_0()
            success = False
        except AssertionError:
            success = True
        self.assertTrue(success, "must raise AssertionError when instance doesn't have raw data")

        rng = np.random.default_rng(557)
        mat = rng.random((132, 18))

        # test that frame 0 is just unchanged raw data when pca_transform is 'off'
        dqm_obj.pca_transform = False
        frame0 = dqm_obj.create_frame_0(mat)
        mat2 = frame0[:, :, 0]
        success = np.array_equal(mat, mat2)
        self.assertTrue(success, 'with all flags off, frame 0 must match raw data')

        # test that frame 0 is 3-D
        success = frame0.ndim == 3
        self.assertTrue(success, 'frame 0 must be 3-D')

        # test PCA transform
        dqm_obj.pca_transform = True
        dqm_obj.pca_num_dims = 11
        dqm_obj.raw_data = mat
        dqm_obj.frames = None
        dqm_obj.create_frame_0()  # note: *not* passing in raw data here
        success = dqm_obj.raw_col_means is not None and dqm_obj.pca_eigvecs is not None and dqm_obj.pca_num_dims > 0 and \
                    dqm_obj.frames is not None
        self.assertTrue(success,
                        'creating frame 0 with raw data in instance must set PCA member variables and frame 0')

        # test that passing raw data to the method produces same results as using it when stored in dqm_obj.raw_data
        frame0 = dqm_obj.create_frame_0(mat)
        success = np.array_equal(frame0, dqm_obj.frames)
        self.assertTrue(success,
                        'passing raw data to method must produce same result as accessing it from dqm_obj.raw_data')
    # end method test_create_frame_0


    def test_clear_basis(self):
        dqm_obj = DQM()
        dqm_obj.verbose = False

        rng = np.random.default_rng(699)
        dqm_obj.raw_data = rng.random((202, 33))
        dqm_obj.create_frame_0()
        dqm_obj._set_basis()
        dqm_obj.clear_basis()

        # test that all relevant instance member variables are reset
        success = dqm_obj.basis_row_nums is None and dqm_obj.non_basis_row_nums is None and dqm_obj.basis_rows is None
        self.assertTrue(success, 'method must reset all relevant member variables')
    # end method test_clear_basis


    def test_set_basis(self):
        dqm_obj = DQM()
        dqm_obj.verbose = False

        rng = np.random.default_rng(619)
        dqm_obj.raw_data = rng.random((80, 9))
        dqm_obj.create_frame_0()

        # test setting of full basis
        dqm_obj._set_basis()
        num_rows = dqm_obj.raw_data.shape[0]
        num_basis_rows1 = len(dqm_obj.basis_row_nums)
        num_basis_rows2 = dqm_obj.basis_rows.shape[0]
        num_non_basis_rows = len(dqm_obj.non_basis_row_nums)
        success = num_basis_rows1 == num_rows and num_basis_rows2 == num_rows and num_non_basis_rows == 0
        self.assertTrue(success, 'setting full basis must work properly')

        # test setting of smaller basis
        test_non_basis_n = 10
        dqm_obj._set_basis(list(range(num_rows - test_non_basis_n)))
        num_basis_rows1 = len(dqm_obj.basis_row_nums)
        num_basis_rows2 = dqm_obj.basis_rows.shape[0]
        num_non_basis_rows = len(dqm_obj.non_basis_row_nums)
        success = num_basis_rows1 == num_rows - test_non_basis_n and \
                  num_basis_rows2 == num_rows - test_non_basis_n and \
                  num_non_basis_rows == test_non_basis_n
        self.assertTrue(success, 'setting smaller basis must work properly')
    # end method test_set_basis


    def test_build_operators(self):
        dqm_obj = DQM()
        dqm_obj.verbose = False

        rng = np.random.default_rng(872)
        num_rows = 100
        num_cols = 31
        dqm_obj.raw_data = rng.random((num_rows, num_cols))

        dqm_obj.create_frame_0()

        # test that building operators with sigma equal to mean row distance produces as many eigenstates as basis rows
        dqm_obj.estimate_mean_row_distance()
        dqm_obj.sigma = dqm_obj.mean_row_distance
        dqm_obj.build_operators()
        success = np.array_equal(dqm_obj.simt.shape, [num_rows, num_rows]) and \
                  np.array_equal(dqm_obj.xops.shape, [num_rows, num_rows, num_cols]) and \
                  np.array_equal(dqm_obj.exph.shape, [num_rows, num_rows])
        self.assertTrue(success, 'building operators with full basis must produce operators of expected size')

        # test that building operators with sigma of 10 times mean row distance decreases the number of significant
        # eigenstates to fewer than half of the number of basis rows
        dqm_obj.sigma = dqm_obj.mean_row_distance * 10
        dqm_obj.build_operators()
        simt = np.copy(dqm_obj.simt)
        xops = np.copy(dqm_obj.xops)
        exph = np.copy(dqm_obj.exph)
        num_eigenstates = simt.shape[0]
        success = num_eigenstates < 0.5 * num_rows and simt.shape[1] == num_rows and \
                  xops.shape[0] == num_eigenstates and xops.shape[1] == num_eigenstates and \
                  exph.shape[0] == num_eigenstates and exph.shape[1] == num_eigenstates
        self.assertTrue(success,
                        'building operators with large sigma must reduce the number of significant eigenstates')
    # end method test_build_operators


    def test_choose_basis_by_distance(self):
        dqm_obj = DQM()
        dqm_obj.verbose = False

        rng = np.random.default_rng(708)
        dqm_obj.raw_data = rng.random((163, 21))
        dqm_obj.create_frame_0()

        # test with single chunk
        dqm_obj.basis_start_with_outlier = True
        num_rows = dqm_obj.raw_data.shape[0]
        test_basis_size = 37
        dqm_obj.basis_size = test_basis_size
        dqm_obj.choose_basis_by_distance()
        success = len(dqm_obj.basis_row_nums) == test_basis_size and dqm_obj.basis_rows.shape[0] == test_basis_size and \
                    len(dqm_obj.non_basis_row_nums) == num_rows - test_basis_size
        self.assertTrue(success, 'basis must have the desired number of rows')

        # test with multiple chunks
        basis_row_nums1 = copy(dqm_obj.basis_row_nums)
        dqm_obj.basis_num_chunks = 2
        dqm_obj.choose_basis_by_distance()
        success = dqm_obj.basis_row_nums != basis_row_nums1
        self.assertTrue(success, 'basis rows must be different when running with multiple chunks')

        # test with start_with_outlier off
        dqm_obj.basis_num_chunks = 1
        dqm_obj.basis_start_with_outlier = False
        dqm_obj.choose_basis_by_distance()
        success = dqm_obj.basis_row_nums != basis_row_nums1
        self.assertTrue(success, 'basis rows must be different when not starting with outlier')
    # end method test_choose_basis_by_distance


    def test_build_overlaps(self):
        dqm_obj = DQM()
        dqm_obj.verbose = False

        rng = np.random.default_rng(760)
        num_rows = 150
        num_cols = 17
        dqm_obj.raw_data = rng.random((num_rows, num_cols))
        dqm_obj.create_frame_0()

        dqm_obj.basis_size = 50
        dqm_obj.choose_basis_by_distance()

        # test building overlaps for all non-basis rows (row numbers passed in)
        row_nums = dqm_obj.non_basis_row_nums
        sigma = 0.6
        batch_size = 100
        overlaps = dqm_obj.build_overlaps(row_nums=row_nums, sigma=sigma, batch_size=batch_size)
        success = overlaps.size == num_rows - dqm_obj.basis_size and \
                  np.min(overlaps) > 0.45 and np.max(overlaps) < 0.82 and \
                  0.637 < np.mean(overlaps) < 0.639
        self.assertTrue(success, 'overlaps must be within expected range')

        # test building overlaps for all non-basis rows (using row numbers stored in instance)
        overlaps2 = dqm_obj.build_overlaps(sigma=sigma, batch_size=batch_size)
        success = np.array_equal(overlaps, overlaps2)
        self.assertTrue(success, 'overlaps must match when using non-basis row numbers stored in instance')

        # test building overlaps for all non-basis rows (using row numbers stored in instance, in smaller batches)
        batch_size = 10
        overlaps2 = dqm_obj.build_overlaps(sigma=sigma, batch_size=batch_size)
        success = np.array_equal(overlaps, overlaps2)
        self.assertTrue(success, 'overlaps must match when using smaller batch size')

        # test building overlaps for all non-basis rows (using row numbers stored in instance, with default
        # batch size, with sigma stored in instance)
        dqm_obj.sigma = sigma
        overlaps2 = dqm_obj.build_overlaps()
        success = np.array_equal(overlaps, overlaps2)
        self.assertTrue(success, 'overlaps must match when using sigma stored in instance')

        # test building overlaps for arbitrary new rows
        num_new_rows = 10
        new_rows = rng.random((num_new_rows, num_cols))
        new_frame_0 = dqm_obj.create_frame_0(new_rows)
        overlaps = dqm_obj.build_overlaps(new_frame_0)
        success = overlaps.size == num_new_rows and \
                  min(overlaps) > 0.47 and max(overlaps) < 0.70 and \
                  0.586 < np.mean(overlaps) < 0.587
        self.assertTrue(success, 'overlaps for new rows must be within expected range for test data set')
    # end method test_build_overlaps


    def test_estimate_mean_row_distance(self):
        dqm_obj = DQM()
        dqm_obj.verbose = False

        rng = np.random.default_rng(509)
        mat = rng.random((500, 16))

        # test that method raises an error when frame 0 is not set
        dqm_obj.frames = None
        try:
            dqm_obj.estimate_mean_row_distance()
            success = False
        except AssertionError:
            success = True
        self.assertTrue(success, "must raise AssertionError when instance doesn't have frame 0")

        dqm_obj.frames = mat[:, :, np.newaxis]
        dqm_obj.estimate_mean_row_distance()
        mu1 = dqm_obj.mean_row_distance

        # test that multiplying raw data by 100 produces the right mean
        test_factor = 100
        dqm_obj.frames = test_factor * mat[:, :, np.newaxis]
        dqm_obj.estimate_mean_row_distance()
        mu2 = dqm_obj.mean_row_distance
        success = np.isclose(mu2 / mu1, test_factor)
        self.assertTrue(success, 'multiplying raw data by 100 must multiply estimated mean by 100')

        # test that, with a different random seed, estimated mean is still close the same value
        dqm_obj.estimate_mean_row_distance(0.01, 17)
        mu3 = dqm_obj.mean_row_distance
        success = np.isclose(mu3 / mu1, test_factor, rtol=0.02, atol=0.02)
        self.assertTrue(success, 'running with a different random seed must produce a similar estimated mean')
    # end method test_estimate_mean_row_distance


    def test_choose_sigma_for_basis(self):
        dqm_obj = DQM()
        dqm_obj.verbose = False

        rng = np.random.default_rng(788)
        num_rows = 600
        num_cols = 19
        dqm_obj.raw_data = rng.random((num_rows, num_cols))
        dqm_obj.create_frame_0()

        dqm_obj.basis_size = 100
        dqm_obj.choose_basis_by_distance()

        # test choosing sigma using all non-basis rows in a single batch
        dqm_obj.choose_sigma_for_basis()
        sigma = dqm_obj.sigma
        test_ratio = sigma / dqm_obj.mean_row_distance
        success = test_ratio > 0.4 and test_ratio < 0.6
        self.assertTrue(success, 'chosen sigma must be near half of mean row distance')

        # test choosing sigma in batches
        batch_size = 20
        num_batches_to_test = 6
        dqm_obj.choose_sigma_for_basis(batch_size=batch_size, num_batches_to_test=num_batches_to_test)
        success = dqm_obj.sigma == sigma
        self.assertTrue(success, 'choosing sigma starting from smaller batches must produce same final sigma')
    # end method test_choose_sigma_for_basis


    def test_stopped_row_nums(self):
        dqm_obj = DQM()
        dqm_obj.verbose = False
        dqm_obj.stopping_threshold = 1e-4

        rng = np.random.default_rng(811)
        num_rows = 100
        num_cols = 22

        # test that a 2-D matrix returns no stopped row numbers
        frames = rng.random((num_rows, num_cols))
        stopped_row_nums = dqm_obj._stopped_row_nums(frames)
        success = stopped_row_nums == []
        self.assertTrue(success, '2-D matrix must produce no stopped row numbers')

        # test that a 3-D array with 1 slice in dim 3 returns no stopped row numbers
        frames = frames[:, :, np.newaxis]  # convert to 3-D array
        stopped_row_nums = dqm_obj._stopped_row_nums(frames)
        success = stopped_row_nums == []
        self.assertTrue(success, '3-D array with 1 slice in dim 3 must produce no stopped row numbers')

        # test that a 3-D array with 2 identical slices in dim 3 returns all row numbers as stopped
        frames = np.concatenate((frames, frames), axis=2)
        stopped_row_nums = dqm_obj._stopped_row_nums(frames)
        success = stopped_row_nums == list(range(num_rows))
        self.assertTrue(success, '3-D array with 2 identical slices in dim 3 must return all row numbers as stopped')

        # test that changing a single row causes that row to no longer be seen as stopped
        changed_row_num = 17
        frames[changed_row_num, :, -1] = rng.random(num_cols)
        stopped_row_nums = dqm_obj._stopped_row_nums(frames)
        not_stopped_row_nums = list(set(range(num_rows)).difference(set(stopped_row_nums)))
        success = np.array_equal(not_stopped_row_nums, [changed_row_num])
        self.assertTrue(success, 'changed row must not be seen as stopped')

        # test that method uses stored frames correctly
        dqm_obj.frames = frames
        stopped_row_nums = dqm_obj._stopped_row_nums()
        not_stopped_row_nums = list(set(range(num_rows)).difference(set(stopped_row_nums)))
        success = np.array_equal(not_stopped_row_nums, [changed_row_num])
        self.assertTrue(success, 'stopped_row_nums must use stored frames correctly')
    # end method test_stopped_row_nums


    def test_set_stopping_threshold(self):
        dqm_obj = DQM()
        dqm_obj.verbose = False

        rng = np.random.default_rng(850)
        num_rows = 60
        num_cols = 14

        dqm_obj.raw_data = rng.random((num_rows, num_cols))
        dqm_obj.create_frame_0()

        # test that default stopping threshold is None
        success = dqm_obj.stopping_threshold is None
        self.assertTrue(success, "default stopping threshold must be None")

        # test that set_stopping_threshold creates a positive stopping threshold less than mean row distance
        dqm_obj.set_stopping_threshold()
        success = dqm_obj.stopping_threshold > 0 and dqm_obj.stopping_threshold < dqm_obj.mean_row_distance
        self.assertTrue(success, 'new stopping threshold must be positive and less than mean row distance')
    # end method test_set_stopping_threshold


    def test_build_frames(self):
        dqm_obj = DQM()
        dqm_obj.verbose = False

        rng = np.random.default_rng(910)
        num_rows = 200
        num_cols = 25
        dqm_obj.raw_data = rng.random((num_rows, num_cols))
        dqm_obj.create_frame_0()
        dqm_obj.basis_size = 100
        dqm_obj.choose_basis_by_distance()
        dqm_obj.choose_sigma_for_basis()
        dqm_obj.build_operators()

        num_frames_to_build = 300

        # test that BuildFrames return expected result
        dqm_obj.build_frames(num_frames_to_build)
        frames1 = np.copy(dqm_obj.frames)
        success = dqm_obj.frames.shape[2] == 240
        self.assertTrue(success, 'build_frames must return expected number of frames')

        # test that all frames are returned when pare_frames is False
        dqm_obj.clear_frames()
        dqm_obj.build_frames(num_frames_to_build, pare_frames=False)
        success = dqm_obj.frames.shape[2] == num_frames_to_build + 1
        self.assertTrue(success, 'build_frames must return all built frames when pare_frames is False')

        # test that passing in frame 0 separately produces same result
        dqm_obj.clear_frames()
        frames2 = dqm_obj.build_frames(num_frames_to_build, dqm_obj.frames)
        success = np.array_equal(frames1, frames2)
        self.assertTrue(success, 'build_frames must produce same result when frame 0 is passed in as a method argument')
    # end method test_build_frames


    def test_build_frames_auto(self):
        dqm_obj = DQM()
        dqm_obj.verbose = False

        rng = np.random.default_rng(948)
        num_rows = 150
        num_cols = 18
        dqm_obj.raw_data = rng.random((num_rows, num_cols))
        dqm_obj.create_frame_0()
        dqm_obj.basis_size = 80
        dqm_obj.choose_basis_by_distance()
        dqm_obj.choose_sigma_for_basis()
        dqm_obj.build_operators()

        # test that building frames in batches produces same result as building them all at once
        dqm_obj.build_frames(400)
        frames1 = np.copy(dqm_obj.frames)
        dqm_obj.clear_frames()
        batch_size = 5
        dqm_obj.build_frames_auto(batch_size)
        frames2 = np.copy(dqm_obj.frames)
        success = np.array_equal(frames1, frames2)
        self.assertTrue(success, 'building frames in batches must produce same result as building all at once')

        # test that building frames in larger batches produces same result
        dqm_obj.clear_frames()
        batch_size = 20
        dqm_obj.build_frames_auto(batch_size)
        frames2 = np.copy(dqm_obj.frames)
        success = np.array_equal(frames1, frames2)
        self.assertTrue(success, 'building frames in larger batches must produce same result as building all at once')
    # end method test_build_frames_auto


    def test_pare_frames(self):
        dqm_obj = DQM()
        dqm_obj.verbose = False

        # test that single frame is returned unchanged
        frames = np.zeros((10, 5, 1))
        frames = dqm_obj.pare_frames(frames)
        success = frames.ndim == 3 and frames.shape[2] == 1 and np.min(frames) == 0 and np.max(frames) == 0
        self.assertTrue(success, 'calling pare_frames on single frame must return same frame')

        # test that 2 identical frames is returned as single frame
        frames = np.concatenate((frames, frames), axis=2)
        frames = dqm_obj.pare_frames(frames)
        success = frames.ndim == 3 and frames.shape[2] == 1 and np.min(frames) == 0 and np.max(frames) == 0
        self.assertTrue(success, 'calling pare_frames on two identical frames must return first frame')

        # test that a single duplicate frame is dropped
        frames = np.random.rand(10, 5, 2)
        frames = np.concatenate((frames, frames[:, :, -1:-1]), axis=2)
        frames = dqm_obj.pare_frames(frames)
        success = frames.ndim == 3 and frames.shape[2] == 2
        self.assertTrue(success, 'pare_frames should return 2 frames')

        # test that multiple duplicate frames are dropped
        frames = np.random.rand(10, 5, 20)
        last_frame = frames[:, :, -1:-1]
        frames = np.concatenate((frames, last_frame, last_frame, last_frame, last_frame), axis=2)
        frames = dqm_obj.pare_frames(frames)
        success = frames.ndim == 3 and frames.shape[2] == 20
        self.assertTrue(success, 'pare_frames should return 20 frames')
    # end method test_pare_frames


    def test_clear_frames(self):
        dqm_obj = DQM()
        dqm_obj.verbose = False

        # test that calling clear_frames does nothing when there are no frames
        dqm_obj.clear_frames()
        success = dqm_obj.frames is None
        self.assertTrue(success, 'clear_frames must do nothing when there are no frames')

        # test that clear_frames keeps first frame and remains 3-D
        dqm_obj.frames = np.zeros((10, 5, 3))  # create an all-zero array
        dqm_obj.frames[0, 0, 0] = 1  # add a single 1 in the first frame
        dqm_obj.clear_frames()
        success = dqm_obj.frames.ndim == 3 and dqm_obj.frames.shape[2] == 1 and dqm_obj.frames[0, 0, 0] == 1
        self.assertTrue(success, 'clear_frames must keep first frame and remain 3-D')
    # end method test_clear_frames


    def test_pca_projection(self):
        dqm_obj = DQM()
        dqm_obj.verbose = False

        rng = np.random.default_rng(988)
        num_rows = 200
        num_cols = 500
        dqm_obj.raw_data = rng.random((num_rows, num_cols))
        dqm_obj.run_pca()
        num_pca_dims = 80

        # test that 'in-sample' subspace projection norm proportions are within expected range
        norm_props = dqm_obj.pca_projection(num_pca_dims=num_pca_dims)
        success = np.min(norm_props) > 0.7 and np.max(norm_props) < 0.9
        self.assertTrue(success, "'in-sample' projection norm proportions must be within expected range")

        # test that 'out-of-sample' subspace projection norm proportions are within expected range
        num_new_rows = 15
        dat_new = rng.random((num_new_rows, num_cols))
        norm_props_new = dqm_obj.pca_projection(dat_new, num_pca_dims=num_pca_dims)
        success = np.min(norm_props_new) > 0.3 and np.max(norm_props_new) < 0.5
        self.assertTrue(success, "'out-of-sample' projection norm proportions must be within expected range")
    # end method test_pca_projection


    def test_run_new_points(self):
        dqm_obj = DQM()
        dqm_obj.verbose = False

        rng = np.random.default_rng(969)
        num_rows = 300
        num_cols = 14
        dqm_obj.raw_data = rng.random((num_rows, num_cols))
        dqm_obj.pca_var_threshold = 0.8
        dqm_obj.create_frame_0()

        dqm_obj.basis_size = 75
        dqm_obj.choose_basis_by_distance()

        dqm_obj.overlap_mean_threshold = 0.8
        dqm_obj.choose_sigma_for_basis()

        dqm_obj.build_operators()
        dqm_obj.build_frames_auto(100)

        new_oos_rows = 20
        dat_raw_oos = rng.random((new_oos_rows, num_cols))

        frames_oos, overlaps_is, overlaps_oos, norm_props_is, norm_props_oos = dqm_obj.run_new_points(dat_raw_oos)

        # test that subspace-projection norm proportions fall within the expected ranges
        success = np.min(norm_props_is) > 0.52 and np.max(norm_props_is) < 0.9995 and \
                  np.min(norm_props_oos) > 0.57 and np.max(norm_props_oos) < 0.99
        self.assertTrue(success,
                    'in-sample and out-of-sample subspace-projection norm proportions must fall within expected ranges')

        # test that overlaps fall within the expected ranges
        success = np.min(overlaps_is) > 0.62 and np.max(overlaps_is) < 0.91 and \
                  np.min(overlaps_oos) > 0.54 and np.max(overlaps_oos) < 0.91
        self.assertTrue(success, "in-sample and out-of-sample overlaps must fall within expected ranges")

        # test that all out-of-sample points wind up in clusters with expected in-sample sizes
        last_frame_all = np.concatenate((dqm_obj.frames[:, :, -1], frames_oos[:, :, -1]), axis=0)
        is_row_nums = list(range(num_rows))
        oos_row_nums = list(range(num_rows, last_frame_all.shape[0]))
        clusters, cluster_sizes = get_clusters(last_frame_all, 0.001)
        # select clusters that have any out-of-sample points
        oos_clusters = [cluster for cluster in clusters if any([x in oos_row_nums for x in cluster])]
        # find count of in-sample rows in each out-of-sample cluster
        oos_cluster_is_counts = [len([x for x in cluster if x in is_row_nums]) for cluster in oos_clusters]
        success = oos_cluster_is_counts == [77, 58, 28, 15, 3, 3, 3, 1]
        self.assertTrue(success, 'all out-of-sample points must be in clusters with expected in-sample sizes')
    # end method test_run_new_points


    def test_run_simple(self):
        rng = np.random.default_rng(9908)
        dat_raw = rng.random((100, 20))

        sigma = 1

        # run using run_simple
        dqm_obj1 = DQM()
        dqm_obj1.verbose = False
        dqm_obj1.run_simple(dat_raw, sigma)

        # run 'manually'
        dqm_obj2 = DQM()
        dqm_obj2.verbose = False
        dqm_obj2.raw_data = dat_raw
        dqm_obj2.sigma = sigma
        dqm_obj2.create_frame_0()
        dqm_obj2.build_operators()
        dqm_obj2.build_frames_auto()

        # test that the 2 runs produce identical frames
        self.assertTrue(np.array_equal(dqm_obj1.frames, dqm_obj2.frames), 'frames must be identical')
    # end function test_run_simple


    def test_python_code(self):
        '''
        run tests to verify that results match when using compiled C++ functions and corresponding all-Python functions
        '''

        dqm_obj1 = DQM()
        dqm_obj1.call_c = True
        dqm_obj1.verbose = False

        dqm_obj2 = DQM()
        dqm_obj2.call_c = False
        dqm_obj2.verbose = False

        rng = np.random.default_rng(997)
        num_rows = 40
        num_cols = 9
        dat_raw = rng.random((num_rows, num_cols))
        dqm_obj1.raw_data = dat_raw
        dqm_obj2.raw_data = dat_raw

        # test that frame 0 matches
        dqm_obj1.create_frame_0()
        dqm_obj2.create_frame_0()
        success = np.allclose(dqm_obj1.frames, dqm_obj2.frames)
        self.assertTrue(success, 'frame 0 must match for call_c=True and call_c=False')

        # test that same basis rows are chosen (choose_basis_by_distance_single_chunk calls
        # ChooseBasisByDistanceC or choose_basis_by_distance_python)
        basis_size = 10
        dqm_obj1.basis_size = basis_size
        dqm_obj1.choose_basis_by_distance()
        dqm_obj2.basis_size = basis_size
        dqm_obj2.choose_basis_by_distance()
        success = dqm_obj1.basis_row_nums == dqm_obj2.basis_row_nums
        self.assertTrue(success, 'basis row numbers must match for call_c=True and call_c=False')

        # test that same value of sigma is chosen (choose_sigma_for_basis calls build_overlaps, which calls
        # BuildOverlapsC or build_overlaps_python)
        dqm_obj1.choose_sigma_for_basis()
        dqm_obj2.choose_sigma_for_basis()
        success = dqm_obj1.sigma == dqm_obj2.sigma
        self.assertTrue(success, 'chosen value of sigma must match for call_c=True and call_c=False')

        # test that overlaps are the same (BuildOverlaps calls BuildOverlapsC or BuildOverlapsMaple)
        overlaps1 = dqm_obj1.build_overlaps()
        overlaps2 = dqm_obj2.build_overlaps()
        success = np.allclose(overlaps1, overlaps2)
        self.assertTrue(success, 'overlaps must match for call_c=True and call_c=False')

        # test that built frames are the same (build_frames calls BuildFramesAutoC or build_frames_python. note
        # that the test of BuildOperators is implicit.)
        dqm_obj1.build_operators()
        dqm_obj1.build_frames_auto()
        dqm_obj2.build_operators()
        dqm_obj2.build_frames_auto()
        success = np.allclose(dqm_obj1.frames, dqm_obj2.frames)
        self.assertTrue(success, 'built frames must match for call_c=True and call_c=False')
    # end method test_python_code


# end class DQMTests


if __name__ == '__main__':
    unittest.main()

