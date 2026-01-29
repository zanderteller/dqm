import os
import platform
import ctypes
import numpy as np
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("pydqm")
except PackageNotFoundError:
    __version__ = "unknown"  # Package not installed (running from source)


def load_dqm_lib(lib_path):
    '''
    Load compiled-code library and set up function signatures.

    Note: multiple modules (DQM and utils) use functions from this library

    :param lib_path: Relative or absolute path to compiled-library file. No default.
    :return: The library (a ctypes.CDLL library object)
    '''

    # create the library object
    dqm_lib = ctypes.CDLL(lib_path)

    # set up signature for: int MakeOperatorsC(double* mat, int num_rows, int num_cols, int nhambasis,
    #                                           int npotbasis, double sigma, double step, double mass,
    #                                           double* simt, double* xops, complex<double>* exph)
    dqm_lib.MakeOperatorsC.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.complex128, ndim=2, flags='C_CONTIGUOUS')
    ]
    dqm_lib.MakeOperatorsC.restype = ctypes.c_int32

    # set up signature for: void ChooseBasisByDistanceC(double* rows, int num_rows, int num_cols,
    #                                                   int basis_size, int* basis_row_nums,
    #                                                   int first_basis_row_num)
    dqm_lib.ChooseBasisByDistanceC.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),
        ctypes.c_int32
    ]
    dqm_lib.ChooseBasisByDistanceC.restype = None

    # set up signature for: void BuildOverlapsC(double sigma, double* basis_rows, double* other_rows,
    #                                           int num_basis_rows, int num_other_rows, int num_cols,
    #                                           double* overlaps)
    dqm_lib.BuildOverlapsC.argtypes = [
        ctypes.c_double,
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')
    ]
    dqm_lib.BuildOverlapsC.restype = None

    # set up signature for: void BuildFramesAutoC(double* new_frames, int num_rows, int num_cols,
    #                                           int num_frames_to_build, double* current_frame,
    #                                           double* basis_rows, int num_basis_rows, double* simt,
    #                                           int num_basis_vecs, double* xops, complex<double> exph,
    #                                           double sigma, double stopping_threshold)
    dqm_lib.BuildFramesAutoC.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
        ctypes.c_int32,
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
        ctypes.c_int32,
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.complex128, ndim=2, flags='C_CONTIGUOUS'),
        ctypes.c_double,
        ctypes.c_double
    ]
    dqm_lib.BuildFramesAutoC.restype = None

    # set up signature for
    # void GetClustersC(double* mat, int num_rows, int num_cols, double max_dist, int* dqm_idxs)
    dqm_lib.GetClustersC.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_double,
        np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),
    ]
    dqm_lib.GetClustersC.restype = None

    # set up signature for
    # void NearestNeighborsC(double* mat, int num_rows, int num_cols, int* nn_row_nums, double* nn_dists)
    dqm_lib.NearestNeighborsC.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
        ctypes.c_int32,
        ctypes.c_int32,
        np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')
    ]
    dqm_lib.NearestNeighborsC.restype = None

    return dqm_lib
# end class method load_dqm_lib


# find out if we have the compiled library file
sys_name = platform.system()
if sys_name == 'Windows':
    lib_ext = 'dll'
elif sys_name == 'Linux':
    lib_ext = 'so'
elif sys_name == 'Darwin':  # Mac OS
    lib_ext = 'dylib'
else:
    # we assume there will be no compiled-library file found, below, with a 'none' extension
    lib_ext = 'none'
    print(f"## WARNING: in dqm package -- compiled-library code not implemented for platform '{sys_name}'")
# end if/else (which system platform we're on)

# we expect to find the compiled-library binary in the relative-path folder 'bin'
lib_name = 'dqm_python.{}'.format(lib_ext)
lib_path = os.path.join(os.path.dirname(__file__), 'bin', lib_name)

# load the library, if we have it (will be imported from here by modules)
dqm_lib = None
if os.path.exists(lib_path):
    dqm_lib = load_dqm_lib(lib_path)
# end if compile-library file exists


# make everything in modules importable from the package
# note: these must come after initializing dqm_lib, to avoid circular-import problems
from .DQM import *
from .utils import *

