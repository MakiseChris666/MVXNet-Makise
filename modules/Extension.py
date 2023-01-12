from torch.utils.cpp_extension import load

cpp = load('voxelutil', './cpp/voxelutil.cpp')