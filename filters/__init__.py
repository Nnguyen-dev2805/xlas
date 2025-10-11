"""
Filters Package for Image Processing
===================================

Tập hợp các bộ lọc và kernel cho xử lý ảnh
- Blur filters (Gaussian, Box, Motion blur)
- Edge detection (Sobel, Laplacian, Prewitt)
- Sharpening filters
- Custom kernels
- Comprehensive kernel types
- Convolution engine
- Kernel demonstration tools

Author: Image Processing Team
"""

from .blur_filters import *
from .edge_detection import *
from .sharpen_filters import *
from .kernels import *
from .kernel_types import KernelGenerator, KernelScaler
from .convolution_engine import ConvolutionEngine, MultiKernelProcessor, KernelAnalyzer

__version__ = "1.0.0"
