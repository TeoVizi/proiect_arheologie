from setuptools import setup, Extension
import numpy

module = Extension(
    "quantization",
    sources=["quantization.c"],
    include_dirs=[numpy.get_include()],
)

setup(
    name="quantization",
    version="1.0",
    ext_modules=[module],
)
