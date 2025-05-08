from setuptools import Extension, setup
from numpy import get_include

setup(
    ext_modules=[
        Extension(
            name="zxy9_biomechanical_model.sim_c_util",
            sources=[ \
				"src/zxy9_biomechanical_model/sim_c_util.c", \
			],
			include_dirs=[get_include()]
        ),
    ],
)
