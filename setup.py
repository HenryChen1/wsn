#!/usr/bin/env python

"""setup that compiles C++ wrapper to be used as python module.
"""

from distutils.core import setup, Extension
import os

# this project uses c++11 codes
os.environ["CC"] = "g++-5 -std=c++11"

swig_opts   = ['-shadow', '-c++']
source_path = 'cc/'
build_path  = 'build/temp.win-amd64-3.7/Release/'+source_path

modified_pso      = Extension('cc._modified_pso',
                              sources=[source_path+'modified_pso.cc',
                                       source_path+'regions.cc',
                                       source_path+'individual.cc',
                                       source_path+'optimizer.cc',
                                       source_path+'modified_pso.i'],
                              swig_opts=swig_opts)
pso               = Extension('cc._pso',
                              sources=[source_path+'pso.cc',
                                       source_path+'pso.i'],
                              extra_objects=[build_path+'regions.obj',
                                             build_path+'individual.obj',
                                             build_path+'optimizer.obj'],
                              swig_opts=swig_opts)
genetic_algorithm = Extension('cc._genetic_algorithm',
                              sources=[source_path+'genetic_algorithm.cc',
                                       source_path+'genetic_algorithm.i'],
                              extra_objects=[build_path+'regions.obj',
                                             build_path+'individual.obj',
                                             build_path+'optimizer.obj'],
                              swig_opts=swig_opts)

ecca               = Extension('cc._ecca',
                              sources=[source_path+'ecca.cc',
                                       source_path+'ecca.i'],
                              extra_objects=[build_path+'regions.obj',
                                             build_path+'individual.obj',
                                             build_path+'optimizer.obj'],
                              swig_opts=swig_opts)

# compile C++ libraries
setup (name        = 'Optimizers',
       version     = '1.0',
       author      = "Jean Da Rolt",
       description = """C++ wrappers for performance optimization.""",
       ext_modules = [modified_pso, genetic_algorithm, pso, ecca],
       py_modules  = ["modified_pso", "genetic_algorithm", "pso", "ecca"],
       )
