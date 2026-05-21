# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

from setuptools import setup
from torch_musa.utils.musa_extension import BuildExtension, MUSAExtension

setup(
    name = 'curope',
    ext_modules = [
        MUSAExtension(
                name='curope',
                sources=[
                    "curope.cpp",
                    "kernels.mu",
                ],
                extra_compile_args = dict(
                    mcc=['-O3'],
                    cxx=['-O3'])
                )
    ],
    cmdclass = {
        'build_ext': BuildExtension
    })
