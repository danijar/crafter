import setuptools
import pathlib


setuptools.setup(
    name='crafter',
    version='1.8.0',
    description='Open world survival game for reinforcement learning.',
    url='http://github.com/danijar/crafter',
    long_description=pathlib.Path('README.md').read_text(),
    long_description_content_type='text/markdown',
    packages=['crafter'],
    package_data={'crafter': ['data.yaml', 'assets/*']},
    entry_points={'console_scripts': ['crafter=crafter.run_gui:main']},
    install_requires=[
        'numpy', 'imageio', 'pillow', 'opensimplex', 'ruamel.yaml',
        # Numba is an optional dependency but we want it installed by default
        # because it speeds up world generation by ~5x.
        'numba',
    ],
    extras_require={'gui': ['pygame']},
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Topic :: Games/Entertainment',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
