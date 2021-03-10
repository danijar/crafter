import setuptools
import pathlib


setuptools.setup(
    name='crafter',
    version='0.1.0',
    description='Open world survival environment for reinforcement learning research.',
    url='http://github.com/danijar/crafter',
    long_description=pathlib.Path('README.md').read_text(),
    long_description_content_type='text/markdown',
    install_requires=['numpy', 'imageio', 'gym'],
    packages=['crafter'],
    package_data={'handout': ['asset/*']},
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
)
