from setuptools import setup

setup(
    name='optionCombo',
    version='0.1.1',
    description='this is a library for finding all the possible option combinations of a single asset.',
    url='https://github.com/556isback/optionCombo',
    author='ZHI LIANG CHEN',
    author_email='theon556isback@gmail.com',
    license='MIT',
    packages=['optionCombo'],
    install_requires=[
        'matplotlib',
        'numpy',
        'pandas',
        'py_vollib',
        'py_vollib_vectorized',
        'seaborn',
        'tqdm'
    ],
    zip_safe=False
)