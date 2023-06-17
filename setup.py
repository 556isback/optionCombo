from setuptools import setup

setup(
    name='optionCombo',
    version='0.1.0',
    description='this is a packgage to scan the possible option combo',
    url='https://github.com/your_username/your_package_name',
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