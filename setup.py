from setuptools import setup, find_packages

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()


setup(
    name='parallel-pandas',
    python_requires='>=3.7',
    version='0.3.6',
    packages=find_packages(),
    author='Dubovik Pavel',
    author_email='geometryk@gmail.com',
    description='Parallel processing on pandas with progress bars',
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords=[
        'parallel pandas',
        'progress bar',
        'parallel apply',
        'parallel groupby',
        'multiprocessing bar',
    ],
    license='MIT',
    install_requires=[
        'pandas >= 1.4.0',
        'dill',
        'psutil',
        'tqdm'
    ],
    platforms='any'
)
