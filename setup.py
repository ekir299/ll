from setuptools import setup, find_packages

setup(
    name='careless',
    version='0.1.0',
    author='Kevin M. Dalton',
    author_email='kmdalton@fas.harvard.edu',
    packages=find_packages(),
    description='Scaling and merging crystallographic data with TensorFlow and Variational Inference',
    install_requires=[
        "reciprocalspaceship>=0.9.1",
        "tqdm",
        "tensorflow>=2.4",
        "tensorflow-probability",
        "numpy",
    ],
    scripts = [
            'careless/careless',
    ],
    setup_requires=['pytest-runner'],
    tests_require=['pytest', 'pytest-cov', 'pytest-xdist'],
)
