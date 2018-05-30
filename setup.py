import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='ml-scratch',
    version='1.0',
    description='ML From Scratch',
    author='Havan Agrawal',
    author_email='havanagrawal@gmail.com',
    long_description=long_description,
    url='https://github.com/havanagrawal/ml-from-scratch',
    packages=setuptools.find_packages(),
    include_package_data=True,
)
