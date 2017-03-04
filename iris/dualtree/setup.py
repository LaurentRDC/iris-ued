from setuptools import setup, find_packages
from os.path import dirname, join

# Read the license file into a single string
with open(join(dirname(__file__), 'LICENSE.txt')) as license_file:
    lic = license_file.read().replace('\n', '')

setup(name = 'dualtree',
      version = '0.7.6',
      description = 'Dual-tree complex wavelet transform and associated algorithms',
      author = 'Laurent P. René de Cotret',
      author_email = 'laurent.renedecotret@mail.mcgill.ca',
      url = 'http://www.physics.mcgill.ca/siwicklab/software.html',
      packages = find_packages(),
      install_requires = ['numpy', 'PyWavelets >= 0.5.1'],
      package_data = {'dualtree': ['*.txt'],
                      'dualtree': ['data/*.npz']},
      license = lic
     )