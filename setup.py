try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'name': 'digAdsClassifier',
    'description': 'digAdsClassifier',
    'author': 'Rahul Kapoor',
    'url': 'https://github.com/usc-isi-i2/dig-ads-classifier',
    'download_url': 'https://github.com/usc-isi-i2/dig-ads-classifier',
    'author_email': 'rahulkap@isi.edu',
    'version': '0.1.0',
    'install_requires': ['digExtractor>=0.2.0'],
    # these are the subdirs of the current directory that we care about
    'packages': ['digAdsClassifier'],
    'scripts': [],
}

setup(**config)
