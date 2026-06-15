from setuptools import Command, find_packages, setup

__lib_name__ = "SpaTemporal"
__lib_version__ = "1.0.0"
__description__ = "SpaTemporal: a deep temporal-aware framework for the integration of spatiotemporal transcriptomics"
__url__ = "https://github.com/JinyunNiu/SpaTemporal"
__author__ = "Jinyun Niu"
__author_email__ = "niujinyun@aliyun.com"
__license__ = "MIT"
__keywords__ = ["spatiotemporal transcriptome", "spatiotemporal graph neural network", "data integrationr", "spatiotemporal domain"]
__requires__ = ["requests",]

setup(
    name = __lib_name__,
    version = __lib_version__,
    description = __description__,
    url = __url__,
    author = __author__,
    author_email = __author_email__,
    license = __license__,
    packages = ['SpaTemporal'],
    install_requires = __requires__,
    zip_safe = False,
    include_package_data = True,
)