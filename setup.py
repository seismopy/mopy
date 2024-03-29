"""
Setup script for mopy.
"""
import glob
import sys
from collections import defaultdict
from os.path import join, exists, isdir
from pathlib import Path


from setuptools import setup

PACKAGE_NAME = "mopy"

# define python versions

python_version = (3, 7)  # tuple of major, minor version requirement
python_version_str = str(python_version[0]) + "." + str(python_version[1])

# produce an error message if the python version is less than required
if sys.version_info < python_version:
    msg = f"{PACKAGE_NAME} only runs on python version >= {python_version}"
    raise Exception(msg)

# get path references
here = Path(__file__).absolute().parent
version_file = here / PACKAGE_NAME / "version.py"

# --- get version
with version_file.open() as fi:
    content = fi.read().split("=")[-1].strip()
    __version__ = content.replace('"', "").replace("'", "")

# --- get readme
with open("README.md") as readme_file:
    readme = readme_file.read()


# --- get sub-packages
def find_packages(base_dir="."):
    """setuptools.find_packages wasn't working so I rolled this"""
    out = []
    for fi in glob.iglob(join(base_dir, "**", "*"), recursive=True):
        if isdir(fi) and exists(join(fi, "__init__.py")):
            out.append(fi)
    out.append(base_dir)
    return out


# --- requirements paths


def read_requirements(path):
    """Read a requirements.txt file, return a list."""
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r") as fi:
        return fi.readlines()


package_req_path = here / "requirements.txt"
test_req_path = here / "tests" / "requirements.txt"
doc_req_path = here / "docs" / "requirements.txt"
# read requirement files
install_requires = read_requirements(package_req_path)
tests_require = read_requirements(test_req_path)
docs_require = read_requirements(doc_req_path)

# create extra requires dict or None
extra_req_dict = {"dev": tests_require + docs_require + install_requires}


setup(
    name="mopy",
    version=__version__,
    description="package to calc source params",
    long_description=readme,
    author="Derrick Chambers, Shawn Boltz, James Holt",
    author_email="djachambeador@gmail.com",
    url="https://github.com/seismopy/mopy",
    packages=find_packages("mopy"),
    package_dir={"mopy": "mopy"},
    include_package_data=True,
    license="GNU Lesser General Public License v3.0 or later (LGPLv3.0+)",
    zip_safe=False,
    keywords="seismology",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering",
    ],
    test_suite="tests",
    install_requires=install_requires,
    tests_require=tests_require,
    extras_require=extra_req_dict,
    python_requires=">=%s" % python_version_str,
)
