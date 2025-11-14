from setuptools import setup, find_packages

setup(
    name="mic_dp",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "diffprivlib",
        "scikit-learn",
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "scipy",
        "lifelines",
        "statsmodels"
    ],
    author="Jane Doe",
    author_email="janedoe@somewhere.com",
    description="A Python package for maximum information coefficient differential privacy",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    keywords="differential-privacy mic machine-learning privacy-preserving feature-selection",
    url="",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security :: Cryptography",
        "Development Status :: 4 - Beta"
    ],
    python_requires=">=3.7",
    include_package_data=True,
)
