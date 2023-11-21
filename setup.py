from setuptools import setup, find_packages

setup(
    name="DataAnalysisLibrary",
    version="0.3.0",
    description="The goal of the 'DataAnalysisLibrary' is to provide helpful libraries for performing data analysis.",
    author="Shin Chungseop",
    author_email="tooha289@naver.com",
    url="https://github.com/tooha289/DataAnalysisLibrary",
    packages=find_packages(),
    install_requires=[
        'matplotlib',
        'seaborn',
        'numpy',
        'pandas',
        'statsmodels',
        'scikit-learn'
    ]
)
