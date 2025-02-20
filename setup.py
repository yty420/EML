
from setuptools import setup, find_packages


setup(
    name='EML',
    version='0.1',  # 版本号，请根据实际情况修改
    packages=find_packages(),
    install_requires=[
        'pandas>=1.0.0',
        'numpy>=1.18.0',
        'scikit-learn>=0.23.0',
        'imbalanced-learn>=0.7.0',
        'xgboost>=1.2.0',
        'matplotlib>=3.2.0',
        'seaborn>=0.11.0',
        'joblib>=0.16.0',
    ],
    author="Tingyu Yang",
    author_email="787260442@qq.com",
    description="easy machine learning",
    keywords="machine learning predict feature select",
    url="https://github.com/yty420", 
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',  
)
