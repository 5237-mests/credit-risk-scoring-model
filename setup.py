from setuptools import setup, find_packages

setup(
    name='credit_risk_model',
    version='0.1.0',
    description='Credit Risk Probability Model for Bati Bank - KAIM Week 5',
    author='Mesfin Mulugeta',
    author_email='msfnmulgeta@gmail.com',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'pytest',
        'flake8',
    ],
    include_package_data=True,
    python_requires='>=3.8',
)
