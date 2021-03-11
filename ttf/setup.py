from setuptools import setup

setup(
    name='tfm',
    description='Thematic Fit Models',
    author=[''],
    author_email=[''],
    version='0.1.0',
    license='MIT',
    packages=['tfm', 'tfm.logging_utils', 'tfm.utils', 'tfm.core'],
    package_data={'tfm': ['logging_utils/*.yml']},
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'tfm = tfm.main:main'
        ],
    },
    install_requires=['pyyaml==4.2b1', 'numpy==1.19.2', 'tqdm==4.45',
                        'scipy==1.4.1', 'pandas==0.23.0', 'scikit-learn>=0.23.1',
                        'pytokenizations==0.7.2', 'transformers==4.2.2', 'sentencepiece==0.1.95',
                        'tensorflow==2.4.1'],
)
