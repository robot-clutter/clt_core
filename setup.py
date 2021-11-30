from setuptools import setup

setup(name='clt_core',
      description='Reinforcement learning for dealing with clutter',
      version='0.1',
      author='Iason Sarantopoulos',
      author_email='iasons@auth.gr',
      install_requires=['numpy==1.19.5', \
                        'sphinx', \
                        'sphinxcontrib-bibtex==2.1.4', \
                        'sphinx_rtd_theme==0.5.1', \
                        'numpydoc==1.1.0', \
                        'pybullet==3.0.8', \
                        'opencv-python==4.5.1.48', \
                        'matplotlib==3.3.3', \
                        'scipy==1.6.0', \
                        'open3d==0.12.0', \
                        'GitPython==3.1.18', \
                        'PyYaml'\
      ]
)