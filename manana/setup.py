from setuptools import setup, find_packages 

#List of requirements\
requirements = ['autograd>=1.3', 'scipy >=1.7.3', 
                'pandas>=1.3.5','cvxopt>=1.2.3','pymanopt>=0.2.5',
                'scikit-learn>=1.0.2','numpy>=1.17.3',
                'torch','torchvision>=0.13','torchaudio','matplotlib']

setup(
	name = "manana",
	version = "1.0.0",
	description = "Manifold Analysis Code from Chi-Ning Chou for AC2023 Group Project WM-Manifolds",
	packages = find_packages(), 
	install_requires = requirements)