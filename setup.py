from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = '-e .'

def get_requirements(file_path:str)->List[str]:
    '''
    This function returns a list of requirements.
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]# Because \n is also captured.

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements

# Metadata of the entire project
setup(
name='mlproject',
version="0.0.1",
author='Hao',
author_email="tayxhwork@gmai.com",
packages=find_packages(), # Finds the packages in __init__.py and build it
install_requires=get_requirements('requirements.txt') # Required libraries 
)