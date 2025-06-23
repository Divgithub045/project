from setuptools import find_packages,setup
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    '''
    Returns a list of requirements from the given file.
    Removes the editable install line (-e .) if present.
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.strip() for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    
    return requirements

setup(
    name="ML_Project",
    version='0.0.1',
    author='Divgithub045',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')

    )