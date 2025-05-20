from setuptools import find_packages, setup
from glob import glob


package_name = 'frontier_explorer'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),  #added
        ('share/' + package_name + '/param', glob('param/*.yaml')),         #added
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='stan',
    maintainer_email='stan@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'explorer = frontier_explorer.explorer:main',
        ],
    },
)
