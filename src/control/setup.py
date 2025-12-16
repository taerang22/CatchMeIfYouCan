from setuptools import find_packages, setup
from pathlib import Path

package_name = 'control'

setup(
    name=package_name,
    version='0.0.2',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
        'setuptools',
        'numpy',
        'scipy',
    ],
    zip_safe=True,
    maintainer='iconlab',
    maintainer_email='dayiethan@gmail.com',
    description='Kinova controller nodes for goal pose following and home return.',
    license='BSD-3-Clause',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'control_node = control.kinova_controller_node:main',
            'mpc_controller = control.kinova_mpc_controller:main',
            'random_goal_publisher = control.random_goal_publisher:main',
        ],
    },
)
