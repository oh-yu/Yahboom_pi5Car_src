from setuptools import setup
import os
from glob import glob

package_name = 'yahboomcar_rl'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob(os.path.join('launch', '*launch.py'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='H.OSHIMA',
    maintainer_email='oshima@todo.todo',
    description='Simple RL (goal-reaching) example for Yahboom pi5 car in Gazebo.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'train = yahboomcar_rl.train:main',
            'evaluate = yahboomcar_rl.evaluate:main',
            'random_agent = yahboomcar_rl.random_agent:main',
            'demo_reach = yahboomcar_rl.demo_reach:main',
        ],
    },
)
