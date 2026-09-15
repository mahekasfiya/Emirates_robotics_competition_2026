import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'erc_vision'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'maps'), glob('maps/*')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
                'column_detector = erc_vision.column_detector:main',
                'book_color_detector = erc_vision.book_color_detector:main',
                'tuck_arms_once = erc_vision.tuck_arms_once:main',
                'set_initial_pose = erc_vision.set_initial_pose:main',
                'approach_column = erc_vision.approach_column:main',
                'grasp_book = erc_vision.grasp_book:main',
                'return_and_place = erc_vision.return_and_place:main',
        ],
    },
)
