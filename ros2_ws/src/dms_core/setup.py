from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'dms_core'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join("share",package_name,"config"),glob("config/*.yaml")),
        (os.path.join("share",package_name,"launch"),glob("launch/*.launch.py")),
        (os.path.join("share",package_name,"models"),glob("models/*.onnx")),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hkit',
    maintainer_email='chlwlgh99@gmail.com',
    description='Driver Monitoring System core nodes',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            "perception_node = dms_core.perception_node:main",
            "identity_node = dms_core.identity_node:main",
            "risk_manager_node = dms_core.risk_manager_node:main",
        ],
    },
)
