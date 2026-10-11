from setuptools import find_packages, setup

package_name = 'mowberry_semantic_nav'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='vologu',
    maintainer_email='vologu@todo.todo',
    description='Semantic navigation AI bridge (VLM + LiDAR + Jev) for Nav2.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'semantic_nav_bridge = '
            'mowberry_semantic_nav.semantic_nav_bridge:main',
        ],
    },
)
