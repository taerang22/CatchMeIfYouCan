from setuptools import setup

package_name = "perception"

setup(
    name=package_name,
    version="0.0.2",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools", "numpy", "scipy", "opencv-python"],
    zip_safe=True,
    maintainer="iconlab",
    maintainer_email="iconlab@example.com",
    description="RealSense ball tracker",
    license="BSD-3-Clause",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "perception_node = perception.ball_tracker_node:main",
        ],
    },
)
