from setuptools import setup

package_name = "prediction"

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
            "prediction_node = prediction.ball_fitting_predict_node:main",
            "prediction_kalman_node = prediction.ball_fitting_predict_kalman_node:main",
            "prediction_simple_node = prediction.ball_prediction_node:main",
            "prediction_simple_node_v2 = prediction.ball_prediction_node_v2:main",
        ],
    },
)
