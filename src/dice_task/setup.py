from glob import glob

from setuptools import find_packages, setup

package_name = "dice_task"

setup(
    name=package_name,
    version="1.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Giacomo Demetrio",
    maintainer_email="giacomodemetrio@gmail.com",
    description="State machine that re-orients a die until a requested face is up.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "dice_task_node = dice_task.dice_task_node:main",
        ],
    },
)
