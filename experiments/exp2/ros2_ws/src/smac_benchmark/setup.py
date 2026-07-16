import os
from glob import glob

from setuptools import find_packages, setup


package_name = "smac_benchmark"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "LICENSE"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="riku2",
    maintainer_email="nonomura_rikuto[at]stu.kobe-u.ac.jp",
    description="Nav2 Hybrid-A* and Theta* benchmark runner for the Polygonal Proximity Map experiments.",
    license="Apache-2.0",
    extras_require={"test": ["pytest"]},
    entry_points={
        "console_scripts": [
            "benchmark = smac_benchmark.run_benchmark:main",
            "batch_benchmark = smac_benchmark.run_batch_benchmark:main",
        ],
    },
)


