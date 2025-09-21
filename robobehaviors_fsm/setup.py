from setuptools import find_packages, setup

package_name = "robobehaviors_fsm"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/fsm_launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="jun",
    maintainer_email="cpark1@olin.edu",
    description="Behavior FSM with cmd_vel multiplexing.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "teleop = robobehaviors_fsm.teleop:main",
            "wall_follower = robobehaviors_fsm.wall_follower:main",
            "person_follower = robobehaviors_fsm.person_follower:main",
            "drive_square = robobehaviors_fsm.drive_square:main",
            "fsm = robobehaviors_fsm.fsm_manager:main",
        ],
    },
)
