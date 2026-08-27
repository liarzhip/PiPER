import time

from pyAgxArm import (
    create_agx_arm_config,
    AgxArmFactory,
    ArmModel,
    PiperFW,
)

config = create_agx_arm_config(
        robot=ArmModel.PIPER_X,
        firmeware_version=PiperFW.DEFAULT,
        interface="socketcan",
        channel="can0",
    )
robot = AgxArmFactory.create_arm(config)
robot.connect()
time.sleep(3)
print("firmware:", robot.get_firmware(timeout=3.0))