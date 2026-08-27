import time
import math

from pyAgxArm import (
    create_agx_arm_config,
    AgxArmFactory,
    ArmModel,
    PiperFW,
)

# ============================================================
# 配置
# ============================================================

CAN_CHANNEL = "can0"      # 如果你的接口实际叫 can_piper，就改成 "can_piper"       
SPEED_PERCENT = 10        # 速度：10%
WAIT_TIMEOUT = 30.0       # 等待运动完成最大时间
JOINTS_LIST = [0,1]       # 需要移动的关节
MOVE_DEG =[30,30]         # 对应关节移动的角度
# ============================================================
# 等待运动结束
# ============================================================

def wait_motion_done(robot, timeout=10.0):
    """
    等待机械臂运动结束。
    motion_status == 0 表示运动结束。
    """
    time.sleep(0.5)
    start_time = time.monotonic()
    while time.monotonic() - start_time < timeout:
        status = robot.get_arm_status()
        if status is not None:
            motion_status = getattr(status.msg, "motion_status", None)
            if motion_status == 0:
                return True
        time.sleep(0.1)
    return False


# ============================================================
# 获取有效关节角
# ============================================================
def get_joint_angles(robot, timeout=30.0):
    start_time = time.monotonic()
    while time.monotonic() - start_time < timeout:
        joints = robot.get_joint_angles()
        if joints is not None:
            return list(joints.msg)
        time.sleep(0.1)
    return None


# ============================================================
# 主程序
# ============================================================
def main():
    print("=" * 60)
    print("PIPER-X J1 小角度运动测试")
    print("=" * 60)
    # --------------------------------------------------------
    # 1. 创建 PIPER-X 配置
    # --------------------------------------------------------
    config = create_agx_arm_config(
        robot=ArmModel.PIPER_X,
        firmeware_version=PiperFW.DEFAULT,
        interface="socketcan",
        channel=CAN_CHANNEL,
    )
    robot = AgxArmFactory.create_arm(config)
    # --------------------------------------------------------
    # 2. 连接 CAN
    # --------------------------------------------------------
    print(f"\n[1] 正在连接 {CAN_CHANNEL} ...")
    robot.connect()
    time.sleep(1)
    print("    channel:", robot.get_channel())
    print("    is_ok :", robot.is_ok())
    print("    fps   :", robot.get_fps())
    if not robot.is_ok():
        print("\n[ERROR] CAN 通信异常")
        return
    # --------------------------------------------------------
    # 3. 检查关节状态
    # --------------------------------------------------------
    print("\n[2] 检查六个关节使能状态...")
    enable_status = robot.get_joints_enable_status_list()
    print("    ", enable_status)
    # 如果还没有全部使能，则尝试使能
    if not all(enable_status):
        print("\n[3] 机械臂未全部使能，尝试使能...")
        start_time = time.monotonic()
        while not robot.enable():
            if time.monotonic() - start_time > 5:
                print("[ERROR] 机械臂使能失败")
                return
            time.sleep(0.05)
        time.sleep(1)
        enable_status = robot.get_joints_enable_status_list()
        print("    使能结果:", enable_status)
        if not all(enable_status):
            print("[ERROR] 仍有部分关节未使能")
            return
    else:
        print("    六个关节已经全部使能 ✓")

    # --------------------------------------------------------
    # 4. 设置低速关节运动
    # --------------------------------------------------------
    print("\n[4] 设置运动参数...")
    robot.set_speed_percent(SPEED_PERCENT)
    robot.set_motion_mode(
        robot.OPTIONS.MOTION_MODE.J
    )
    print(f"    速度：{SPEED_PERCENT}%")
    print("    模式：Joint / J")

    # --------------------------------------------------------
    # 5. 获取当前位置
    # --------------------------------------------------------
    print("\n[5] 获取当前关节角...")
    current_joints = get_joint_angles(robot)
    if current_joints is None:
        print("[ERROR] 无法获取关节角")
        return
    print("\n当前关节角：")
    for i, angle in enumerate(current_joints):
        print(
            f"    J{i + 1}: "
            f"{angle: .6f} rad"
            f"    ({math.degrees(angle): .2f}°)"
        )

    # --------------------------------------------------------
    # 6. 生成测试目标
    # --------------------------------------------------------
    target_joints = current_joints.copy()
    delta = [math.radians(num) for num in MOVE_DEG]
    # J1 大致限制 ±150°。
    # 如果已经很靠近正限位，就向负方向测试。
    for i in JOINTS_LIST:
        if current_joints[i] + delta[i] < math.radians(145):
            target_joints[i] += delta[i]
            direction = "+"
        else:
            target_joints[i] -= delta[i]
            direction = "-"
        print("\n测试内容：")
        if i == 0:
            print(
                f"    J1:"
                f" {math.degrees(current_joints[0]):.2f}°"
                f" -> {math.degrees(target_joints[0]):.2f}°"
            )
            print(f"    移动量：{direction}{MOVE_DEG[i]}°")

        elif i == 1:
            print(
                f"    J2:"
                f" {math.degrees(current_joints[1]):.2f}°"
                f" -> {math.degrees(target_joints[1]):.2f}°"
            )
            print(f"    移动量：{direction}{MOVE_DEG[i]}°")
        print(f"    速度：{SPEED_PERCENT}%")



    # --------------------------------------------------------
    # 7. 人工确认
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("请确认：")
    print("1. 机械臂周围没有人员或障碍物")
    print("2. 机械臂固定牢固")
    print("3. 可以随时断电/急停")
    print("=" * 60)
    input("\n确认安全后按 Enter 开始移动...")

    # --------------------------------------------------------
    # 8. 第一次运动：J1 +30°
    # --------------------------------------------------------
    print("\n[6] J1 开始小幅移动...")
    robot.move_j(target_joints)
    if wait_motion_done(robot, WAIT_TIMEOUT):
        print("    运动完成 ✓")
    else:
        print("    [WARNING] 等待运动完成超时")
    time.sleep(0.5)
    joints_after_move = get_joint_angles(robot)
    if joints_after_move is not None:
        print(
            "\n    J1 实际位置："
            f"{math.degrees(joints_after_move[0]):.2f}°"
        )

    # --------------------------------------------------------
    # 9. 等待用户确认后回原位置
    # --------------------------------------------------------
    input("\n按 Enter 让 J1 返回测试前位置...")
    print("\n[7] 返回原位置...")
    robot.move_j(current_joints)
    if wait_motion_done(robot, WAIT_TIMEOUT):
        print("    返回完成 ✓")
    else:
        print("    [WARNING] 返回运动超时")
    time.sleep(0.5)
    final_joints = get_joint_angles(robot)
    if final_joints is not None:
        print(
            "\n    J1 最终位置："
            f"{math.degrees(final_joints[0]):.2f}°"
        )
    print("\n" + "=" * 60)
    print("PIPER-X Joints 运动测试完成")
    print("=" * 60)

    robot.disable() # 关闭使能的时候确保机械臂各个臂不会发生碰撞，因为会关闭各个关节的通电
    robot.disconnect()
    


if __name__ == "__main__":
    main()