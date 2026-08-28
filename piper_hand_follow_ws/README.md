# PIPER-X + DaBai DC1 手掌跟随（V1）

## 架构

DaBai DC1 RGB+Depth（D2C对齐）
→ MediaPipe Hands（0/5/9/13/17关键点求掌心）
→ 11×11深度中值
→ 相机坐标 XYZ
→ `/hand/pose_camera`
→ TF(camera→base)
→ 偏移/滤波/死区/工作空间
→ `/control/move_p`
→ agx_arm_ros
→ PIPER-X

## 重要假设

V1 默认 **Eye-to-Hand：相机固定在机械臂外部**。
真实运动前必须标定并提供 `base_link <- camera_color_optical_frame` 的 TF。

如果你的 DaBai DC1 是装在手腕/末端上的，请不要直接启用运动；需要改成 Eye-in-Hand 的 TF 链。

## 安装 MediaPipe

```bash
pip3 install "mediapipe==0.10.21" --break-system-packages
```

## 编译

```bash
cd ~/piper_hand_follow_ws
source /opt/ros/jazzy/setup.bash
source ~/agx_arm_ws/install/setup.bash
export PYTHONPATH="$HOME/PiPER_X/orbbec/pyorbbecsdk/install/lib:$PYTHONPATH"
colcon build --symlink-install
source install/setup.bash
```

## 阶段 1：只测试视觉（推荐先做）

```bash
source ~/piper_hand_follow_ws/scripts/setup_env.sh
ros2 run piper_hand_follow hand_vision_node
```

另一个终端：

```bash
source ~/piper_hand_follow_ws/scripts/setup_env.sh
ros2 topic echo /hand/pose_camera
```

移动手，确认 X/Y/Z 连续、方向合理、Z 是真实手掌距离。

## 阶段 2：加入 TF，但不控制机械臂

完成外参/手眼标定后发布真实 TF。下面数值仅示意，不可直接照抄：

```bash
ros2 run tf2_ros static_transform_publisher   --x 0.30 --y 0.00 --z 0.50   --qx 0.0 --qy 0.0 --qz 0.0 --qw 1.0   --frame-id base_link   --child-frame-id camera_color_optical_frame
```

验证：

```bash
ros2 run tf2_ros tf2_echo base_link camera_color_optical_frame
```

然后：

```bash
ros2 launch piper_hand_follow hand_follow.launch.py enable_motion:=false
```

此时只打印目标 TCP，不向 PIPER-X 发命令。

## 阶段 3：真机跟随

先启动 PIPER-X 驱动：

```bash
source /opt/ros/jazzy/setup.bash
source ~/agx_arm_ws/install/setup.bash

ros2 launch agx_arm_ctrl start_single_agx_arm.launch.py   can_port:=can0   arm_type:=piper_x   fw_version:=v189   auto_enable:=true
```

确认：

```bash
ros2 topic echo /feedback/tcp_pose --once
```

再启动跟随：

```bash
source ~/piper_hand_follow_ws/scripts/setup_env.sh
ros2 launch piper_hand_follow hand_follow.launch.py enable_motion:=true
```

## 安全默认值

- `enable_motion=false`
- 控制 3 Hz
- 20 mm 死区
- 单次目标最大变化 20 mm
- 0.4 s 丢手超时
- 保持当前 TCP 姿态
- 默认偏移：手掌上方 +0.25 m（base Z）
- 工作空间越界直接拒绝

参数位于：

`src/piper_hand_follow/config/hand_follow.yaml`

> `/control/move_p` 适合这个低频演示版本，不是高带宽视觉伺服。
> 后续若要更丝滑，应改为速度/伺服控制而不是高频连续 MoveP。
