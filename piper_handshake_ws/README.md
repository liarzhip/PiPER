## 这是关于PIPER_X机械臂握手的项目介绍
### 0. 主链路
```sh
DaBai DC1 # Orbbec 相机采集深度和彩色图像
   ↓
hand_vision_node # 来自“piper_hand_follow/vision_node.py”，但直接被handshake.launch.py直接启动，使用MediaPipe计算手掌3D位姿
   ↓
/handshake/palm_pose_camera # 
   ↓
stability_detector # 判断手是否稳定（连续30帧），稳定则冻结（后续不再接受）此刻手掌pose
   ↓
/handshake/locked_pose_camera
   ↓
locked_pose_transformer # 将手掌的Pose从相机的坐标系转换到机械臂的base_link坐标系
   ↓
/handshake/locked_pose_base
   ↓
handshake_planner # 根据手掌位置和姿态计算接近的位置和最终目标的Pose
   ↓
/handshake/approach_pose
/handshake/handshake_pose
   ↓
moveit_auto_handshake_controller # 使用MoveIt自动规划执行和到位检查返回Home
   ↓
MoveIt2
   ↓
PIPER-X
   ↓
Approach → Handshake → Home
```

### 1. hand_vision_node
负责整个项目最前面的视觉：
```sh
DaBai DC1
   ↓
pyorbbecsdk
   ↓
RGB + Depth
   ↓
MediaPipe Hands
   ↓
手部关键点 （包括腕部Wrist，食指Index指关节MCP，中指Middle指关节MCP，无名指Ring指关节MCP，小指Pinky指关节MCP）
```
通过以上手部关键点，返回，Palm Center 和 X Y Normal，最终得到完整的人的手掌在相机的哪里，朝什么方向[x, y, z, qx, qy, qz, qw]，并发布在/handshake/palm_pose_camera

### 2. stability_detector.py
订阅/handshake/palm_pose_camera的手掌Pose信息。满足位置的变动方差和方向变化在一定范围以内时，就把这一刻的Pose冻结，一旦锁定后，后面手再动也不会修改已经锁定的目标。\
同时还提供了/handshake/reset服务，可以重新判断冻结的目标：
```sh
ros2 service call \
  /handshake/reset \
  std_srvs/srv/Trigger \
  "{}"
```

### 3. locked_pose_transformer.py
通过订阅/handshake/palm_pose_camera获得手掌的位置，然后通过TF完成
$${}^{base}T_{palm} = {}^{base}T_{camera}\;{}^{camera}T_{palm}$$
最后将转换后的手掌基于机械臂的base_link的坐标系的空间坐标发布在/handshake/locked_pose_base

### 4. handere_static_tf
不是一个独立的python文件，而是在handshake.launch.py中的一个功能。通过创建tf2_ros/static_transform_publisher，具体而言：读取config/handeye_calibration.json（里面就是之前ArUco手眼标定得到的参数），将相机的坐标与Base坐标相连，使得TF树完整。

### 5. handshake_planner.py
通过订阅/handshake/locked_pose_base获得冻结手掌在空间的坐标信息。具体规划如下：\
（1）保证夹爪的Z轴（夹爪的朝向）始终指向手的位置，然后Y轴（夹爪开合的方向）始终与手掌的法向重合。\
（2）发布两个位置：距离手掌15厘米的Approach pose和目标Handshake Pose。\
（3）并且发布/handshake/plan_ready = true。

### 6. moveit_auto_handshake_controller.py
主要负责关节规划和动作：
```sh
WAIT_TARGET
   ↓
PLAN_APPROACH # 先获得/handshake/approach_pose
   ↓
EXECUTE_APPROACH # Approach速度155
   ↓
VERIFY_APPROACH #（那自身当前的姿态，计算与之前计算的目标姿态在同一个base_link坐标系下是否重合）
   ↓
PLAN_HANDSHAKE #先获得/handshake/handshake_pose
   ↓
EXECUTE_HANDSHAKE # Handshake速度5%
   ↓
VERIFY_HANDSHAKE
   ↓
DWELL
   ↓
PLAN_HOME # 最开始会读取joint的位置信息并保存为home_joint_positions
   ↓
EXECUTE_HOME
   ↓
VERIFY_HOME
   ↓
DONE
```
## 终端执行命令
### 1. 环境配置
```sh

cd ~/PiPER_X/piper_handshake_ws

source /opt/ros/jazzy/setup.bash
source ~/PiPER_X/agx_arm_ws/install/setup.bash
source ~/PiPER_X/piper_hand_follow_ws/install/setup.bash
```

### 2.启动PIPER+Movit
```sh
ros2 launch agx_arm_ctrl start_single_agx_arm_moveit.launch.py \
  can_port:=can0 \
  arm_type:=piper_x \
  effector_type:=agx_gripper \
  fw_version:=v189 \
  auto_enable:=true \
  follow:=true \
  speed_percent:=15
```
### 3. 启动握手项目
```sh
ros2 launch piper_handshake handshake.launch.py
```

### 4. 握手结束后可重新重新冻结启动
```sh
ros2 service call \
  /handshake/reset \
  std_srvs/srv/Trigger \
  "{}"

```