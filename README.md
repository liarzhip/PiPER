# 这里是整个文件夹的项目总揽
## Python SDK:pyAgxArm控制
### 1. 激活CAN
```sh
sudo ip link set can0 up type can bitrate 1000000
```
### 2. 测试关节移动控制
```sh
cd ~/PiPER_X/pyAgxArm/pyAgxArm/demos/piper_x
python3 my_move_test.py
```

## ROS2 控制
### 1. 启动
#### 1.1. 只启动PIPER_X驱动
```sh
ros2 launch agx_arm_ctrl start_single_agx_arm.launch.py \
  can_port:=can0 \
  arm_type:=piper_x \
  effector_type:=agx_gripper \
  fw_version:=v189 \
  auto_enable:=true \
  tcp_offset:='[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]'
```

#### 1.2. 启动驱动+RViz
```sh
ros2 launch agx_arm_ctrl start_single_agx_arm_rviz.launch.py \
  can_port:=can0 \
  arm_type:=piper_x \
  fw_version:=v189 \
  follow:=true \
  control:=false\
  effector_type:=agx_gripper
  tcp_offset:='[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]'
```

#### 1.3. 最高层启动——真机驱动 + MoveIt2 + RViz
```sh
ros2 launch agx_arm_ctrl start_single_agx_arm_moveit.launch.py \
  can_port:=can0 \
  arm_type:=piper_x \
  fw_version:=v189 \
  effector_type:=agx_gripper\
  auto_enable:=true
  tcp_offset:='[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]'
```

(1)将会产生以下话题：
```sh
ros2 topic list

/control/joint_states # → 直接关节目标
/control/move_c # → 圆弧运动
/control/move_j # → 关节运动
/control/move_js
/control/move_l # → 直线运动
/control/move_mit
/control/move_p # → 点到点运动
/feedback/arm_status # 返回整个机械臂的状态
/feedback/gripper_status # 返回夹爪状态
/feedback/joint_states # 返回六个关节状态
/feedback/leader_joint_states # 返回
/feedback/tcp_pose # 返回末端工具中心点的位姿
/parameter_events
/rosout
```
根据关节坐标进行控制，例如:\
直接将Joint1旋转到0.13rad
```sh
ros2 topic pub /control/joint_states \
sensor_msgs/msg/JointState \
"{name: ['joint1'], position: [0.13], velocity: [], effort: []}" -1
```

(2)将会产生如下的服务：
```sh
ros2 service list

/agx_arm_ctrl_single_node/describe_parameters
/agx_arm_ctrl_single_node/get_parameter_types
/agx_arm_ctrl_single_node/get_parameters
/agx_arm_ctrl_single_node/get_type_description
/agx_arm_ctrl_single_node/list_parameters
/agx_arm_ctrl_single_node/set_parameters
/agx_arm_ctrl_single_node/set_parameters_atomically
/control_enable
/emergency_stop
/enable_agx_arm
/move_home
```
可以通过申请服务进行控制，例如：
使能与失能
```sh
ros2 service call /enable_agx_arm \
std_srvs/srv/SetBool "{data: true}"

ros2 service call /enable_agx_arm \
std_srvs/srv/SetBool "{data: false}"
```
### 2.官方 MoveJ / MoveP / MoveL 测试
cd ~/PiPER_X/agx_arm_ws/agx_arm_ws/src/agx_arm_ros

#### (1) 关节测试
```sh
ros2 topic pub /control/move_j \
sensor_msgs/msg/JointState \
"$(cat test/piper/test_move_j.yaml)" -1
```
#### （2）点到点测试
```sh
ros2 topic pub /control/move_p \
geometry_msgs/msg/PoseStamped \
"$(cat test/piper/test_move_p.yaml)" -1
```
#### (3) 直线测试
```sh
ros2 topic pub /control/move_l \
geometry_msgs/msg/PoseStamped \
"$(cat test/piper/test_move_l.yaml)" -1
```

#### （4）圆弧测试
```sh
ros2 topic pub /control/move_c \
geometry_msgs/msg/PoseArray \
"$(cat test/piper/test_move_c.yaml)" -1
```

