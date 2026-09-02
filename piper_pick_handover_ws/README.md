## 这是PiPER机械臂自动抓取和摆放耳机项目的介绍
### 结构介绍

### 完整测试指令
#### 1. 启动Piper机械臂驱动
```sh
source /opt/ros/jazzy/setup.bash
source ~/PiPER_X/agx_arm_ws/install/setup.bash
source ~/PiPER_X/piper_pick_handover_ws/install/setup.bash

ros2 launch agx_arm_ctrl start_single_agx_arm.launch.py \
  can_port:=can0 \
  arm_type:=piper_x \
  effector_type:=agx_gripper \
  fw_version:=v189 \
  auto_enable:=true \
  tcp_offset:='[0.0,0.0,0.0,0.0,0.0,0.0]'
```
#### 2. 启动MoveIt
```sh

ros2 launch agx_arm_ctrl start_single_agx_arm_rviz.launch.py \
  can_port:=can0 \
  arm_type:=piper_x \
  fw_version:=v189 \
  follow:=true \
  control:=false \
  effector_type:=agx_gripper
```

#### 3. 启动视觉
```sh
ros2 launch piper_pick_handover scene_perception.launch.py
```

#### 4. 启动任务节点
```sh
ros2 launch \
piper_pick_handover \
pick_handover_step7e.launch.py
```

#### 5. 识别完成后，移动到目标上方
```sh
ros2 service call \
/moveit/execute_pregrasp \
std_srvs/srv/Trigger \
"{}"
```
#### 6.完成抓取
```sh
ros2 service call \
/moveit/execute_grasp \
std_srvs/srv/Trigger \
"{}"
```

#### 7. Lift夹爪
```sh
ros2 service call \
/moveit/execute_lift \
std_srvs/srv/Trigger \
"{}"
```

#### 8. 夹爪朝向手掌（第一次定位）位置进行第二次具体定位
```sh
ros2 service call \
/handover/recompute_observe_hand \
std_srvs/srv/Trigger \
"{}"
```

#### 9. 规划到手掌位置
```sh
ros2 service call \
/moveit/plan_observe_hand \
std_srvs/srv/Trigger \
"{}"
```

#### 10. 执行到手掌位置
```sh
ros2 service call \
/moveit/execute_observe_hand \
std_srvs/srv/Trigger \
"{}"
```
### 干净的编译
```sh
cd ~/PiPER_X/piper_pick_handover_ws

rm -rf build install log

source /opt/ros/jazzy/setup.bash
source ~/PiPER_X/agx_arm_ws/install/setup.bash

colcon build \
  --symlink-install \
  --packages-select piper_pick_handover

source install/setup.bash
```

