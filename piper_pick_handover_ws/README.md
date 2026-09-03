## 这是PiPER机械臂自动抓取和摆放耳机项目的介绍

### 结构介绍

这个项目现在是 “三终端、三层架构”。

```yaml
终端 1：PIPER + MoveIt
负责机械臂驱动、ros2_control、MoveIt规划与执行

终端 2：视觉
负责 DaBai DC1、YOLO-Seg、MediaPipe、
手眼变换、目标锁定、抓取位姿生成

终端 3：任务
负责桌面碰撞体、WorkHome、抓取执行、
Handover规划和总任务FSM
```

**项目总体任务链路**

```sh
                    /manager/start
                           │
                           ▼
                    RESET TARGETS
                    /targets/reset
                           │
                           ▼
                前往自定义的 WORK_HOME
                /work_home/execute
                           │
                           ▼
                 开启第一次视觉识别
           /perception/start_observation
                           │
                           ▼
                 YOLO-Seg 检测耳机盒
                           │
                           ▼
        camera_color_optical_frame 中耳机盒Pose
 /perception/earbud_case_pose_camera
                           │
                           ▼
                   手眼坐标变换
                           │
                           ▼
       /targets/earbud_case_pose_base_live
                           │
                           ▼
                   多帧稳定锁定
                           │
                           ▼
       /targets/earbud_case_pose_base
       /targets/case_locked = true
                           │
                           ▼
                 停止视觉推理
          /perception/stop_observation
                           │
                           ▼
                 创建桌面碰撞体
        /scene/apply_table_from_case
                           │
                           ▼
                   生成抓取几何
                    ┌──────┴──────┐
                    ▼             ▼
           /pick/pregrasp_pose   /pick/grasp_pose
                    │
                    └──────┬──────┘
                           ▼
                   /pick/lift_pose
                           │
                           ▼
                  MoveIt PreGrasp
          /moveit/execute_pregrasp
                           │
                           ▼
                     打开夹爪
       /moveit/execute_gripper_open
                           │
                           ▼
               Cartesian 下探到 Grasp
       /moveit/execute_grasp_preview
                           │
                           ▼
                   实际夹住耳机盒
       /moveit/execute_gripper_grasp
                           │
                           ▼
              /pick/object_grasped=true
                           │
                           ▼
                   Cartesian Lift
             /moveit/execute_lift
                           │
                           ▼
                 PIPER机械零位
                    /move_home
                           │
                           ▼
                  joint1~6 ≈ 0
                           │
                           ▼
                  清空旧Handover
             /handover/reset_plan
                           │
                           ▼
                  Arm Palm Final
             /targets/arm_palm_final
                           │
                           ▼
                   重新开启视觉
           /perception/start_observation
                           │
                           ▼
              MediaPipe检测人的手掌
                           │
                           ▼
       /perception/palm_pose_camera
                           │
                           ▼
           /targets/palm_pose_base_live
                           │
                           ▼
                  多帧稳定锁定
                           │
                           ▼
        /targets/palm_final_pose_base
        /targets/palm_final_locked=true
                           │
                           ▼
                 停止视觉推理
          /perception/stop_observation
                           │
                           ▼
           根据当前机械HOME末端TF
              + Palm Final位置
                           │
                           ▼
             Handover Planner
                ┌──────────┴──────────┐
                ▼                     ▼
 /handover/approach_pose     /handover/final_pose
                │
                ▼
      /moveit/execute_handover_approach
                │
                ▼
       /moveit/execute_handover_final
                │
                ▼
                 松开
       /moveit/execute_gripper_open
                │
                ▼
              WORK_HOME
       /work_home/execute
                │
                ▼
           remove table
        /scene/remove_table
                │
                ▼
               DONE
```

**1. 第一终端：PIPER + MoveIt 层**

```sh
source /opt/ros/jazzy/setup.bash
source ~/PiPER_X/agx_arm_ws/install/setup.bash
source ~/PiPER_X/piper_pick_handover_ws/install/setup.bash

ros2 launch \
  piper_pick_handover \
  piper_moveit.launch.py
```

launch文件包含：

```yaml
agx_arm_ctrl/start_single_agx_arm_moveit.launch.py
# 默认配置
can_port        = can0
arm_type        = piper_x
effector_type   = agx_gripper
fw_version      = v189
speed_percent   = 10
auto_enable     = true
follow          = true
auto_control_gate = true
```

**2. 第二终端：视觉层**

```sh
export PYTHONPATH="$HOME/PiPER_X/orbbec/pyorbbecsdk/install/lib:$PYTHONPATH"

source /opt/ros/jazzy/setup.bash
source ~/PiPER_X/agx_arm_ws/install/setup.bash
source ~/PiPER_X/piper_pick_handover_ws/install/setup.bash

ros2 launch \
  piper_pick_handover \
  pick_handover_vision.launch.py
```

launch文件主要启动节点如下：

```yaml
handeye_static_tf # 这个是发布手眼标定的结果，不是一个独立的节点文件
scene_perception_node
target_transformer_node
target_lock_node
grasp_planner_node
```

scene_perception_node作为整个视觉系统的入口：

```yaml
DaBai DC1
+
RGB
+
Depth
↓
D2C对齐
↓
YOLO-Seg → 耳机盒
MediaPipe → 手掌
↓
3D Pose
```

**3. 第三终端：任务层**

```sh
source /opt/ros/jazzy/setup.bash
source ~/PiPER_X/agx_arm_ws/install/setup.bash
source ~/PiPER_X/piper_pick_handover_ws/install/setup.bash

ros2 launch \
  piper_pick_handover \
  pick_handover_task.launch.py
```

launch文件启动如下节点：

```yaml
moveit_executor_node # 是整个项目运动执行核心节点
table_scene_node # 添加桌面碰撞体，避免夹爪碰到桌面
work_home_controller_node
handover_planner_node
manager_node # 整个项目的总控制器
```

manager_node需要手动控制的服务：\
（1）启动：

```sh
/manager/start
std_srvs/srv/Trigger
```

（2）停止

```sh
/manager/stop
std_srvs/srv/Trigger
```

（3）查询状态

```sh
/manager/status
std_srvs/srv/Trigger
```

**4. 三个终端正常启动后：在第四个终端执行启动**

```sh
ros2 service call \
  /manager/status \
  std_srvs/srv/Trigger \
  "{}"
```

### 如果没有manager的完整人工测试指令

#### 1. 启动Piper_MovIt launch 文件后，测试是否能返回机械臂的关节信息，测试连接是否成功

```sh
ros2 topic echo /feedback/joint_states --once
```

#### 2. 前往work_home，先规划后执行：

```sh
ros2 service call \
  /work_home/plan \
  std_srvs/srv/Trigger \
  "{}"

  ros2 service call \
  /work_home/execute \
  std_srvs/srv/Trigger \
  "{}"
```

#### 3. 视觉操作：先reset存储的targets（如果有），再开启observation，输出定格后的earbud_case_pose_in_camera,最后通过tf坐标变换转换到机械臂base_link的空间坐标。

```sh
ros2 service call \
  /targets/reset \
  std_srvs/srv/Trigger \
  "{}"

ros2 service call \
  /perception/start_observation \
  std_srvs/srv/Trigger \
  "{}"

ros2 topic echo \
  /perception/earbud_case_pose_camera

ros2 topic echo \
  /targets/earbud_case_pose_base_live

ros2 topic echo \
  /targets/case_locked

ros2 topic echo \
  /targets/earbud_case_pose_base \
  --once

```

#### 4. 抓取目标路径规划，会规划三个位置坐标

```sh
ros2 topic echo \
  /pick/grasp_plan_ready \
  --once
ros2 topic echo \
  /pick/pregrasp_pose \
  --once
ros2 topic echo \
  /pick/grasp_pose \
  --once
ros2 topic echo \
  /pick/lift_pose \
  --once
```

#### 5. 生成桌面

```sh
ros2 service call \
  /scene/apply_table_from_case \
  std_srvs/srv/Trigger \
  "{}"

ros2 topic echo \
  /scene/table_ready \
  --once
```

#### 6. 抓取运动

```sh
ros2 service call /moveit/execute_pregrasp std_srvs/srv/Trigger "{}"

ros2 service call /moveit/execute_gripper_open std_srvs/srv/Trigger "{}"

ros2 service call /moveit/execute_grasp_preview std_srvs/srv/Trigger "{}"

ros2 service call /moveit/execute_gripper_grasp std_srvs/srv/Trigger "{}"

ros2 topic echo /pick/object_grasped --once

ros2 service call /moveit/execute_lift std_srvs/srv/Trigger "{}"
```

#### 7. 回到机械Home（零点）

```sh
ros2 service call \
  /move_home \
  std_srvs/srv/Empty \
  "{}"

ros2 topic echo \
  /feedback/joint_states \
  --once
```

#### 8. 检测Palm Final

```sh
ros2 service call \
  /handover/reset_plan \
  std_srvs/srv/Trigger \
  "{}"

ros2 service call \
  /targets/arm_palm_final \
  std_srvs/srv/Trigger \
  "{}"

ros2 service call \
  /perception/start_observation \
  std_srvs/srv/Trigger \
  "{}"

ros2 topic echo \
  /targets/palm_final_locked

ros2 topic echo \
  /targets/palm_final_pose_base \
  --once
```

#### 9. handover task：先规划后执行

```sh
ros2 service call \
  /handover/recompute_plan \
  std_srvs/srv/Trigger \
  "{}"

ros2 topic echo /handover/status --once
ros2 topic echo /handover/plan_ready --once
ros2 topic echo /handover/approach_pose --once
ros2 topic echo /handover/final_pose --once

ros2 service call \
  /moveit/execute_handover_approach \
  std_srvs/srv/Trigger \
  "{}"

ros2 service call \
  /moveit/execute_handover_final \
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
