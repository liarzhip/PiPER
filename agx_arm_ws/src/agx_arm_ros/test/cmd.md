## Nero Command Test

```bash
ros2 topic pub /control/move_j sensor_msgs/msg/JointState "$(cat test/nero/test_move_j.yaml)" -1
```

```bash
ros2 topic pub /control/move_p geometry_msgs/msg/PoseStamped "$(cat test/nero/test_move_p.yaml)" -1
```

## Piper Command Test

```bash
ros2 topic pub /control/move_j sensor_msgs/msg/JointState "$(cat test/piper/test_move_j.yaml)" -1
```

```bash
ros2 topic pub /control/move_p geometry_msgs/msg/PoseStamped "$(cat test/piper/test_move_p.yaml)" -1
```

```bash
ros2 topic pub /control/move_l geometry_msgs/msg/PoseStamped "$(cat test/piper/test_move_l.yaml)" -1
```

```bash
ros2 topic pub /control/move_c geometry_msgs/msg/PoseArray "$(cat test/piper/test_move_c.yaml)" -1
```

## Gripper Control

```bash
# Gripper control (width: 0.05m, force: 1.0N)
ros2 topic pub /control/joint_states sensor_msgs/msg/JointState "$(cat test/gripper/test_gripper_joint_states.yaml)" -1
```

```bash
# Arm + Gripper combo control
ros2 topic pub /control/joint_states sensor_msgs/msg/JointState "$(cat test/gripper/test_arm_gripper_joint_states.yaml)" -1
```

## Hand Control

```bash
# Hand position control (all fingers to 10)
ros2 topic pub /control/hand agx_arm_msgs/msg/HandCmd "$(cat test/hand/test_hand_position.yaml)" -1
```

```bash
# Hand speed control (all fingers speed 50)
ros2 topic pub /control/hand agx_arm_msgs/msg/HandCmd "$(cat test/hand/test_hand_speed.yaml)" -1
```

```bash
# Hand current control (all fingers current 50)
ros2 topic pub /control/hand agx_arm_msgs/msg/HandCmd "$(cat test/hand/test_hand_current.yaml)" -1
```

```bash
# Hand position-time control (all fingers to 50, time 1s)
ros2 topic pub /control/hand_position_time agx_arm_msgs/msg/HandPositionTimeCmd "$(cat test/hand/test_hand_position_time.yaml)" -1
```

```bash
# Hand control via joint_states (all fingers)
ros2 topic pub /control/joint_states sensor_msgs/msg/JointState "$(cat test/hand/test_hand_joint_states.yaml)" -1
```

```bash
# Arm + Hand combo control
ros2 topic pub /control/joint_states sensor_msgs/msg/JointState "$(cat test/hand/test_arm_hand_joint_states.yaml)" -1
```
