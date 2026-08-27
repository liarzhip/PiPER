# 测试 API 覆盖清单

统计来源：`tests/test_*.py`（虚拟 CAN 依赖 `tests/slaves/*`、`tests/conftest.py`）。

| 图例 | 含义 |
|------|------|
| **✅** | 虚拟 CAN 有用例，且对反馈/返回值有断言（含 CPV 往返、leader 解码等） |
| **🔶** | 虚拟 CAN 冒烟：主要校验**能发帧**（如 `demo_style`），不断言 MIT 编码、固件差异 |
| **○** | 离线/无 CAN（如 `test_mdh_fk.py`） |
| **—** | 驱动无此 API，或当前无测试 |

## API 归类

| 类别 | API（Piper 族 / Nero 共有除非注明） | 测试 |
|------|-------------------------------------|------|
| 工厂 | `load_class`, `create_arm`, `connect`, `disconnect` | `test_factory_config.py` ✅（各机型 `*FW` 路由 + 连接冒烟） |
| 末端 | `init_effector` | 仅 Piper：`test_agx_gripper_*`、`test_revo2_*` ✅ @ `DEFAULT`（Nero 无末端用例） |
| 运动 | `set_speed_percent`, `set_motion_mode`, `enable`, `disable`, `move_j/js`, `move_mit`, `move_p/l/c`, `electronic_emergency_stop`, `reset` | `test_piper_*`、`test_nero_*`：深度 ✅ @ `DEFAULT`；`move_mit` 多档 🔶 |
| 读取 | `get_joint_angles`, `get_flange_pose`, `get_arm_status`, `get_driver_states`, `get_motor_states`, `get_joint_enable_status`, `get_joints_enable_status_list`, `get_firmware`, `get_fps` | `*_read_apis_*` ✅ @ `DEFAULT` |
| 运动学 | `fk`；`get_mdh`, `fk_from_mdh`（`utiles`） | `fk`：Piper 族 + Nero ○；`get_mdh` / `fk_from_mdh`：仅 `"piper"` ○ |
| 坐标换算 | `get_tcp_pose`, `get_flange2tcp_pose`, `get_tcp2flange_pose` | —（抽象层，无虚拟 CAN） |
| 限位保护 | `get/set_joint_angle_vel_limits`, `get/set_joint_acc_limits`, `get/set_flange_vel_acc_limits`, `get/set_crash_protection_rating`, `clear_joint_error` | `*_proprietary_apis_*` 等 ✅ @ Piper/Nero `DEFAULT` |
| 主从 | `set_leader_mode`, `set_follower_mode`；**Nero** `set_normal_mode`, `get_leader_joint_angles`；**Piper** `move_leader_to_home`, `move_leader_follower_to_home`, `restore_leader_drag_mode` | 发帧 ✅；Nero `get_leader_joint_angles` ✅ @ `DEFAULT`+`V112`；Piper `get_leader_joint_angles` **—** |
| CPV | `set_motion_mode(cpv)`, `move_cpv_pos/vel`, `get/set_cpv_*` | Piper 1–6 轴、Nero **`V112+`** 1–7 轴 ✅（含 `cpv_each_public_api_once`） |
| 校准 | `calibrate_joint` | Piper ✅ @ `DEFAULT`；Nero **`V111+`** ✅（`test_nero_calibrate_joint_v111`） |
| Piper 专有 | `set_installation_pos`, `set_payload`, `get/set_joint_assistance_rating`, `set_links_vel_acc_period_feedback`, `set_*_to_default` | `test_piper_*` ✅ @ `DEFAULT` |
| 夹爪 | `move_gripper_m/deg`, `get/set_gripper_teaching_pendant_param`, `get_gripper_status`, `disable/reset/calibrate_gripper` 等 | `test_agx_gripper_virtual_can.py` ✅ |
| Revo2 | `position_ctrl`, `speed/current_ctrl`, `position_time_ctrl`, `get_hand_status`, `get_finger_pos/spd/current` | `test_revo2_virtual_can.py` ✅ |

## Nero 固件增量（测试视角）

自上而下：每行相对**上一档** SDK 驱动的变化。行为细节见 [固件参考](../docs/nero/firmware_reference.md#nero-固件参考)。

| SDK 驱动 | 固件 | 驱动层变化（摘要） | 虚拟 CAN |
|----------|------|-------------------|----------|
| `NeroFW.V112` | ≥ 1.12 | +CPV；leader `0x155`–`0x170`；主从同 Piper、上电 CAN 推送 | CPV、leader、warn ✅ @ V112 |
| `NeroFW.V111` | 1.11 | +`calibrate_joint`；12-bit MIT；位姿无 1.10 workaround | `calibrate_joint` ✅；`demo_style` 🔶 |
| `NeroFW.DEFAULT` | ≤ 1.10 | 基线（8-bit MIT、位姿 workaround 等） | `demo_style` 🔶；`read`/`proprietary`/运动扩展 ✅ |

## Piper 固件增量（测试视角）

机型：`piper` / `piper_h` / `piper_l` / `piper_x`（`PiperFW` 路由相同）。行为细节见 [固件参考](../docs/piper/firmware_reference.md#piper-固件参考)。

| SDK 驱动 | 固件标签 | 驱动层变化（摘要） | 虚拟 CAN |
|----------|----------|-------------------|----------|
| `PiperFW.V188` | ≥ S-V1.8-8 | 12-bit MIT 无 CRC；`0x2A1`/`0x151`；**piper_x** 4/5 轴取反 | `demo_style` 🔶；深度用例未单测 |
| `PiperFW.V183` | S-V1.8-3~7 | MIT 8-bit+CRC，全关节 `t_ff` ±8 | `demo_style` 🔶；深度用例未单测 |
| `PiperFW.DEFAULT` | ≤ S-V1.8-2 | 基线；CPV 1–6；MIT 8-bit+CRC 分轴量程 | `test_piper_driver_virtual_can.py` 主体 ✅ |

**变型：** `piper_h` / `piper_l` / `piper_x` 在 `test_factory_config` 有路由，`test_mdh_fk` 仅测 `fk`；无其它虚拟 CAN 业务用例。

## 缺口

| 项 | 说明 |
|----|------|
| 坐标 / TCP | `get_tcp_pose`, `get_flange2tcp_pose`, `get_tcp2flange_pose` 无测试 |
| Piper `get_leader_joint_angles` | 驱动有 API，虚拟 CAN 未测 |
| MIT / 电机状态 | 各档 FW 仅 🔶 发 `move_mit`；未断言 8/12-bit、CRC、`piper_x` 取反；Nero `get_motor_states` 的 `current` 取反未测 |
| Nero `V112` | `demo_style` 未 parametrized；`set_normal_mode` no-op、`calibrate_joint` 继承无专项断言 |
| Piper `V183`/`V188` | 除 `demo_style` 与 factory 路由外，无深度虚拟 CAN |
| Piper `piper_x` @ `V188` | 4/5 轴取反逻辑未单测 |
| Piper 变型 | `piper_h`/`piper_l`/`piper_x` 除 factory + `fk` 外无虚拟 CAN |
| MDH | `get_mdh("nero")` 未测 |
| 末端 | 夹爪 / Revo2 仅 Piper @ `DEFAULT` |

## 运行

```bash
python3 -m pytest -q tests
python3 -m pytest -q tests/test_{piper,nero}_driver_virtual_can.py
python3 -m pytest -q tests/test_{factory_config,mdh_fk,agx_gripper_virtual_can,revo2_virtual_can}.py
```
