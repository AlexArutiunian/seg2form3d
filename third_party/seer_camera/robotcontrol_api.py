      
import requests
import json
import time
import numpy as np
import math
import socket
import re
import sys
import io
import time
from datetime import datetime
import messageV4_3dcameradata_pb2


cam_data_msg = messageV4_3dcameradata_pb2.MessageV4_AllCameraData()

class RobotcontrolAPI:
    def __init__(self, hostname="localhost", port=8088):
        print("RobotcontrolAPI")
        self.api_path = "/universalPost"
        self.base_url = f"http://{hostname}:{port}"
        print(f"[INFO] Whole URL set to: {self.base_url + self.api_path}",)
    
    def post(self, node: str, data: dict):
        json_data = json.dumps(data)
        params = {
            "node_name": node,
            "service_name": "serviceDispatcher",
            "route_length": len(json_data)
        }
        response = requests.post(self.base_url + self.api_path, json_data, params=params)
        # response = requests.get(base_url + api_path, params=params)
        # 检查请求是否成功
        if response.status_code == 200:
            # 从响应头中获取 route_length
            # print("post response:", response.text)
            route_length = response.headers.get('route_length')
            
            if route_length is not None:
                # 将 route_length 转换为整数
                route_length = int(route_length)
                # 获取响应体内容
                content = response.text
                # 将内容分成两部分
                if route_length <= len(content):
                    part1 = content[:route_length]
                    part2 = content[route_length:]
                    
                    # 输出分割后的内容
                    # print("Part 1:", part1)
                    # print("Part 2:", part2)
                    return part2
                else:
                    print("Error: route_length is greater than content length.")
            else:
                print("Error: 'route_length' header not found.")
        else:
            print("Error: Request failed with status code", response.status_code)
        return None
    
    # 机械臂相关
    def getCurrentState(self):
        data = {
            "func_name": "getCurrentState",
            "command": "get_arm_state"
        }
        state = self.post("ArmPlanning", data)
        print("getCurrentState:", state)
        return state
    
    def getRightJoint(self):
        data = {
            "func_name": "getCurrentState",
            "command": "right_joints_pos"
        }
        state = self.post("ArmPlanning", data)
        status_data = json.loads(state)
        arm_state = status_data['status']
        print("aright_joints_pos:", arm_state)
        return arm_state["right_joints_pos"]
    
    def getLeftJoint(self):
        data = {
            "func_name": "getCurrentState",
            "command": "left_joints_pos"
        }
        state = self.post("ArmPlanning", data)
        status_data = json.loads(state)
        arm_state = status_data['status']
        print("left_joints_pos:", arm_state)
        return arm_state["left_joints_pos"]
    
    def getRightEE(self):
        data = {
            "func_name": "getCurrentState",
            "command": "right_ee_pose"
        }
        state = self.post("ArmPlanning", data)
        status_data = json.loads(state)
        arm_state = status_data['status']
        print("right_ee_pose:", arm_state)
        return arm_state["right_ee_pose"]
    
    def getLeftEE(self):
        data = {
            "func_name": "getCurrentState",
            "command": "left_ee_pose"
        }
        state = self.post("ArmPlanning", data)
        status_data = json.loads(state)
        arm_state = status_data['status']
        print("left_ee_pose:", arm_state)
        return arm_state["left_ee_pose"]

    # jack_pos
    def getJackPos(self):
        data = {
            "func_name": "getCurrentState",
            "command": "jack_pos"
        }
        state = self.post("ArmPlanning", data)
        status_data = json.loads(state)
        print("jack_pos:", status_data)
        return status_data['status']['jack_pos']
    
    # head_pos
    def getHeadPos(self):
        data = {
            "func_name": "getCurrentState",
            "command": "head_joints_pos"
        }
        state = self.post("ArmPlanning", data)
        status_data = json.loads(state)
        print("head_joints_pos:", status_data)
        return status_data['status']['head_joints_pos']
    
    def getWaistPos(self):
        data = {
            "func_name": "getCurrentState",
            "command": "waist_joint_pos"
        }
        state = self.post("ArmPlanning", data)
        status_data = json.loads(state)
        arm_state = status_data['status']
        print("waist_joint_pos:", arm_state)
        return arm_state["waist_joint_pos"]
    
    def getRightVel(self):
        data = {
            "func_name": "getCurrentState",
            "command": "right_joints_vel"
        }
        state = self.post("ArmPlanning", data)
        status_data = json.loads(state)
        arm_state = status_data['status']
        print("right_joints_vel:", arm_state)
        return arm_state["right_joints_vel"]
    
    def getLeftVel(self):
        data = {
            "func_name": "getCurrentState",
            "command": "left_joints_vel"
        }
        state = self.post("ArmPlanning", data)
        status_data = json.loads(state)
        arm_state = status_data['status']
        print("left_joints_vel:", arm_state)
        return arm_state["left_joints_vel"]
    
    def getRightEffort(self):
        data = {
            "func_name": "getCurrentState",
            "command": "right_joints_effort"
        }
        state = self.post("ArmPlanning", data)
        status_data = json.loads(state)
        arm_state = status_data['status']
        print("right_joints_effort:", arm_state)
        return arm_state["right_joints_effort"]
    
    def getLeftEffort(self):
        data = {
            "func_name": "getCurrentState",
            "command": "left_joints_effort"
        }
        state = self.post("ArmPlanning", data)
        status_data = json.loads(state)
        arm_state = status_data['status']
        print("left_joints_effort:", arm_state)
        return arm_state["left_joints_effort"]
    
    def getRightALL(self):
        data = {
            "func_name": "getCurrentState",
            "command": "right_joints_state"
        }
        state = self.post("ArmPlanning", data)
        status_data = json.loads(state)
        arm_state = status_data['status']
        print("right_joints_state:", arm_state)
        return arm_state["right_joints_state"]
    
    def getLeftALL(self):
        data = {
            "func_name": "getCurrentState",
            "command": "left_joints_state"
        }
        state = self.post("ArmPlanning", data)
        status_data = json.loads(state)
        arm_state = status_data['status']
        print("left_joints_state:", arm_state)
        return arm_state["left_joints_state"]
    
    def getDualALL(self):
        data = {
            "func_name": "getCurrentState",
            "command": "all_joints_state"
        }
        state = self.post("ArmPlanning", data)
        status_data = json.loads(state)
        arm_state = status_data['status']
        print("Dual_joints_state:", arm_state)
        return arm_state

    def getALL(self):
        data = {
            "func_name": "getCurrentState",
            # "command": "all_joints_state"
            "command": "all_state"
        }
        state = self.post("ArmPlanning", data)
        status_data = json.loads(state)
        arm_state = status_data.get('status', dict())
        return arm_state

    def getRightIK(self, pose):
        data = {
            "func_name": "getCurrentState",
            "command": "right_arm_ik",
            "pose": pose
        }
        state = self.post("ArmPlanning", data)
        status_data = json.loads(state)
        arm_state = status_data['status']
        print("getrightik:", status_data)
        return arm_state

    def getLeftIK(self, pose):
        data = {
            "func_name": "getCurrentState",
            "command": "left_arm_ik",
            "pose": pose
        }
        state = self.post("ArmPlanning", data)
        status_data = json.loads(state)
        arm_state = status_data['status']
        print("getleftik:", status_data)
        return arm_state

    def controlJack(self, joint, vel=0.5, acc=0.3):
        data = {
            "func_name": "robotControl",
            "move_group_name": "jack",
            "move_jack": joint,
            "velocity": vel,
            "acceleration": acc
        }
        state = self.post("ArmPlanning", data)
        status_data = json.loads(state)
        arm_state = status_data['status']
        print("controlJack:", status_data)
        return arm_state
    
    # def controlChassis(self, dist=1.0, vx=0.5, vy=0.0, loc_mode=0, task_id=None):
    # # """
    # # 控制AGV底盘移动（基于里程计或定位模式）
    
    # # Args:
    # #     dist (float): 移动距离（m），正数前进，负数后退（loc_mode=0时生效）
    # #     vx (float): X方向速度（m/s），正数前进，负数后退
    # #     vy (float): Y方向速度（m/s），正数左移，负数右移
    # #     loc_mode (int): 0=里程计模式（相对移动），1=定位模式（绝对坐标移动）
    # #     task_id (str): 任务ID（可选），默认自动生成 "odo"+时间戳
    
    # # Returns:
    # #     dict: 底盘执行状态（API返回的JSON数据）
    # # """
    #     if task_id is None:
    #         task_id = "odo" + str(time.time())  # 默认任务ID
    
    #     data = {
    #             "func_name": "updateMoveTaskList",
    #             "move_task_list": [{
    #                 "task_id": task_id,
    #                 "id": "SELF_POSITION",
    #                 "source_id": "SELF_POSITION",
    #                 "skill_name": "GoByOdometer",
    #                 "loc_mode": loc_mode,  # 0=里程计模式，1=定位模式
    #                 "dist": dist,          # 移动距离（m）
    #                 "vx": vx,              # X轴速度（m/s）
    #                 "vy": vy               # Y轴速度（m/s）
    #             }]
    #         }
    
    #     # 发送请求并解析返回状态
    #     # state = self.post("ArmPlanning", data)
    #     # status_data = json.loads(state)
    #     # print("controlChassis:", status_data)
    #     # return status_data
    #     state = self.post("ArmPlanning", data)
    #     status_data = json.loads(state)
    #     chassiss_state = status_data['status']
    #     print("controlChassis:", chassiss_state)
    #     return chassiss_state
    
    def controlChassis(self, dist=0.5, vx=0.2, vy=0.0, loc_mode=0, task_id=None):
        self.post(
            node="Tracking",  # 修正参数名从 node_name 改为 node
            data={
                "func_name": "updateMoveTaskList",
                "move_task_list": [{
                    "task_id": "odo" + str(time.time()),
                    "id": "SELF_POSITION",
                    "source_id": "SELF_POSITION",
                    "skill_name": "GoByOdometer",
                    "loc_mode": loc_mode,  # 0=里程计模式，1=定位模式
                    "dist": dist,
                    "vx": vx,
                    "vy": vy
                }]
            }
        )
    
    def controlWaist(self, joint, vel=0.5, acc=0.3):
        data = {
            "func_name": "robotControl",
            "move_group_name": "waist",
            "move_waist": joint,
            "velocity": vel,
            "acceleration": acc
            
        }
        state = self.post("ArmPlanning", data)
        status_data = json.loads(state)
        arm_state = status_data['status']
        print("controlWaist:", status_data)
        return arm_state
    
    def controlRight(self, joint):
        data = {
            "func_name": "robotControl",
            "move_group_name": "right_arm",
            "move_right_joints": joint,
            "velocity": 0.5,
            "acceleration": 0.2
            
        }
        state = self.post("ArmPlanning", data)
        status_data = json.loads(state)
        arm_state = status_data['status']
        return arm_state
    
    def controlLeft(self, joint):
        data = {
            "func_name": "robotControl",
            "move_group_name": "left_arm",
            "move_left_joints": joint,
            "velocity": 0.5,
            "acceleration": 0.2
            
        }
        state = self.post("ArmPlanning", data)
        status_data = json.loads(state)
        arm_state = status_data['status']
        return arm_state
    
    def controlHead(self, joint):
        data = {
            "func_name": "robotControl",
            "move_group_name": "head",
            "move_head": joint,
            "velocity": 0.3,
            "acceleration": 0.3
        }
        state = self.post("ArmPlanning", data)
        status_data = json.loads(state)
        arm_state = status_data['status']
        return arm_state
    
    def controlDualArm(self, right_joints, left_joints):
        data = {
            "func_name": "robotControl",
            "move_group_name": "dual_arm",
            "move_right_joints": right_joints,
            "move_left_joints": left_joints,
            "velocity": 0.5,
            "acceleration": 0.3
        }
        state = self.post("ArmPlanning", data)
        status_data = json.loads(state)
        arm_state = status_data['status']
        return arm_state
    
    def controlRightPose(self, pose):
        data = {
            "func_name": "robotControl",
            "move_group_name": "right_arm",
            "move_right_to_pose": pose,
            "velocity": 0.3,
            "acceleration": 0.2
            
        }
        state = self.post("ArmPlanning", data)
        status_data = json.loads(state)
        arm_state = status_data['status']
        return arm_state
    
    def controlLeftPose(self, pose):
        data = {
            "func_name": "robotControl",
            "move_group_name": "left_arm",
            "move_left_to_pose": pose,
            "velocity": 0.3,
            "acceleration": 0.2
        }
        state = self.post("ArmPlanning", data)
        status_data = json.loads(state)
        arm_state = status_data['status']
        return arm_state
    
    def controlDualPose(self, right_pose, left_pose):
        data = {
            "func_name": "robotControl",
            "move_group_name": "dual_arm",
            "move_right_to_pose": right_pose,
            "move_left_to_pose": left_pose,
            "velocity": 0.3,
            "acceleration": 0.2
        }
        state = self.post("ArmPlanning", data)
        status_data = json.loads(state)
        arm_state = status_data['status']
        return arm_state
    
    def controlHeadDualArm(self, head_joints, right_joints, left_joints):
        data = {
            "func_name": "robotControl",
            "move_group_name": "head_dual_arm",
            "move_head": head_joints,
            "move_right_joints": right_joints,
            "move_left_joints": left_joints,
            "velocity": 0.3,
            "acceleration": 0.2
        }
        state = self.post("ArmPlanning", data)
        status_data = json.loads(state)
        arm_state = status_data['status']
        return arm_state

    def setJackZero(self):
        data = {
            "func_name": "robotControl",
            "jack_motor_set_zero": "set_jack_zero"
        }
        state = self.post("ArmPlanning", data)
        status_data = json.loads(state)
        arm_state = status_data['status']
        return arm_state
    
    def openArmCollision(self):
        data = {
            "func_name": "robotControl",
            "set_arm_collision": True
        }
        state = self.post("ArmPlanning", data)
        status_data = json.loads(state)
        arm_state = status_data['status']
        return arm_state
    
    def closeArmCollision(self):
        data = {
            "func_name": "robotControl",
            "set_arm_collision": False
        }
        state = self.post("ArmPlanning", data)
        status_data = json.loads(state)
        arm_state = status_data['status']
        return arm_state
    
    def openArmCollision_torque(self, joint_torque_min=None, joint_torque_max=None):
        if joint_torque_min is None:
            joint_torque_min = [-50,-50,-50,-50,-50,-50,-50,-50,-50,-50,-50,-50,-50,-50],
        if joint_torque_max is None:
            joint_torque_max = [50,50,50,50,50,50,50,50,50,50,50,50,50,50],
        data = {
            "func_name" : "robotControl",
            "set_arm_collision" : True,
            "joint_torque_min" : joint_torque_min,
            "joint_torque_max" : joint_torque_max
        }
        state = self.post("ArmPlanning", data)
        status_data = json.loads(state)
        arm_state = status_data['status']
        return arm_state
    
    def waitMove(self):
        time.sleep(0.1)
        while True:
            data = {
                "func_name": "getCurrentState",
                "command": "get_arm_state"
            }
            state = self.post("ArmPlanning", data)
            status_data = json.loads(state)
            arm_state = status_data['status']
            print("arm_state:", arm_state)
            if arm_state["get_arm_state"] == "RUNNING":
                time.sleep(0.1)
            elif arm_state["get_arm_state"] == "IDLE":
                time.sleep(0.1)
                break
            else:
                print("arm_state:", arm_state)
    
    # arm_name [str] : right_arm left_arm dual_arm
    def dragModeRecord(self, arm_name, trajectory_name):
        data = {
            "func_name": "robotControl",
            "drag_teach_mode": True,
            "arm_name": arm_name,
            "record": 1,
            "trajectory_name": trajectory_name
        }
        state = self.post("ArmPlanning", data)
        return state
    
    def dragMode(self, arm_name, trajectory_name = "test_trajectory1"):
        data = {
            "func_name": "robotControl",
            "drag_teach_mode": True,
            "arm_name": arm_name,
            "record": 0,
            "trajectory_name": trajectory_name
        }
        state = self.post("ArmPlanning", data)
        return state
    
    def outdragMode(self):
        data = {
            "func_name": "robotControl",
            "drag_teach_mode": False
        }
        state = self.post("ArmPlanning", data)
        return state
    
    def dragReplay(self, arm_name = "right_arm", trajectory_name = "test_trajectory1"):
        data = {
            "func_name": "robotControl",
            "replay_drag_trajectory": arm_name,
            "trajectory_name": trajectory_name
        }
        state = self.post("ArmPlanning", data)
        return state
    # 实时控制
    def switch_real_time_mode(self, group_name):
        data = {
            "func_name": "robotControl",
            "group_name": group_name,
            "real_time_mode": True
        }
        state = self.post("ArmPlanning", data)
        return state
    
    def out_switch_real_time_mode(self):
        data = {
            "func_name": "robotControl",
            "real_time_mode": False
        }
        state = self.post("ArmPlanning", data)
        return state
    
    def left_real_time_trajectory(self, joints):
        data = {
            "func_name": "robotControl",
            "left_joints_cmd": joints,
        }
        state = self.post("ArmPlanning", data)
        return state
    
    def right_real_time_trajectory(self, joints):
        data = {
            "func_name": "robotControl",
            "right_joints_cmd": joints,
        }
        state = self.post("ArmPlanning", data)
        return state
    
    def dual_real_time_trajectory(self, dual_joints):
        data = {
            "func_name": "robotControl",
            "dual_joints_cmd": dual_joints,
        }
        state = self.post("ArmPlanning", data)
        return state
    
    def all_real_time_trajectory(self, dual_joints, jack_pos, waist_joint, head_joints):
        data = {
            "func_name": "robotControl",
            "dual_joints_cmd": dual_joints,   # [14]
            "jack_pos_cmd": jack_pos,         # [1]
            "waist_joint_cmd": waist_joint,   # [1]
            "head_joints_cmd": head_joints    # [2]
        }
        state = self.post("ArmPlanning", data)
        return state
    
    def jack_head_dual_real_time_trajectory(self, dual_joints, jack_pos, head_joints):
        data = {
            "func_name": "robotControl",
            "dual_joints_cmd": dual_joints,   # [14]
            "jack_pos_cmd": jack_pos,         # [1]
            "head_joints_cmd": head_joints    # [2]
        }
        state = self.post("ArmPlanning", data)
        return state

    def open_dual_arm_optional(self, max_vels=None, max_accs=None, max_jerks=None):
        if max_vels is None:
            max_vels = [1.2, 1.2, 4.14, 4.14, 4.14, 4.14, 4.14, 1.2, 1.2, 4.14, 4.14, 4.14, 4.14, 4.14]
        
        if max_accs is None:
            max_accs = [1.5, 1.5, 6.28, 6.28, 6.28, 6.28, 6.28, 1.5, 1.5, 6.28, 6.28, 6.28, 6.28, 6.28]
            
        if max_jerks is None:
            max_jerks = [20.28, 20.28, 50.28, 50.28, 50.28, 50.28, 50.28, 20.28, 20.28, 50.28, 50.28, 50.28, 50.28, 50.28]

        data = {
            "func_name": "robotControl",
            "group_name": "dual_arm",
            "real_time_mode": True,
            "max_vels": max_vels,
            "max_accs": max_accs,
            "max_jerks": max_jerks
        }
        state = self.post("ArmPlanning", data)
        return state

    def open_right_arm_optional(self, group_name = "right_arm", max_vels=None, max_accs=None, max_jerks=None):
        if max_vels is None:
            max_vels = [1.2, 1.2, 4.14, 4.14, 4.14, 4.14, 4.14]
        
        if max_accs is None:
            max_accs = [1.5, 1.5, 6.28, 6.28, 6.28, 6.28, 6.28]
            
        if max_jerks is None:
            max_jerks = [20.28, 20.28, 50.28, 50.28, 50.28, 50.28, 50.28]
        data = {
            "func_name" : "robotControl",
            "group_name" : group_name,
            "real_time_mode" : True,
            "max_vels" : max_vels,
            "max_accs" : max_accs,
            "max_jerks" : max_jerks
        }
        state = self.post("ArmPlanning", data)
        return state
    
    def open_left_arm_optional(self, group_name = "left_arm", max_vels=None, max_accs=None, max_jerks=None):
        if max_vels is None:
            max_vels = [1.2, 1.2, 4.14, 4.14, 4.14, 4.14, 4.14]
        
        if max_accs is None:
            max_accs = [1.5, 1.5, 6.28, 6.28, 6.28, 6.28, 6.28]
            
        if max_jerks is None:
            max_jerks = [20.28, 20.28, 50.28, 50.28, 50.28, 50.28, 50.28]
        data = {
            "func_name" : "robotControl",
            "group_name" : group_name,
            "real_time_mode" : True,
            "max_vels" : max_vels,
            "max_accs" : max_accs,
            "max_jerks" : max_jerks
        }
        state = self.post("ArmPlanning", data)
        return state
    
    def open_head_jack_dual_arm_optional(self, group_name = "jack_head_dual_arm", max_vels=None, max_accs=None, max_jerks=None):
        if max_vels is None:
            max_vels = [0.5, 3.14, 3.14, 1.2, 1.2, 4.14, 4.14, 4.14, 4.14, 4.14, 1.2, 1.2, 4.14, 4.14, 4.14, 4.14, 4.14]
        
        if max_accs is None:
            max_accs = [0.5, 3.14, 3.14, 1.5, 1.5, 6.28, 6.28, 6.28, 6.28, 6.28, 1.5, 1.5, 6.28, 6.28, 6.28, 6.28, 6.28]
            
        if max_jerks is None:
            max_jerks = [0.0, 10.28, 10.28, 20.28, 20.28, 50.28, 50.28, 50.28, 50.28, 50.28, 20.28, 20.28, 50.28, 50.28, 50.28, 50.28, 50.28]
        data = {
            "func_name" : "robotControl",
            "group_name" : group_name,
            "real_time_mode" : True,
            "max_vels" : max_vels,
            "max_accs" : max_accs,
            "max_jerks" : max_jerks
        }
        state = self.post("ArmPlanning", data)
        return state
    
    def open_jack_waist_head_dual_arm_optional(self, group_name = "jack_waist_head_dual_arm", max_vels=None, max_accs=None, max_jerks=None):
        if max_vels is None:
            max_vels = [0.5, 3.14, 3.14, 3.14, 1.2, 1.2, 4.14, 4.14, 4.14, 4.14, 4.14, 1.2, 1.2, 4.14, 4.14, 4.14, 4.14, 4.14]
        
        if max_accs is None:
            max_accs = [0.5, 3.14, 3.14, 3.14, 1.5, 1.5, 6.28, 6.28, 6.28, 6.28, 6.28, 1.5, 1.5, 6.28, 6.28, 6.28, 6.28, 6.28]
            
        if max_jerks is None:
            max_jerks = [0.0, 10.28, 10.28, 10.28, 20.28, 20.28, 50.28, 50.28, 50.28, 50.28, 50.28, 20.28, 20.28, 50.28, 50.28, 50.28, 50.28, 50.28]
        data = {
            "func_name" : "robotControl",
            "group_name" : group_name,
            "real_time_mode" : True,
            "max_vels" : max_vels,
            "max_accs" : max_accs,
            "max_jerks" : max_jerks
        }
        state = self.post("ArmPlanning", data)
        return state
    
    # 阻抗
    def generate_excitation_trajectory(self, amplitude_sine, amplitude_cosine, amplitude_fifth, duration, timestep):
        har_num = amplitude_sine.shape[1]
        omega = 2.0 * np.pi / duration

        time_array = np.arange(0, duration + timestep, timestep)
        N = len(time_array)

        pos_traj = np.zeros((N, 7))  
        vel_traj = np.zeros((N, 7))  

        for idx, t in enumerate(time_array):
            pos = np.zeros(7)
            vel = np.zeros(7)

            for i in range(har_num):
                k = i + 1
                pos += amplitude_sine[:, i] * np.sin(omega * k * t) / (omega * k) \
                    - amplitude_cosine[:, i] * np.cos(omega * k * t) / (omega * k)
                vel += amplitude_sine[:, i] * np.cos(omega * k * t) \
                    + amplitude_cosine[:, i] * np.sin(omega * k * t)

            t_vec = np.array([1, t, t**2, t**3, t**4, t**5])
            pos += amplitude_fifth @ t_vec

            t_vec_vel = np.array([0, 1, 2*t, 3*t**2, 4*t**3, 5*t**4])
            vel += amplitude_fifth @ t_vec_vel

            s = 0.4
            s7 = 0.6
            pos[:5] *= s
            vel[:5] *= s
            pos[5:] *= s7
            vel[5:] *= s7

            pos_traj[idx, :] = pos
            vel_traj[idx, :] = vel

        return pos_traj

    def left_compliant_task_trajectory(self, left_desired_ee_pos):
        data = {
            "func_name" : "robotControl",
            "left_desired_ee_pos": left_desired_ee_pos,
            "max_velocity": 0.2,
            "max_acceleration": 0.3
        }
        self.post("ArmPlanning", data)

    def right_compliant_task_trajectory(self, right_desired_ee_pos):
        data = {
            "func_name" : "robotControl",
            "right_desired_ee_pos": right_desired_ee_pos,
            "max_velocity": 0.2,
            "max_acceleration": 0.3
        }
        self.post("ArmPlanning", data)
        
    def dual_compliant_task_trajectory(self, dual_desired_ee_pos):
        data = {
            "func_name" : "robotControl",
            "dual_desired_ee_pos": dual_desired_ee_pos,
            "max_velocity": 0.2,
            "max_acceleration": 0.3
        }
        self.post("ArmPlanning", data)

    def compliantMode(self, arm_name, mode_type, spring_ratio, damping_ratio, mass, r_com):
        data = {
            "func_name" : "robotControl",
            "compliant_control_mode" : True,
            "arm_name" : arm_name,
            "mode_type" : mode_type,
            "spring_ratio": spring_ratio,
            "damping_ratio": damping_ratio,
            "mass": mass,
            "r_com": r_com
        }
        state = self.post("ArmPlanning", data)

    def waitCompliant(self):
        time.sleep(0.1)
        while True:
            data = {
                "func_name": "getCurrentState",
                "command": "get_arm_state"
            }
            state = self.post("ArmPlanning", data)
            status_data = json.loads(state)
            arm_state = status_data['status']
            if arm_state.get("get_arm_state") == "COMPLIANT":
                time.sleep(0.05)
                break
            else:
                time.sleep(0.1)

    def left_compliant_trajectory(self, left_desired_pos):
        data = {
            "func_name" : "robotControl",
            "left_desired_pos": left_desired_pos
        }
        self.post("ArmPlanning", data)

    def right_compliant_trajectory(self, right_desired_pos):
        data = {
            "func_name" : "robotControl",
            "right_desired_pos": right_desired_pos
        }
        self.post("ArmPlanning", data)

    def dual_compliant_trajectory(self, dual_desired_pos):
        data = {
            "func_name" : "robotControl",
            "dual_desired_pos": dual_desired_pos
        }
        self.post("ArmPlanning", data)

    def outcompliantMode(self):
        data = {
            "func_name" : "robotControl",
            "compliant_control_mode" : False
        }
        state = self.post("ArmPlanning", data)
        
    # def ExcitationTrajectory(self):
    #     data = {
    #         "func_name": "robotControl",
    #         "excitation_trajectory": True,
    #         "amplitude_sine": [
    #             [0.0769846767654865, 0.147750568199432, -0.492885886306759, 0.754678179626206, 1.01656270825411,
    #              -0.285096624439121, -0.285277675580051, -0.540229560803672],
    #             [0.959907505401296, 0.0131979540981403, -0.0688816292272123, 0.112777184224310, 0.0408516722490407,
    #              0.110979433478667, -0.245987008070171, 0.302476082577314],
    #             [3.26086165051952, -0.0901713880476215, 0.265516499266602, 0.951845037208443, 0.175356630297162,
    #              -0.0927240045921166, -1.05937626012659, 0.0603984170158273],
    #             [-0.137188635302063, 0.0541532061425681, 0.287292201503073, -0.0328102627422568, 0.137432344697275,
    #              -0.975665399973818, -0.0337343512866358, 0.639704967072883],
    #             [-1.32339613309102, 0.329238069422633, -0.432408597386512, 0.343230343149307, -0.658926874699022,
    #              -0.373181589736196, -0.431374194000121, 0.755583891356382],
    #             [-0.444523086070529, 0.0673385796463732, -0.0871776037012712, -0.0413150691340519, -0.0587552665515011,
    #              0.233375577987157, 0.0920726799874045, -0.189019550602337],
    #             [0.554805639337095, 0.122284775639480, 0.264282047870730, 0.0604394594696638, 0.325377991045677,
    #              -0.0477909123183056, -0.628553240384681, -0.0597264646286733]
    #         ],
    #         "amplitude_cosine": [
    #             [0.439896638727794, 0.0511963694962525, 0.198926640037784, -0.229891176143516, -0.266253537894379,
    #              0.262971483928943, -0.0453812600252814, 0.0741505151462745],
    #             [0.945430708758303, 0.163554648086067, 0.182586905427702, -0.0908168032675452, 0.210146910887288,
    #              -0.970500323789440, 0.0297241421098817, 0.593644928644327],
    #             [-0.535083480625224, 0.0926082456361234, -0.328882304592768, 0.305419503811220, -0.466485716346588,
    #              -0.113205705369925, 0.286284270801412, 0.0312014696105918],
    #             [0.0688202050834786, 0.0645982816657266, 0.0666243591981304, -0.380223794248660, 0.127321951516083,
    #              0.0126950878887590, -0.188959004539680, 0.201434874616326],
    #             [0.0361842131848624, 0.162949261480300, -0.253045128134895, 0.176115847545153, 0.190285937375042,
    #              -0.432586681711330, -0.217881746177467, 0.381651819968799],
    #             [-0.250925896792077, -0.0365265374904148, -0.0701182411158502, 0.0926360814057475, -0.350454183206802,
    #              0.118065668020491, -0.126686458597806, 0.209533377803771],
    #             [0.305987735659865, 0.0683325441621368, 0.127868190853668, -0.444050516357571, 0.0656408586268116,
    #              -0.0797086932094321, 0.329752409508725, -0.0843931863224280]
    #         ],
    #         "amplitude_fifth": [
    #             [2.23307703136328, -0.392486385715631, -0.0776602864633596, 0.00953831227217544, -0.000304337199245751,
    #              2.90730656085654e-06],
    #             [4.88970482662164, -1.22532119473138, -0.171949266531092, 0.0250779643768660, -0.000871788737663096,
    #              9.07645329430658e-06],
    #             [-2.81426718484550, -3.47170658154122, 0.0914095652520060, 0.0324805465558800, -0.00182715969502068,
    #              2.57163450484537e-05],
    #             [0.258007532252102, 0.0608159298889742, 0.0127191652109489, -0.00152367690171854, 4.79190335060405e-05,
    #              -4.50488369547966e-07],
    #             [0.285902839651230, 1.79123508498454, -0.0200394524688913, -0.0185666485574577, 0.000972864544470418,
    #              -1.32684080369225e-05],
    #             [-1.48836535408903, 0.428003738438755, 0.0437942691818184, -0.00767521503921851, 0.000286440153779107,
    #              -3.17039806250930e-06],
    #             [1.47138348867474, -0.591119296030986, -0.0558291216740226, 0.0102899336230570, -0.000390431966321690,
    #              4.37866145208149e-06]
    #         ]
    #     }
    #     self.post("ArmPlanning", data)
    def ExcitationTrajectory(self):
        data = {
            "func_name" : "robotControl",
            "excitation_trajectory" : True,
            # left
            "amplitude_sine": [
                [0.496984834360383, -0.418412013220084, -0.106935577463573, -0.376426425379385, 0.421393661175103, 0.0805432335513272, 0.377628307922894, 0.210036187598116],
                [-0.396108159311673, -0.0692987894819131, 0.0130003785540861, 0.784054160071369, 0.415776980559260, 0.274587256196507, -1.24096286837386, -0.327204399859642],
                [0.329363476125197, 0.328982344075882, -0.701312403857965, 0.265215403949707, 0.859463263401664, 0.161812990447597, -0.522287101848612, -0.469260972423066],
                [49.8543553349133, 2.97313429314674, 0.778095155466986, 0.0765133065371888, 0.492455276188938, -0.272535492918540, 0.540481141541365, -0.498051805111450],
                [-1.86640522439799, -0.0633974194249294, -0.0663938403026457, 0.340707074283023, -0.258723986256527, -0.278228662671427, 0.550719474978033, -0.113236160649397],
                [0.335824081445145, 0.0219736412161619, 0.0497212291596968, 0.217562295156834, 0.0199134654993458, 0.0155840017591763, -0.304400612976229, 0.0344905229244595],
                [-0.800309952138434, -0.0772242093748194, 0.0785078741790462, 0.0185243072251679, -0.125456560578125, -0.0322591064485868, 0.270711500158135, -0.177828693269316]
            ],
            "amplitude_cosine":  [
                [-1.23709222125765, -0.326855861051946, -0.260084695616704, 0.0297402986020978, -0.624114299387154, 1.22778517409930, 0.357282334098881, -0.803419472754052],
                [0.00746208469019677, 0.0515360865273191, -0.199359921034225, -0.221331743097601, 0.614904466830252, -0.718462714088096, 0.431242287779846, -0.0626145638623680],
                [-1.34191320704691, 0.00837144299428111, -0.0788839038033610, 0.538970617912011, 0.203023334892056, -0.259905096284343, 0.0683012075395132, -0.320842127890704],
                [0.408539150725206, 0.199648056634896, 0.00399831672945509, -0.329967312519792, 0.595256038975102, 0.525276870643978, -0.798675443511762, 0.0762757871316035],
                [1.13719780198832, 0.440952320501954, -0.247995583437955, 0.997007024016086, -0.326609190023822, -0.651268384684113, -0.305554575548398, 0.559342520715725],
                [-1.05005912395625, -0.243516153587880, -0.148989948037609, 0.138667782189639, 0.0831600963142236, -0.462639043028135, 0.157877375060092, 0.109594530270217],
                [-0.442355792082414, -0.0798688326728480, 0.00138403888281817, 0.121372900891102, 0.0446737552400536, 0.00464636986593143, 0.233382414060086, -0.306560878416857]
            ],
            "amplitude_fifth": [
                [-6.92017982755484, -0.684812208544781, 0.233769755865569, -0.00797562585165163, -0.000120707053785348, 5.07268302625750e-06],
                [-0.150591387855144, 0.546155441645870, 0.00956582757144222, -0.00670611563416116, 0.000314048387104849, -4.04559586404324e-06],
                [-6.02723079290524, -0.251976999870404, 0.213511227361834, -0.0114343373811177, 9.72474749184762e-05, 1.86649629533640e-06],
                [2.52693227230460, -53.9444472097645, -0.0677935353161123, 0.603902315796235, -0.0300444634891093, 0.000399588497850107],
                [6.57307264921112, 1.75495874444185, -0.215549454242362, -0.00512957798875209, 0.000735477686642855, -1.29996944032731e-05],
                [-5.78226539363738, -0.390668624184590, 0.189283685904531, -0.00827814990269543, -6.72291798640861e-06, 2.89384166062666e-06],
                [-2.13310399409362, 0.845334840246933, 0.0712107813014442, -0.0141399947561734, 0.000548753557138792, -6.26173955738473e-06]
            ]
        }

    # 底盘导航
    def search(self,lm: str):
        # "actionParameters":{"hold_dir": -5.0, "reach_angle": 3.14}
        """
        搜索拓扑线路，去目标站点 LM4
        """
        task_id = "search_" + str(time.time())
        # node="Task",  # 修正参数名从 node_name 改为 node
        data = {
            # node="Tracking",  # 修正参数名从 node_name 改为 node
            "func_name": "setTask",
            "task": {
                "task_id": task_id,
                "id": lm
            }
        }
        self.post(node="Task",data=data)
        return task_id
    
    def search_recfile(self,lm: str):
        # "actionParameters":{"hold_dir": -5.0, "reach_angle": 3.14}
        """
        搜索拓扑线路，识别去目标站点 LM4顶升
        """
        task_id = "search_" + str(time.time())
        # node="Task",  # 修正参数名从 node_name 改为 node
        data = {
            "func_name": "setTask",
            "task": {
                "task_id": task_id,
                "id": lm,
                "actionParameters": {
                    "operation": "JackLoad",
                    "recfile": "s0002.shelf",
                    "recognize": True
                    }
                }
            }
        self.post(node="Task",data=data)
        return task_id
    
    def search_hold_dir(self,lm: str):
        # "actionParameters":{"hold_dir": -5.0, "reach_angle": 3.14}
        """
        搜索拓扑线路，识别去目标站点 LM4并且保持车体在地图坐标下朝向为90度不变。
        """
        task_id = "search_" + str(time.time())
        # node="Task",  # 修正参数名从 node_name 改为 node
        data = {
            "func_name": "setTask",
            "task": {
                "task_id": task_id,
                "id": lm,
                "actionParameters": {"hold_dir": 90.0}
                }
            }
        self.post(node="Task",data=data)
        return task_id

    def taskStatus(self,task_id: str):
        """_summary_
        Args:
            task_id (str): _description_
        Returns:
            str:    StatusNone,
                    Waiting,
                    Running,
                    Suspended,
                    Completed,
                    Failed,
                    Canceled
        """
        data = {
            "func_name": "taskStatus",
            "cnt": str(time.time()),
            "task_id": task_id
        }
        ret = self.post("Task", data)
        if ret:
            response = json.loads(ret)
            return response.get("status", "StatusNone")
        return "StatusNone"
    
    def cancelTask(self):
        """取消任务"""
        data = {
            "func_name": "cancelTask"
        }
        state = self.post("Task", data)
        return state
    
    def wait_until_complete(self,task_id, interval=2, timeout=20, task_name=None):
        """
        循环检测任务状态直到为 Completed 或超时
        Args:
            task_id (str): 任务ID
            interval (int): 检查间隔秒数
            timeout (int): 超时时间（秒）
        Returns:
            bool: True 如果完成, False 如果失败/超时
        """
        run_time = time.time()
        display_name = task_name if task_name else task_id
        while True:
            status = self.taskStatus(task_id)
            print(f"[{time.strftime('%H:%M:%S')}] 任务 [{display_name}] 状态: {status}")
            if status.lower() == "completed":
                print(f"任务 {task_id} 已完成。")
                return True
            elif status.lower() in ["failed", "canceled"]:
                print(f"任务 {task_id} 失败或被取消。")
                return False
        
            if time.time() - run_time > timeout:
                print(f"任务 {task_id} 超时（{timeout}秒）未完成")
                return False
            time.sleep(interval)  # 每 2 秒检查一次

    def run_search_sequence(self,station_list):
        """
        顺序执行多个搜索任务
        Args:
            station_list (list): 站点名称列表, 例如 ["LM1", "LM2", "LM3"]
        """
        for station in station_list:
            print(f"\n 开始执行搜索任务: {station}")
            task_id = self.search(station)
        
            success = self.wait_until_complete(task_id)
            if not success:
                print(f" 停止执行后续任务（{station} 未完成）")
                break
        else:
            print("\n 所有任务执行完毕！")

    # 顶升到位判定
    def waitJackTarget(self, target_pos, timeout=10, tolerance=0.005):
        """
        阻塞等待 Jack 到达指定位置
        Args:
            target_pos (float): 目标高度/位置 (与 controlJack 下发的值一致)
            timeout (int): 超时时间 (秒)
            tolerance (float): 允许误差范围 (默认 0.01)
        """
        try:
            if isinstance(target_pos, list):
                if len(target_pos) > 0:
                    target_val = float(target_pos[0])
                else:
                    print("[-] 参数错误: 传入了空列表")
                    return False
            # 如果是其他类型，尝试直接转 float
            else:
                target_val = float(target_pos)
        except Exception as e:
            print(f"[-] 参数错误: target_pos 无法识别，传入类型: {type(target_pos)}, 值: {target_pos}")
            return False

        start_time = time.time()
        print(f"[*] 等待 Jack 到位: 目标 {target_val:.4f}, 容差 {tolerance}")
        
        while True:
            try:
                # 1. 获取当前位置
                curr_pos = self.getJackPos()
                
                if isinstance(curr_pos, list):
                    if len(curr_pos) > 0:
                        current_val = float(curr_pos[0]) # 取列表第一个元素
                    else:
                        current_val = 0.0
                else:
                    current_val = float(curr_pos)

                # 2. 计算误差
                diff = abs(current_val - target_val)
                
                # 3. 判断是否到位
                if diff <= tolerance:
                    print(f"[+] Jack 已到位 (当前: {current_val:.4f}, 目标: {target_val:.4f})")
                    return True
                
            except Exception as e:
                print(f"[-] 获取 Jack 位置异常: {e}")
                # 增加短暂延时防止报错刷屏太快
                time.sleep(0.5)

            # 4. 超时检查
            if time.time() - start_time > timeout:
                val_str = f"{current_val:.4f}" if 'current_val' in locals() else "Unknown"
                print(f"[-] Jack 等待超时 (当前: {val_str}, 目标: {target_val:.4f})")
                return False
            
            time.sleep(0.2)

    # 电池相关
    def getBatteryMsg(self):
        '''获取电池信息'''
        data = {
            "func_name": "getBatteryMsg"
        }
        state = self.post("pyBatteryServer", data)
        print("电池信息:", state)
        return state

    def getBatteryAllMsg(self):
        '''获取电池所有信息'''
        data = {
            "func_name": "getChannelsData",
            "data_type": "json",
            "list": [
                {
                    "channel_name": "/BatteryInfo/Battery-000",
                    "message_type": "rbk4.protocol.MessageV4_Battery"
                }
            ]   
        }
        state = self.post("NetProtocol", data)
        print("电池信息:", state)
        return state
    
    def getBatterypercent(self):
        '''获取电池电量信息'''
        data = {
            "func_name": "getChannelsData",
            "data_type": "json",
            "list": [
                {
                    "channel_name": "/BatteryInfo/Battery-000",
                    "message_type": "rbk4.protocol.MessageV4_Battery"
                }
            ]   
        } 
        # 临时屏蔽 print 输出
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        state = self.post("NetProtocol", data)

        # 恢复正常输出
        sys.stdout = old_stdout
        # 从返回字符串中匹配 "percetage": 后面的数字
        match = re.search(r'"percetage"\s*:\s*([0-9.]+)', state)
        if match:
            percent = float(match.group(1))
            print(f"电池百分比: {percent* 100:.1f}%")   #  只打印百分比
            return percent
        else:
            print("未找到电池百分比字段！")
            print("返回内容:", state)
            return None

    # def getBatteryStatus(self):
    #     return self.post(
    #         node="pyBatteryServer",
    #         data={
    #             "func_name": "getBatteryMsg"
    #         }
    #     )
    
    # 相机相关
    def getCameraData(self, camera_name="all",type="RGBD"):
        """获取相机数据"""
        data = {
            "func_name": "getCameraData",
            "name":camera_name,
            "type": type
        }
        state = self.post("DeviceCamera", data)
        binary_data = state.encode('latin-1')
        cam_data_msg.ParseFromString(binary_data)
        for cam_data in cam_data_msg.cam_datas:
                # print("-" * 30)
            print(f"Timestamp : {cam_data.header.timestamp}")
            print(f"RGB Width : {cam_data.rgb.width}")
            print(f"RGB Height: {cam_data.rgb.height}")
            print(f"Depth Width : {cam_data.depth.width}")
        # print("getCameraData:", state)
        return state, cam_data_msg.cam_datas[0]
    
    def controlRecognizer(self, reco_file='workbin.srec', tries=1, task_ctrl=10):
        data = {
            "func_name": "ctrlRecognizer",
            "reco_file": reco_file,
            "task_ctrl": task_ctrl,  # 10:开始识别, 12:暂停识别, 13:结束识别
            "tries": tries
        }
        state = self.post("AppRecognition", data)
        print("controlRecognizer:", state)
        return state
    
    def getRecResult(self, task_id="workbin.srec.InstanceGrasp"):
        data = {
            "func_name": "getRecResult",
            "task_id": task_id,
        }
        state = self.post("AppRecognition", data)
        print("getRecResult:", state)
        return state
    
    def HandEyeCalibrationService(self, name):
        data = {
            "request": {
                "service_name": "HandEyeCalibration",
                # "gripper_device_pair": [
                #     {
                #         "device_name": "Camera-001",  # 相机设备名称
                #         "gripper_name": "head1_pitch_link",  # 相机安装部件名称
                #         # "gripper_name": "arm1_ee_link",  # 相机安装部件名称
                #     },
                # ],
                "device_name": [
                    "Camera-000",  # 头部相机设备名称
                    # "Camera-001",  # 头部相机设备名称
                    # "Camera-002",  # 头部相机设备名称
                ],
            }
        }
        # response, data = self.post("Calib", data)
        state = self.post("Calib", data)
        print("HandEyeCalibrationService:", state)
        return state

    def HandEyeCalibrationGetResultService(self, name):
        data = {
            "request": {
                "service_name": "HandEyeCalibrationGetResult",
            }
        }
        # response, data = self.post("Calib", data)
        # return response, data
        state = self.post("Calib", data)
        print("HandEyeCalibrationGetResultService:", state)
        return state
    
    def LiftWaistCalibrationService(self, name):
        data = {
            "request": {
                "service_name": "LiftWaistCalibration",
                # "gripper_device_pair": [
                #     # {"device_name": "Camera-001", "gripper_name": "head1_pitch_link"},
                #     {"device_name": "Camera-002", "gripper_name": "arm1_ee_link"},
                #     # {"device_name": "Camera-002", "gripper_name": "arm2_ee_link"},
                # ],
                "device_name": [
                    "Camera-000",  # 头部相机设备名称
                    # "Camera-001",  # 头部相机设备名称
                    # "Camera-002",  # 头部相机设备名称
                ],
            }
        }
        # response, data = self.post("Calib", data)
        # return response, data
        state = self.post("Calib", data)
        print("LiftWaistCalibrationService:", state)
        return state
    
    def LiftWaistCalibrationGetResultService(self, name):
        data = {
            "request": {
                "service_name": "LiftWaistCalibrationGetResult",
            }
        }
        # response, data = self.post("Calib", data)
        # return response, data
        state = self.post("Calib", data)
        print("启动标定:", state)
        return state

    # 夹爪相关
    # def controlGripper(self, angle=0, velocity=50):
    #     data = {
    #         "func_name": "sendControl",
    #         "band_name": ["Gripper-000", "Gripper-001"],
    #         "param": [{
    #             "finger_name": "thumb",
    #             "bend_angle": angle,
    #             "velocity": velocity
    #           }]
    #     }
    #     state = self.post("GripManager", data)
    #     print("controlGripper:", state)
        
    # def getGripperPos(self):
    #     data = {
    #         "func_name": "getStatus",
    #         "band_name": ["Gripper-000", "Gripper-001"],
    #         "finger_names": ["thumb", "index", "middle"]
    #     }
    #     state = self.post("GripManager", data)
    #     print("getGripperPos:", state)
    #     return state

    # 灵巧手api
    def controlgrip(self, band_name = "Gripper-000", finger_name = "thumb", bend_angle=12.0, velocity=50.0, acceleration=0.0, deceleration=0.0, force=0.0):
        data = {
            "func_name": "sendControl",
            "band_name": [band_name],
            "hand_id": 1,
            "cmd_type":"EtherCAT",
            "param": [
                {
                    "finger_name": finger_name,
                    "bend_angle": bend_angle,
                    "velocity": velocity,
                    "acceleration": acceleration,
                    "deceleration": deceleration,
                    "force": force
                }
            ]
        }
        self.post("GripManager", data)

    # 奥意夹爪api
    def controlRohGrip(self, band_name = ["Gripper-000","Gripper-001"], finger_params = [{
        "finger_name": "index",
        "bend_angle": 40.0,
        "velocity": 50.0
      },
      {
        "finger_name": "middle",
        "bend_angle": 50.0,
        "velocity": 50.0
      },
      {
        "finger_name": "ring",
        "bend_angle": 50.0,
        "velocity": 50.0
      },
      {
        "finger_name": "little",
        "bend_angle": 50.0,
        "velocity": 50.0
      }]):
        data = {
            "func_name": "sendControl",
            "band_name": band_name,
            "hand_id": 2,
            "cmd_type":"EtherCAT",
            "param": finger_params
        }
        self.post("GripManager", data)

    def getGrip(self):
        data = {
            "func_name": "getStatus",
            "band_name": ["Gripper-000","Gripper-001"],
            "finger_names":  ["thumb", "index", "middle","ring","little"]
        }
        state = self.post("GripManager", data)
        print("getGrip:", state)
        return state
    
    # 强脑夹爪api
    def controlBrainCoGrip(self, band_name = ["Gripper-000","Gripper-001"], finger_params = [{ 
        "finger_name": "thumb",
        "bend_angle": 50.0,
        "rotation_angle": 0.0,
        "velocity": 10.0
      },
      {
        "finger_name": "index",
        "bend_angle": 50.0,
        "velocity": 50.0
      },
      {
        "finger_name": "middle",
        "bend_angle": 50.0,
        "velocity": 50.0
      },
      {
        "finger_name": "ring",
        "bend_angle": 50.0,
        "velocity": 50.0
      },
      {
        "finger_name": "little",
        "bend_angle": 50.0,
        "velocity": 50.0
      }]):
        data = {
            "func_name": "sendControl",
            "band_name": band_name,
            "param": finger_params
        }
        self.post("GripManager", data)

    def getBrainCoGrip(self):
        data = {
            "func_name": "getStatus",
            "band_name": ["Gripper-000","Gripper-001"],
            "finger_names":  ["thumb", "index", "middle","ring","little"]
        }
        state = self.post("GripManager", data)
        print("getBrainCoGrip:", state)
        return state
    
    def controlBrainCoTorque(self, band_name = ["Gripper-000","Gripper-001"], finger_params = [
        { "finger_name": "thumb",
         "force": 50.0
        },
       {
        "finger_name": "index",
   
        "force": 50.0
       },
       {
        "finger_name": "middle",

        "force": 50.0
       },
       {
        "finger_name": "ring",

        "force": 50.0
       },
       {
        "finger_name": "little",

        "force": 50.0
      }]):
        data = {
            "func_name": "sendControl",
            "band_name": band_name,
            "command":4,
            "param": finger_params
        }
        self.post("GripManager", data)

    # 智元夹爪api
    def controlZhiYuanGrip(self, band_name = ["Gripper-000","Gripper-001"], finger_params = [{
        "finger_name": "thumb",
        "bend_Angle": 50.0,
        "velocity": 100.0,
        "acceleration": 100.0,
        "deceleration": 100.0,
        "force": 100.0
      }]):
        data = {
            "func_name": "sendControl",
            "band_name": band_name,
            "hand_id": 2,
            "cmd_type":"RS485",
            "param": finger_params
        }
        self.post("GripManager", data)
    
    def getZhiYuanGrip(self):
        data = {
            "func_name": "getStatus",
            "band_name": ["Gripper-000"],
            "finger_names":  ["thumb"]
        }
        state = self.post("GripManager", data)
        print("getZhiYuanGrip:", state)
        return state

    def openTorque(self):
        data = {
            "func_name": "sendControl",
            "band_name": ["Gripper-000","Gripper-001"],
            "command":4,
            "param": [
                { "finger_name": "thumb",
                "force": 50.0
                },
            {
                "finger_name": "index",
                "force": 80.0
            },
            {
                "finger_name": "middle",
                "force": 80.0
            },
            {
                "finger_name": "ring",
                "force": 80.0
            },
            {
                "finger_name": "little",
                "force": 80.0
            }
            ]
        }
        self.post("GripManager", data)

    def closeLeftGrip(self):
        data = {
            "func_name" : "sendControl",
            "band_name": ["Gripper-000"],
            "param": [
                { "finger_name": "thumb",
                    "bend_angle": 60.0,
                    "rotation_angle": 60.0,
                    "velocity": 10.0
                },
                {
                    "finger_name": "index",
                    "bend_angle": 100.0,
                    "velocity": 50.0
                },
                {
                    "finger_name": "middle",
                    "bend_angle": 100.0,
                    "velocity": 50.0
                },
                {
                    "finger_name": "ring",
                    "bend_angle": 100.0,
                    "velocity": 50.0
                },
                {
                    "finger_name": "little",
                    "bend_angle": 100.0,
                    "velocity": 50.0
                }
            ]
        }
        self.post("GripManager", data)

    def closeRightGrip(self):
        data = {
            "func_name" : "sendControl",
            "band_name": ["Gripper-001"],
            "param": [
                { "finger_name": "thumb",
                    "bend_angle": 60.0,
                    "rotation_angle": 60.0,
                    "velocity": 10.0
                },
                {
                    "finger_name": "index",
                    "bend_angle": 100.0,
                    "velocity": 50.0
                },
                {
                    "finger_name": "middle",
                    "bend_angle": 100.0,
                    "velocity": 50.0
                },
                {
                    "finger_name": "ring",
                    "bend_angle": 100.0,
                    "velocity": 50.0
                },
                {
                    "finger_name": "little",
                    "bend_angle": 100.0,
                    "velocity": 50.0
                }
            ]
        }
        self.post("GripManager", data)

    def openLeftGrip(self):
        data = {
            "func_name" : "sendControl",
            "band_name": ["Gripper-000"],
            "param": [
                { "finger_name": "thumb",
                    "bend_angle": 50.0,
                    "rotation_angle": 0.0,
                    "velocity": 60.0
                },
                {
                    "finger_name": "index",
                    "bend_angle": 10.0,
                    "velocity": 50.0
                },
                {
                    "finger_name": "middle",
                    "bend_angle": 10.0,
                    "velocity": 50.0
                },
                {
                    "finger_name": "ring",
                    "bend_angle": 10.0,
                    "velocity": 50.0
                },
                {
                    "finger_name": "little",
                    "bend_angle": 10.0,
                    "velocity": 50.0
                }
            ]
        }
        self.post("GripManager", data)

    def openRightGrip(self):
        data = {
            "func_name" : "sendControl",
            "band_name": ["Gripper-001"],
            "param": [
                { "finger_name": "thumb",
                    "bend_angle": 50.0,
                    "rotation_angle": 0.0,
                    "velocity": 60.0
                },
                {
                    "finger_name": "index",
                    "bend_angle": 10.0,
                    "velocity": 50.0
                },
                {
                    "finger_name": "middle",
                    "bend_angle": 10.0,
                    "velocity": 50.0
                },
                {
                    "finger_name": "ring",
                    "bend_angle": 10.0,
                    "velocity": 50.0
                },
                {
                    "finger_name": "little",
                    "bend_angle": 10.0,
                    "velocity": 50.0
                }
            ]
        }
        self.post("GripManager", data)
    
    # 计时功能
    def starttime(self):
        self.start_timestamp = None
        self.start_time = None
        #记录开始时间
        self.start_time = datetime.now()
        self.start_timestamp = time.time()#开始计时
        print("开始时间:", self.start_time.strftime("%Y-%m-%d %H:%M:%S"))

    
    def endtime(self):
        #记录结束时间
        end_timestamp = time.time()
        end_time = datetime.now()
        print("结束时间:", end_time.strftime("%Y-%m-%d %H:%M:%S"))
        #计算经过时间
        elapsed_time = end_timestamp - self.start_timestamp
        print("运行持续时间: {:.2f} 秒".format(elapsed_time))

    # DO状态
    def getSoftEmc(self):
        data = {
             "func_name": "cmdGetSoftEmc"
        }
        state = self.post("SRC5000DO", data)
        # return state
        # status_data = json.loads(state)
        print("软限位状态:", state)
        return state
    
    def setSoftEmc(self, status: bool = False):
        data = {
             "func_name": "cmdSetSoftEmc",
             "param": {
                 "status": status
             }
        }
        state = self.post("SRC5000DO", data)
        print("设置软限位状态:", state)
        return state
    
    def getSoftEmcNum(self):
        data = {
             "func_name": "cmdGetSoftEmcNum"
        }
        state = self.post("SRC5000DO", data)
        print("软限位编号状态:", state)
        return state

    def getcmdAll(self):
        data = {
             "func_name": "cmdGetAll"
        }
        state = self.post("SRC5000DO", data)
        print("所有DO状态:", state)
        return state

    def getAllDO(self):
        data = {
             "func_name": "cmdGetAllDo"
        }
        state = self.post("SRC5000DO", data)
        print("所有DO状态:", state)
        return state

    def getDOByIndex(self, index: int = 0):
        data = {
             "func_name": "cmdGetDoByIndex",
             "param": {
                 "id": index
             }
        }
        state = self.post("SRC5000DO", data)
        print(f"DO{index}状态:", state)
        return state

    def setDOByIndex(self, index: int = 0, status: bool = False):
        data = {
             "func_name": "cmdSetDo",
             "param": {
                 "id": index,
                 "status": status
             }
        }
        state = self.post("SRC5000DO", data)
        print(f"设置DO{index}状态:", state)
        return state
    
    def getDoNum(self):
        data = {
             "func_name": "cmdGetDoNum"
        }
        state = self.post("SRC5000DO", data)
        print("DO数量:", state)
        return state
    
    def GetAllCanRes(self):
        data = {
             "func_name": "cmdGetAllCanRes"
        }
        state = self.post("SRC5000DO", data)
        print("全部 CAN 终端电阻信息:", state)
        return state
    
    def GetCanResByIndex(self, index: int = 0):
        data = {
             "func_name": "cmdGetCanResByIndex",
             "param": {
                 "id": index
             }
        }
        state = self.post("SRC5000DO", data)
        print(f"CAN 终端电阻信息 {index} :", state)
        return state
    
    def SetCanResByIndex(self, index: int = 0, status: bool = False):
        data = {
             "func_name": "cmdSetCanRes",
             "param": {
                 "id": index,
                 "status": status
             }
        }
        state = self.post("SRC5000DO", data)
        print(f"设置 CAN 终端电阻信息 {index} :", state)
        return state
    
    def GetCanResNum(self):
        data = {
             "func_name": "cmdGetCanResNum"
        }
        state = self.post("SRC5000DO", data)
        print("CAN 终端电阻数量:", state)
        return state
    
    def getRelay(self):
        data = {
             "func_name": "cmdGetRelay"
        }
        state = self.post("SRC5000DO", data)
        print("继电器状态:", state)
        return state
    
    def setRelay(self, status: bool = False):
        data = {
             "func_name": "cmdSetRelay",
             "param": {
                 "status": status
             }
        }
        state = self.post("SRC5000DO", data)
        print("设置继电器状态:", state)
        return state
    
    def getRelayNum(self):
        data = {
             "func_name": "cmdGetRelayNum"
        }
        state = self.post("SRC5000DO", data)
        print("继电器数量:", state)
        return state
    
    # DI状态
    def getAllDI(self):
        data = {
             "func_name": "cmdGetAllDi"
        }
        state = self.post("SRC5000DI", data)
        print("所有DI状态:", state)
        return state
    
    def getDIByIndex(self, index: int = 0):
        data = {
             "func_name": "cmdGetDiByIndex",
             "param": {
                 "id": index
             }
        }
        state = self.post("SRC5000DI", data)
        print(f"DI{index}状态:", state)
        return state
    
    def setDIByIndex(self, index: int = 0, status: bool = True, emulation: bool = True):
        data = {
             "func_name": "cmdSetDiEmulation",
             "param": {
                 "emulation": emulation,
                 "index": index,
                 "status": status
             }
        }
        state = self.post("SRC5000DI", data)
        print(f"设置DI{index}状态:", state)
        return state
    
    def getSoftEmcDI(self):
        data = {
             "func_name": "cmdGetEmc"
        }
        state = self.post("SRC5000DI", data)
        print("getemcmessage:", state)
        return state
    
    def setSoftEmcDI(self, status: bool = True, emulation: bool = True):
        data = {
             "func_name": "cmdSetEmc",
             "param": {
                 "emulation": emulation,
                 "status": status
             }
        }
        state = self.post("SRC5000DI", data)
        print("setsoftemc:", state)
        return state
    
    def getDiNum(self):
        data = {
             "func_name": "cmdGetDiNum"
        }
        state = self.post("SRC5000DI", data)
        print("DI数量:", state)
        return state
    
    # 地图管理
    def GetMapListInfo(self):
        data = {
             "func_name": "getMapListInfo"
        }
        state = self.post("AppMap", data)
        print("获取地图列表信息:", state)
        return state

    def DownloadMap(self, map_type="all"):
        data = {
             "func_name": "downloadMap",
             "type": map_type
        }
        state = self.post("AppMap", data)
        print("下载地图:", state)
        return state
    
    def DownloadMapDir(self, path="default", map_type="dir"):
        data = {
             "func_name": "downloadMap",
             "type": map_type,
             "path": path
        }
        state = self.post("AppMap", data)
        print("下载地图目录:", state)
        return state
    
    def DownloadMapFile(self, path="default/0.smap", map_type="file"):
        data = {
             "func_name": "downloadMap",
             "type": map_type,
             "path": path
        }
        state = self.post("AppMap", data)
        print("下载地图文件:", state)
        return state
    
    def UploadMap(self, map_type="all"):
        data = {
             "func_name": "uploadMap",
             "type": map_type
        }
        state = self.post("AppMap", data)
        print("上传地图:", state)
        return state
    
    def UploadMapDir(self, path="default", map_type="dir"):
        data = {
             "func_name": "uploadMap",
             "type": map_type,
             "path": path
        }
        state = self.post("AppMap", data)
        print("上传地图目录:", state)
        return state
    
    def UploadMapFile(self, path="default/0.smap", map_type="file"):
        data = {
             "func_name": "uploadMap",
             "type": map_type,
             "path": path
        }
        state = self.post("AppMap", data)
        print("上传地图文件:", state)
        return state
    
    def RemoveMap(self, map_name="test1"):
        data = {
             "func_name": "removeMap",
             "map_name": map_name
        }
        state = self.post("AppMap", data)
        print("删除地图:", state)
        return state
    
    def RenameMap(self, from_map="test1", to_map="test2"):
        data = {
             "func_name": "renameMap",
             "from": from_map,
             "to": to_map
        }
        state = self.post("AppMap", data)
        print("重命名地图:", state)
        return state
    
    def UploadAndSwitchMap(self, map_name="test1"):
        data = {
             "func_name": "uploadAndSwitchMap",
             "map_name": map_name
        }
        state = self.post("AppMap", data)
        print("上传并切换地图:", state)
        return state

    def SwitchMap(self, map_name):
        data = {
             "func_name": "switchMap",
             "map_name": map_name
        }
        state = self.post("AppMap", data)
        print("已切换地图")
        return state
    
    def reLocate(self, center_x=0.0, center_y=0.0, center_angle=0.0, length=2.0, angle_scatter=45.0):
        data = {
             "func_name": "reLocate",
             "center_x": center_x,
             "center_y": center_y,
             "center_angle": center_angle,
             "length": length,
             "angle_scatter": angle_scatter
        }
        state = self.post("Localization", data)
        print("重定位:", state)
        return state
    
    def getDeviceNameUsedForLoc(self):
        data = {
             "func_name": "getDeviceNameUsedForLoc"
        }
        state = self.post("Localization", data)
        print("获取用于定位的设备名称:", state)
        return state
    
    def testTF(self, parent_name="x_base_link", child_name="head_laser"):
        data = {
             "func_name": "testTF",
             "parent_name": parent_name,
             "child_name": child_name
        }
        state = self.post("Localization", data)
        print("测试TF:", state)
        return state
    
    def startScanMap(self):
        data = {
             "func_name": "startScanMap",
             "mode": 4
        }
        state = self.post("MapLogger", data)
        print("开始扫描建图:", state)
        return state
    
    def stopScanMap(self):
        data = {
             "func_name": "stopScanMap"
        }
        state = self.post("MapLogger", data)
        print("停止扫描建图:", state)
        return state
    
    def getMappingStatus(self):
        data = {
             "func_name": "getMappingStatus"
        }
        state = self.post("MapLogger", data)
        print("获取建图状态:", state)
        return state
    
    def getMapLogPath(self, index=-1):
        data = {
             "func_name": "getMapLogPath",
             "index": index
        }
        state = self.post("MapLogger", data)
        print("获取地图日志路径:", state)
        return state
    
    # 激光雷达相关
    def getLasersDeviceInfo(self):
        data = {
             "func_name": "getLasersDeviceInfo"
        }
        state = self.post("DeviceLaser", data)
        print("获取激光雷达设备信息:", state)
        return state
    
    # IMU相关
    def openIMUDataStream(self):
        data = {
             "func_name": "openIMUDataStream"
        }
        state = self.post("ServiceSystem", data)
        print("打开IMU数据流:", state)
        return state
    
    def closeIMUDataStream(self):
        data = {
             "func_name": "closeIMUDataStream"
        }
        state = self.post("ServiceSystem", data)
        print("关闭IMU数据流:", state)
        return state
    
    def getIMUInfo(self):
        data = {
             "func_name": "getIMUInfo"
        }
        state = self.post("ServiceSystem", data)
        print("获取IMU信息:", state)
        return state
    
    # 标定相关
    def startTreSteerACOffsetCalib4(self):
        data = {
             "func_name": "startCalib",
             "type": "startCalib",
             "calibType": "TreSteerACOffsetCalib4",
             "deviceType": "Motor",
             "deviceName": "Motor-0011"
        }
        state = self.post("Calib", data)
        print("启动标定:", state)
        return state
    
    # 报错相关
    def getAbnormals(self):
        '''获取异常信息'''
        data = {
            "func_name": "getAbnormals"
        }

        # 临时屏蔽 print 输出
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            state = self.post("AbnormalManager", data)
        except Exception as e:
            state = f"获取异常信息失败: {e}"
        finally:
            # 恢复正常输出
            sys.stdout = old_stdout

        print("错误信息:", state)
        return state
        # status_data = json.loads(state)
        # return status_data['status']

    def clearAbnormals(self, error_codes = [56000, 56001]):
        '''清除异常信息'''
        data = {
            "func_name": "clearAbnormals",
            "error_codes": error_codes
        }
        state = self.post("AbnormalManager", data)
        print("清除错误信息:", state)
        return state

    def getprocessstate(self):
        '''获取进程运行状态信息'''
        data = {
            "func_name": "getChannelsData",
            "data_type": "json",
            "list": [
                {
                    "channel_name": "/System/AppsStatus",
                    "message_type": "rbk4.protocol.MessageV4_AppsStatus"
                }
            ]
        }
        state = self.post("NetProtocol", data)
        print("进程运行状态:", state)
        return state