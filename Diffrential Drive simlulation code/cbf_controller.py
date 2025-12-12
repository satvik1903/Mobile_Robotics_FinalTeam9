#!/usr/bin/env python3
"""
=============================================================================
WAREHOUSE CBF CONTROLLER - Fixed Coordinates & Static Obstacles
=============================================================================
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import numpy as np
import math

try:
    import cvxpy as cp
except ImportError:
    print("ERROR: cvxpy not installed!")
    exit(1)


def normalize_angle(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def quaternion_to_yaw(x, y, z, w):
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


class CBFController(Node):
    def __init__(self):
        super().__init__('cbf_controller')
        
        # =============================================================
        # 1. WORLD CONFIGURATION
        # =============================================================
        self.robot_names = ['turtlebot1', 'turtlebot2']
        
        # EXACT SPAWN POSES FROM WORLD.SDF [x, y, yaw]
        # Robot 1 (Blue) spawns at (0, 4) facing SOUTH (-pi/2)
        # Robot 2 (Green) spawns at (-4, 0) facing EAST (0)
        self.spawn_config = {
            'turtlebot1': {'pos': np.array([0.0, 4.0]), 'yaw': -1.5708},
            'turtlebot2': {'pos': np.array([-4.0, 0.0]), 'yaw': 0.0}
        }

        self.goals = {
            'turtlebot1': np.array([0.0, -4.0]),  # Blue Goal
            'turtlebot2': np.array([4.0, 0.0]),   # Green Goal
        }
        
        # STATIC OBSTACLES (Original Positions)
        # We increase radius slightly to 1.2m to account for the robot's size + margin
        self.static_obstacles = [
            {'pos': np.array([-3.0, 3.0]), 'radius': 1.2},  # Rack 1
            {'pos': np.array([3.0, -3.0]), 'radius': 1.2}   # Rack 2
        ]
        
        # =============================================================
        # 2. CBF PARAMETERS
        # =============================================================
        self.D_SAFE_ROBOTS = 0.8      # Distance between robots
        self.ALPHA = 1.0              # Lower alpha = smoother, less jerky avoidance
        
        # =============================================================
        # 3. CONTROLLER PARAMETERS
        # =============================================================
        self.K_linear = 1.0
        self.K_angular = 4.0
        self.max_linear_vel = 0.6
        self.max_angular_vel = 2.0
        self.goal_tolerance = 0.3
        
        # =============================================================
        # 4. STATE TRACKING
        # =============================================================
        self.current_states = {name: None for name in self.robot_names}
        self.initial_odom = {name: None for name in self.robot_names}
        self.goal_reached = {name: False for name in self.robot_names}
        self.all_odom_received = False
        
        # ROS Setup
        self.cmd_publishers = {}
        self.odom_subscribers = {}
        
        for robot_name in self.robot_names:
            self.cmd_publishers[robot_name] = self.create_publisher(
                Twist, f'/model/{robot_name}/cmd_vel', 10)
            
            self.odom_subscribers[robot_name] = self.create_subscription(
                Odometry, f'/model/{robot_name}/odometry',
                lambda msg, name=robot_name: self.odometry_callback(msg, name),
                10)

        self.control_timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info("CBF Controller Started: Fixed Coordinate Transforms")

    def odometry_callback(self, msg, robot_name):
        # 1. Get Raw Odometry Data (Relative to spawn point)
        raw_x = msg.pose.pose.position.x
        raw_y = msg.pose.pose.position.y
        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        raw_yaw = quaternion_to_yaw(qx, qy, qz, qw)
        
        # 2. Initialize Offset on first message
        if self.initial_odom[robot_name] is None:
            self.initial_odom[robot_name] = {
                'pos': np.array([raw_x, raw_y]),
                'yaw': raw_yaw
            }
            return

        # 3. TRANSFORM RAW ODOM -> WORLD FRAME
        # Get deltas (how much moved since start)
        dx = raw_x - self.initial_odom[robot_name]['pos'][0]
        dy = raw_y - self.initial_odom[robot_name]['pos'][1]
        
        # Get spawn configuration
        spawn_yaw = self.spawn_config[robot_name]['yaw']
        spawn_pos = self.spawn_config[robot_name]['pos']
        
        # Rotate delta by spawn yaw to align with World
        # (This fixes the "Different Direction" bug)
        world_dx = dx * math.cos(spawn_yaw) - dy * math.sin(spawn_yaw)
        world_dy = dx * math.sin(spawn_yaw) + dy * math.cos(spawn_yaw)
        
        # Add to spawn position
        current_world_x = spawn_pos[0] + world_dx
        current_world_y = spawn_pos[1] + world_dy
        
        # Calculate World Yaw
        # World Yaw = (Current Raw Yaw - Initial Raw Yaw) + Spawn Yaw
        dyaw = raw_yaw - self.initial_odom[robot_name]['yaw']
        current_world_yaw = normalize_angle(dyaw + spawn_yaw)
        
        self.current_states[robot_name] = np.array([current_world_x, current_world_y, current_world_yaw])
        
        if not self.all_odom_received:
            if all(s is not None for s in self.current_states.values()):
                self.all_odom_received = True

    def compute_nominal_control(self, robot_name):
        state = self.current_states[robot_name]
        if state is None: return np.array([0.0, 0.0])
        
        goal = self.goals[robot_name]
        dx = goal[0] - state[0]
        dy = goal[1] - state[1]
        dist = math.sqrt(dx*dx + dy*dy)
        
        if dist < self.goal_tolerance:
            self.goal_reached[robot_name] = True
            return np.array([0.0, 0.0])
            
        speed = min(self.K_linear * dist, self.max_linear_vel)
        return np.array([speed * dx/dist, speed * dy/dist])

    def apply_cbf_filter(self, nominal_velocities):
        n_robots = len(self.robot_names)
        velocities = cp.Variable((n_robots, 2))
        constraints = []
        objective_terms = []

        for i, robot_name in enumerate(self.robot_names):
            v_nom = nominal_velocities[robot_name]
            objective_terms.append(cp.sum_squares(velocities[i] - v_nom))
            
            # Max speed limits
            constraints += [velocities[i] >= -self.max_linear_vel,
                            velocities[i] <= self.max_linear_vel]

        # 1. INTER-ROBOT COLLISION AVOIDANCE
        if self.current_states['turtlebot1'] is not None and \
           self.current_states['turtlebot2'] is not None:
            
            p1 = self.current_states['turtlebot1'][:2]
            p2 = self.current_states['turtlebot2'][:2]
            p_rel = p1 - p2
            dist_sq = np.sum(p_rel**2)
            h = dist_sq - self.D_SAFE_ROBOTS**2
            
            constraints.append(
                2 * p_rel @ (velocities[0] - velocities[1]) >= -self.ALPHA * h
            )

        # 2. STATIC OBSTACLE AVOIDANCE (Racks)
        for i, robot_name in enumerate(self.robot_names):
            if self.current_states[robot_name] is None: continue
            p_robot = self.current_states[robot_name][:2]
            
            for obs in self.static_obstacles:
                p_obs = obs['pos']
                r_safe = obs['radius']
                
                diff = p_robot - p_obs
                dist_sq = np.sum(diff**2)
                h_obs = dist_sq - r_safe**2
                
                # If we are dangerously close (within 2.5m), activate constraint
                if dist_sq < 2.5**2:
                    constraints.append(
                        2 * diff @ velocities[i] >= -self.ALPHA * h_obs
                    )

        prob = cp.Problem(cp.Minimize(sum(objective_terms)), constraints)
        try:
            prob.solve(solver=cp.OSQP, verbose=False)
            if prob.status in ['optimal', 'optimal_inaccurate']:
                return {name: velocities[i].value for i, name in enumerate(self.robot_names)}
            else:
                return {n: np.array([0.0, 0.0]) for n in self.robot_names}
        except Exception:
            return {n: np.array([0.0, 0.0]) for n in self.robot_names}

    def velocity_to_twist(self, robot_name, velocity_world):
        state = self.current_states[robot_name]
        theta = state[2] # Robot's World Yaw
        
        cmd = Twist()
        
        # Transform World Velocity -> Robot Body Frame
        # v_x_body = v_x_world * cos(theta) + v_y_world * sin(theta)
        # v_y_body = -v_x_world * sin(theta) + v_y_world * cos(theta)
        
        v_x_world = velocity_world[0]
        v_y_world = velocity_world[1]
        
        # Simple heading logic
        v_mag = math.sqrt(v_x_world**2 + v_y_world**2)
        if v_mag < 0.01: return cmd
        
        desired_heading = math.atan2(v_y_world, v_x_world)
        heading_error = normalize_angle(desired_heading - theta)
        
        # Angular control
        cmd.angular.z = max(-self.max_angular_vel, 
                          min(self.max_angular_vel, self.K_angular * heading_error))
        
        # Linear control (only move if aligned)
        if abs(heading_error) < math.pi/2:
            cmd.linear.x = v_mag * math.cos(heading_error)
            cmd.linear.x = max(0.0, min(self.max_linear_vel, cmd.linear.x))
            
        return cmd

    def control_loop(self):
        if not self.all_odom_received: return
        
        # Stop if done
        if all(self.goal_reached.values()):
            for name in self.robot_names:
                self.cmd_publishers[name].publish(Twist())
            return

        nom_vels = {name: self.compute_nominal_control(name) for name in self.robot_names}
        safe_vels = self.apply_cbf_filter(nom_vels)
        
        for name in self.robot_names:
            self.cmd_publishers[name].publish(
                self.velocity_to_twist(name, safe_vels[name])
            )

def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(CBFController())
    rclpy.shutdown()

if __name__ == '__main__':
    main()