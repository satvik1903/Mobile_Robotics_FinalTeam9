#!/usr/bin/env python3
"""
CBF Holonomic Controller - Equal Speed, Symmetric Avoidance
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import numpy as np
import cvxpy as cp
import threading
import time

ROBOT_CONFIG = {
    0: {'name': 'BLUE', 'goal': [3.0, 3.0], 'v_max': 0.5, 'Kp': 1.2, 'gamma': 3.0},
    1: {'name': 'RED', 'goal': [0.0, 3.0], 'v_max': 0.5, 'Kp': 1.2, 'gamma': 3.0}
}

D_SAFE = 0.5
DT = 0.05
GOAL_TOLERANCE = 0.15

class RobotState:
    def __init__(self, rid):
        self.id = rid
        self.cfg = ROBOT_CONFIG[rid]
        self.pos = np.array([0.0, 0.0])
        self.vel = np.array([0.0, 0.0])
        self.goal = np.array(self.cfg['goal'], dtype=float)
        self.reached_goal = False
        self.odom_received = False

class CBFController(Node):
    def __init__(self):
        super().__init__('cbf_holonomic_controller')
        
        self.robots = {i: RobotState(i) for i in ROBOT_CONFIG}
        self.lock = threading.Lock()
        self.running = True
        self.start_time = None
        self.min_dist = float('inf')
        
        # Subscribe to odometry
        for rid in self.robots:
            self.create_subscription(
                Odometry, f'/robot{rid}/odom',
                lambda msg, r=rid: self.odom_cb(msg, r), 10
            )
            self.get_logger().info(f'Subscribed to /robot{rid}/odom')
        
        # Publish velocity commands
        self.vel_pubs = {}
        for rid in self.robots:
            self.vel_pubs[rid] = self.create_publisher(Twist, f'/robot{rid}/cmd_vel', 10)
            self.get_logger().info(f'Publishing to /robot{rid}/cmd_vel')
        
        self.create_timer(DT, self.control)
        self.create_timer(0.5, self.status)
        
        self.get_logger().info('=' * 55)
        self.get_logger().info('  CBF HOLONOMIC - EQUAL SPEED, SYMMETRIC AVOIDANCE')
        self.get_logger().info('=' * 55)
        self.get_logger().info(f'BLUE: (0,0) -> (3,3) | RED: (3,0) -> (0,3)')
        self.get_logger().info(f'Speed: {ROBOT_CONFIG[0]["v_max"]} m/s | D_safe: {D_SAFE}m')
        self.get_logger().info('=' * 55)
    
    def odom_cb(self, msg, rid):
        with self.lock:
            r = self.robots[rid]
            r.pos = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
            r.vel = np.array([msg.twist.twist.linear.x, msg.twist.twist.linear.y])
            r.odom_received = True
            if self.start_time is None and all(x.odom_received for x in self.robots.values()):
                self.start_time = time.time()
                self.get_logger().info('All odom received - Starting control!')
    
    def desired_vel(self, r):
        if r.reached_goal:
            return np.zeros(2)
        d = r.goal - r.pos
        dist = np.linalg.norm(d)
        if dist < GOAL_TOLERANCE:
            r.reached_goal = True
            t = time.time() - self.start_time if self.start_time else 0
            self.get_logger().info(f'{r.cfg["name"]} reached goal at t={t:.1f}s')
            return np.zeros(2)
        u = r.cfg['Kp'] * d
        speed = np.linalg.norm(u)
        return u * r.cfg['v_max'] / speed if speed > r.cfg['v_max'] else u
    
    def cbf_filter(self, ri, u_des):
        u = cp.Variable(2)
        cons = []
        for rj in self.robots.values():
            if rj.id == ri.id:
                continue
            p = ri.pos - rj.pos
            h = p @ p - D_SAFE**2
            cons.append(2 * p @ u >= -ri.cfg['gamma'] * h + 2 * p @ rj.vel)
        v = ri.cfg['v_max']
        cons += [u[0] <= v, u[0] >= -v, u[1] <= v, u[1] >= -v]
        prob = cp.Problem(cp.Minimize(cp.sum_squares(u - u_des)), cons)
        try:
            prob.solve(solver=cp.OSQP, verbose=False)
            if prob.status in ['optimal', 'optimal_inaccurate']:
                return u.value
        except:
            pass
        return np.zeros(2)
    
    def control(self):
        with self.lock:
            if not all(r.odom_received for r in self.robots.values()):
                return
            
            if not self.running:
                for pub in self.vel_pubs.values():
                    pub.publish(Twist())
                return
            
            if all(r.reached_goal for r in self.robots.values()):
                if self.running:
                    t = time.time() - self.start_time if self.start_time else 0
                    self.get_logger().info('=' * 55)
                    self.get_logger().info('ALL ROBOTS REACHED GOALS!')
                    self.get_logger().info(f'Time: {t:.1f}s | Min dist: {self.min_dist:.3f}m')
                    self.get_logger().info('=' * 55)
                    self.running = False
                for pub in self.vel_pubs.values():
                    pub.publish(Twist())
                return
            
            for r in self.robots.values():
                if r.reached_goal:
                    self.vel_pubs[r.id].publish(Twist())
                    continue
                
                u_des = self.desired_vel(r)
                u_safe = self.cbf_filter(r, u_des)
                
                msg = Twist()
                msg.linear.x = float(u_safe[0])
                msg.linear.y = float(u_safe[1])
                self.vel_pubs[r.id].publish(msg)
            
            dist = np.linalg.norm(self.robots[0].pos - self.robots[1].pos)
            self.min_dist = min(self.min_dist, dist)
    
    def status(self):
        if not self.running or self.start_time is None:
            return
        with self.lock:
            r0, r1 = self.robots[0], self.robots[1]
            dist = np.linalg.norm(r0.pos - r1.pos)
            t = time.time() - self.start_time
            s = f't={t:5.1f}s | dist={dist:.2f}m | '
            s += f'BLUE: ({r0.pos[0]:.1f},{r0.pos[1]:.1f}) | '
            s += f'RED: ({r1.pos[0]:.1f},{r1.pos[1]:.1f})'
            if dist < D_SAFE * 1.3:
                s += ' << AVOIDING!'
            self.get_logger().info(s)

def main(args=None):
    rclpy.init(args=args)
    node = CBFController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
