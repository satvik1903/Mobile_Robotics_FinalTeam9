import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.optimize import minimize


class TurtleBot:
    def __init__(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta
        self.radius = 0.2  # Robot radius
        self.max_v = 0.5  # Max linear velocity (m/s)
        self.max_w = 1.5  # Max angular velocity (rad/s)
        self.max_a = 0.5  # Max linear acceleration (m/s^2)
        self.max_alpha = 2.0  # Max angular acceleration (rad/s^2)

        # Current velocities
        self.v = 0.0
        self.w = 0.0

        # History storage
        self.history = {
            'x': [x],
            'y': [y],
            'theta': [theta],
            'cbf_active': [False],
            'h_values': [0.0],
            'v': [0.0],
            'w': [0.0],
            'time': [0.0]
        }

    def update(self, v_cmd, w_cmd, dt, time, cbf_active, h_value):
        """Update robot state with velocity commands and log all data"""
        # Velocities are now already constrained by QP, just apply them directly
        self.v = v_cmd
        self.w = w_cmd

        # Update position
        self.x += self.v * np.cos(self.theta) * dt
        self.y += self.v * np.sin(self.theta) * dt
        self.theta += self.w * dt
        self.theta = np.arctan2(np.sin(self.theta), np.cos(self.theta))  # Normalize

        # Store everything together
        self.history['x'].append(self.x)
        self.history['y'].append(self.y)
        self.history['theta'].append(self.theta)
        self.history['v'].append(self.v)
        self.history['w'].append(self.w)
        self.history['time'].append(time)
        self.history['cbf_active'].append(cbf_active)
        self.history['h_values'].append(h_value)


class CBFController:
    def __init__(self, gamma=3.0, lidar_rays=72):
        self.gamma = gamma
        self.lidar_rays = lidar_rays

    def nominal_controller(self, robot, goal):
        """Simple proportional controller to drive to goal"""
        dx = goal[0] - robot.x
        dy = goal[1] - robot.y

        # Desired velocity
        distance = np.sqrt(dx ** 2 + dy ** 2)
        v_des = min(0.4, 0.5 * distance)

        # Desired angular velocity (proportional to angle error)
        angle_to_goal = np.arctan2(dy, dx)
        angle_diff = np.arctan2(np.sin(angle_to_goal - robot.theta),
                                np.cos(angle_to_goal - robot.theta))
        w_des = 3.0 * angle_diff

        return v_des, w_des

    def lidar_scan(self, robot, obstacles, max_range=10.0):
        """
        Simulate 360° LIDAR scan around the robot.
        Returns list of (angle, distance, hit_point) for rays that hit obstacles.
        Properly handles occlusion - rays stop at the first obstacle they hit.
        """
        # Handle single obstacle (dict) or multiple obstacles (list)
        if isinstance(obstacles, dict):
            obstacles = [obstacles]

        detections = []

        for i in range(self.lidar_rays):
            # Ray angle in global frame
            ray_angle = 2 * np.pi * i / self.lidar_rays

            # Ray direction
            ray_dx = np.cos(ray_angle)
            ray_dy = np.sin(ray_angle)

            # Find the closest intersection across all obstacles
            closest_hit = None
            closest_distance = max_range

            for obstacle in obstacles:
                # Check intersection with this circular obstacle
                # Vector from robot to obstacle center
                to_obs_x = obstacle['x'] - robot.x
                to_obs_y = obstacle['y'] - robot.y

                # Quadratic equation coefficients: at² + bt + c = 0
                a = ray_dx ** 2 + ray_dy ** 2  # = 1 for unit vector
                b = -2 * (ray_dx * to_obs_x + ray_dy * to_obs_y)
                c = to_obs_x ** 2 + to_obs_y ** 2 - obstacle['radius'] ** 2

                discriminant = b ** 2 - 4 * a * c

                if discriminant >= 0:
                    # Ray hits this circle
                    t1 = (-b - np.sqrt(discriminant)) / (2 * a)
                    t2 = (-b + np.sqrt(discriminant)) / (2 * a)

                    # Take the closest positive intersection (first hit)
                    t = t1 if t1 > 0 else t2

                    if t > 0 and t < closest_distance:
                        # This is the closest hit so far
                        closest_distance = t
                        hit_x = robot.x + t * ray_dx
                        hit_y = robot.y + t * ray_dy
                        closest_hit = {
                            'angle': ray_angle,
                            'distance': t,
                            'point': (hit_x, hit_y),
                            'obstacle': obstacle
                        }

            # Add the closest hit (if any) to detections
            if closest_hit is not None:
                detections.append(closest_hit)

        return detections

    def compute_cbf_all_obstacles(self, robot, obstacles):
        """
        Compute CBF considering all obstacles.
        Performs LIDAR scan that detects all obstacles with proper occlusion.
        Returns the minimum h value and the most critical point.
        """
        # Scan all obstacles (with occlusion handling built-in)
        all_detections = self.lidar_scan(robot, obstacles)

        if not all_detections:
            # No detections, return safe value
            return 10.0, None, all_detections

        # Find the single closest detection point across ALL obstacles
        closest_detection = min(all_detections, key=lambda d: d['distance'])

        obs_x, obs_y = closest_detection['point']
        dx = robot.x - obs_x
        dy = robot.y - obs_y
        dist_sq = dx ** 2 + dy ** 2
        safety_buffer = 0.05  # 5cm extra margin
        h = dist_sq - (robot.radius + safety_buffer) ** 2

        # Create a single critical point dictionary
        critical_point = {
            'point': closest_detection['point'],
            'distance': closest_detection['distance'],
            'angle': closest_detection['angle'],
            'h': h,
            'dx': dx,
            'dy': dy
        }

        # Store for visualization
        self.last_detections = all_detections

        return h, critical_point, all_detections

    def compute_cbf_derivative(self, robot, dx, dy, v, w):
        """
        Compute CBF time derivative for a single point
        h(x) = ||p_robot - p_surface||^2 - R_robot^2
        h_dot = 2*(p_robot - p_surface)^T * p_robot_dot
        """
        h_dot = 2 * (dx * v * np.cos(robot.theta) + dy * v * np.sin(robot.theta))
        return h_dot

    def safe_controller(self, robot, goal, obstacles):
        """CBF-based QP controller with single closest point"""
        v_des, w_des = self.nominal_controller(robot, goal)

        # Handle both single obstacle (dict) and multiple obstacles (list)
        if isinstance(obstacles, dict):
            obstacles = [obstacles]

        # Compute CBF value for the single closest point
        h_min, critical_point, detections = self.compute_cbf_all_obstacles(robot, obstacles)

        # Store detections for visualization
        self.last_detections = detections

        # Only activate CBF if close to any obstacle
        if h_min > 2.0:
            return v_des, w_des, False  # Not close, use nominal control

        # Get the critical point data
        dx = critical_point['dx']
        dy = critical_point['dy']

        # Check if we're heading toward the critical obstacle
        angle_to_obs = np.arctan2(-dy, -dx)
        angle_diff = np.arctan2(np.sin(angle_to_obs - robot.theta),
                                np.cos(angle_to_obs - robot.theta))

        # Calculate angle to goal
        dx_goal = goal[0] - robot.x
        dy_goal = goal[1] - robot.y
        angle_to_goal = np.arctan2(dy_goal, dx_goal)
        angle_to_goal_diff = np.arctan2(np.sin(angle_to_goal - robot.theta),
                                        np.cos(angle_to_goal - robot.theta))

        # Determine turn bias (which direction to turn)
        if abs(angle_to_goal_diff) < np.pi / 4:
            # Goal is roughly ahead - turn away from obstacle
            turn_bias = -np.sign(angle_diff) if angle_diff != 0 else 1.0
        else:
            # Goal is to the side - turn toward goal
            turn_bias = np.sign(angle_to_goal_diff)

        # If pointing toward obstacle and close, we need to turn
        heading_toward_obstacle = abs(angle_diff) < np.pi / 3 and h_min < 1.0

        if heading_toward_obstacle:
            def objective(u):
                v, w = u
                # Moderate penalty on forward velocity
                if h_min < 0.3:
                    v_penalty = 8.0 * v ** 2  # Very close - strong stop
                elif h_min < 0.6:
                    v_penalty = 4.0 * v ** 2  # Close - moderate stop
                else:
                    v_penalty = 2.0 * (v - v_des) ** 2  # Not too close - gentle slowdown

                # Encourage turning
                w_target = turn_bias * robot.max_w * 0.5
                w_penalty = (w - w_target) ** 2

                return v_penalty + 0.5 * w_penalty
        else:
            # Normal QP objective - just stay close to desired
            def objective(u):
                v, w = u
                return (v - v_des) ** 2 + 0.5 * (w - w_des) ** 2

        # SINGLE-POINT CBF CONSTRAINT
        def constraint(u):
            v, w = u
            h_dot = self.compute_cbf_derivative(robot, dx, dy, v, w)
            return h_dot + self.gamma * h_min

        constraints = [{'type': 'ineq', 'fun': constraint}]

        # Velocity bounds with acceleration limits
        dt = 0.05
        v_min = max(0, robot.v - robot.max_a * dt)
        v_max = min(robot.max_v, robot.v + robot.max_a * dt)
        w_min = max(-robot.max_w, robot.w - robot.max_alpha * dt)
        w_max = min(robot.max_w, robot.w + robot.max_alpha * dt)

        bounds = [(v_min, v_max), (w_min, w_max)]

        # Initial guess: bias toward solution
        if heading_toward_obstacle:
            turn_direction = turn_bias
            initial_guess = [0.2, turn_direction * robot.max_w * 0.4]
        else:
            initial_guess = [v_des, w_des]

        # Clamp initial guess to bounds
        initial_guess[0] = np.clip(initial_guess[0], v_min, v_max)
        initial_guess[1] = np.clip(initial_guess[1], w_min, w_max)

        result = minimize(
            objective,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 100, 'ftol': 1e-6}
        )

        if result.success:
            return result.x[0], result.x[1], True
        else:
            # Emergency: turn toward goal if possible, otherwise away from obstacle
            turn_direction = turn_bias
            print(f"QP failed at h={h_min:.3f}, turning")
            return 0.0, turn_direction * robot.max_w * 0.5, True


def run_simulation():
    # Initialize robot
    robot = TurtleBot(0.0, 0.0, 0.0)
    goal = np.array([5.0, 3.0])

    # Multiple obstacles - creating a narrow passage
    obstacles = [
        {'x': 2.2, 'y': 1.7, 'radius': 0.5},
        {'x': 1.5, 'y': -0.2, 'radius': 0.4}
    ]

    # Simulation parameters
    dt = 0.05
    max_time = 60.0
    time = 0.0

    controller = CBFController(gamma=3.0, lidar_rays=72)

    # Run simulation
    print("Starting simulation...")
    print(f"Robot max acceleration: {robot.max_a} m/s²")
    print(f"Robot max angular acceleration: {robot.max_alpha} rad/s²")
    print(f"Number of obstacles: {len(obstacles)}")

    while time < max_time:
        # Check if goal reached
        dist_to_goal = np.sqrt((robot.x - goal[0]) ** 2 + (robot.y - goal[1]) ** 2)
        if dist_to_goal < 0.15:
            print(f"Goal reached at t={time:.2f}s")
            break

        # Compute safe control considering all obstacles
        v, w, cbf_active = controller.safe_controller(robot, goal, obstacles)

        # Compute CBF value for logging
        h, _, _ = controller.compute_cbf_all_obstacles(robot, obstacles)

        # Update robot with all data at once
        robot.update(v, w, dt, time, cbf_active, h)

        time += dt

    print(f"Simulation completed at t={time:.2f}s")
    print(f"Minimum CBF value: {min(robot.history['h_values']):.3f}")

    return robot, obstacles, goal, controller


# Run simulation
robot, obstacles, goal, controller = run_simulation()

# Get trajectory from robot's history
trajectory = robot.history

# Create figure with four subplots
fig = plt.figure(figsize=(16, 10))

# Top left: Trajectory
ax1 = fig.add_subplot(221)
ax1.set_xlim(-0.5, 6)
ax1.set_ylim(-1, 4)
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3)
ax1.set_xlabel('X (m)', fontsize=12)
ax1.set_ylabel('Y (m)', fontsize=12)
ax1.set_title('TurtleBot Trajectory with Single-Point CBF Safety', fontsize=14, fontweight='bold')

# Plot all obstacles
for i, obstacle in enumerate(obstacles):
    circle = Circle((obstacle['x'], obstacle['y']), obstacle['radius'],
                    color='red', alpha=0.7, label='Obstacle' if i == 0 else '')
    ax1.add_patch(circle)

    # Safety boundary
    safety_circle = Circle((obstacle['x'], obstacle['y']),
                           obstacle['radius'] + robot.radius,
                           color='orange', alpha=0.15, linestyle=':',
                           fill=False, linewidth=2,
                           label='Robot Safety Radius' if i == 0 else '')
    ax1.add_patch(safety_circle)

# Color trajectory
ax1.plot(trajectory['x'], trajectory['y'], 'b-', linewidth=2, alpha=0.8, label='Trajectory')

# Plot goal
ax1.plot(goal[0], goal[1], 'g*', markersize=20, label='Goal')

# Plot start and end positions
ax1.plot(trajectory['x'][0], trajectory['y'][0], 'go', markersize=12, label='Start')
ax1.plot(trajectory['x'][-1], trajectory['y'][-1], 'gs', markersize=12, label='End')

# Plot robot orientation at intervals
step = max(1, len(trajectory['x']) // 20)
for i in range(0, len(trajectory['x']), step):
    x, y, theta = trajectory['x'][i], trajectory['y'][i], trajectory['theta'][i]
    dx = 0.25 * np.cos(theta)
    dy = 0.25 * np.sin(theta)
    ax1.arrow(x, y, dx, dy, head_width=0.12, head_length=0.08,
              fc='darkblue', ec='darkblue', alpha=0.6)

# Visualize LIDAR detections at final position
if hasattr(controller, 'last_detections') and controller.last_detections:
    # Draw all LIDAR rays
    for det in controller.last_detections:
        hit_x, hit_y = det['point']
        ax1.plot([robot.x, hit_x], [robot.y, hit_y], 'g-', alpha=0.2, linewidth=0.5)
        ax1.plot(hit_x, hit_y, 'go', markersize=2, alpha=0.4)

    # Highlight critical point (single closest point)
    closest_det = min(controller.last_detections, key=lambda d: d['distance'])
    cp_x, cp_y = closest_det['point']
    ax1.plot(cp_x, cp_y, 'ro', markersize=8, alpha=0.8, label='Critical Point')
    ax1.plot([robot.x, cp_x], [robot.y, cp_y], 'r-', linewidth=2, alpha=0.6)

ax1.legend(loc='upper left', fontsize=9)

# Top right: CBF value over time
ax2 = fig.add_subplot(222)
time_array = np.array(trajectory['time'])
ax2.plot(time_array, trajectory['h_values'], 'b-', linewidth=2, label='h(x)')
ax2.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Safety Boundary (h=0)')
ax2.fill_between(time_array, -1, 0, alpha=0.2, color='red', label='Unsafe Region')
ax2.grid(True, alpha=0.3)
ax2.set_xlabel('Time (s)', fontsize=12)
ax2.set_ylabel('CBF Value h(x)', fontsize=12)
ax2.set_title('Control Barrier Function Over Time', fontsize=14, fontweight='bold')
ax2.legend(loc='upper right', fontsize=10)
ax2.set_ylim(-0.5, max(trajectory['h_values']) + 0.5)

# Bottom left: Linear velocity over time
ax3 = fig.add_subplot(223)
ax3.plot(time_array, trajectory['v'], 'g-', linewidth=2, label='Linear velocity (v)')
ax3.axhline(y=robot.max_v, color='r', linestyle='--', linewidth=1, alpha=0.5, label='Max velocity')
ax3.grid(True, alpha=0.3)
ax3.set_xlabel('Time (s)', fontsize=12)
ax3.set_ylabel('Linear Velocity (m/s)', fontsize=12)
ax3.set_title('Linear Velocity Over Time', fontsize=14, fontweight='bold')
ax3.legend(loc='upper right', fontsize=10)
ax3.set_ylim(-0.1, robot.max_v + 0.1)

# Bottom right: Angular velocity over time
ax4 = fig.add_subplot(224)
ax4.plot(time_array, trajectory['w'], 'purple', linewidth=2, label='Angular velocity (w)')
ax4.axhline(y=robot.max_w, color='r', linestyle='--', linewidth=1, alpha=0.5, label='Max velocity')
ax4.axhline(y=-robot.max_w, color='r', linestyle='--', linewidth=1, alpha=0.5)
ax4.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
ax4.grid(True, alpha=0.3)
ax4.set_xlabel('Time (s)', fontsize=12)
ax4.set_ylabel('Angular Velocity (rad/s)', fontsize=12)
ax4.set_title('Angular Velocity Over Time', fontsize=14, fontweight='bold')
ax4.legend(loc='upper right', fontsize=10)
ax4.set_ylim(-robot.max_w - 0.2, robot.max_w + 0.2)

plt.tight_layout()
plt.show()

print(f"\n{'=' * 50}")
print(f"SIMULATION RESULTS")
print(f"{'=' * 50}")
print(f"Total trajectory points: {len(trajectory['x'])}")
print(f"Start position: ({trajectory['x'][0]:.2f}, {trajectory['y'][0]:.2f})")
print(f"End position: ({trajectory['x'][-1]:.2f}, {trajectory['y'][-1]:.2f})")
print(f"Goal position: ({goal[0]:.2f}, {goal[1]:.2f})")
print(f"Minimum CBF value (h_min): {min(trajectory['h_values']):.4f}")
print(f"Safety maintained: {min(trajectory['h_values']) >= 0}")
print(f"CBF was active for {sum(trajectory['cbf_active'])} time steps")
print(f"Robot traveled from history: {len(robot.history['x'])} points")
print(f"Average linear velocity: {np.mean(trajectory['v']):.3f} m/s")
print(f"Max linear velocity: {np.max(trajectory['v']):.3f} m/s")
print(f"Min linear velocity: {np.min(trajectory['v']):.3f} m/s")
print(f"Average angular velocity magnitude: {np.mean(np.abs(trajectory['w'])):.3f} rad/s")
if hasattr(controller, 'last_detections') and controller.last_detections:
    print(f"Total LIDAR detections at end: {len(controller.last_detections)}")
print(f"{'=' * 50}")