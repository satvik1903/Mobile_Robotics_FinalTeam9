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

        # current velocities
        self.v = 0.0
        self.w = 0.0

        # history storage
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
        self.v = v_cmd
        self.w = w_cmd

        # update position
        self.x += self.v * np.cos(self.theta) * dt
        self.y += self.v * np.sin(self.theta) * dt
        self.theta += self.w * dt
        self.theta = np.arctan2(np.sin(self.theta), np.cos(self.theta))  # Normalize

        # store everything together
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
        dx = goal[0] - robot.x
        dy = goal[1] - robot.y

        # desired velocity
        distance = np.sqrt(dx ** 2 + dy ** 2)
        v_des = min(robot.max_v, 0.5 * distance)

        # desired angular velocity (proportional to angle error)
        angle_to_goal = np.arctan2(dy, dx)
        angle_diff = np.arctan2(np.sin(angle_to_goal - robot.theta),
                                np.cos(angle_to_goal - robot.theta))
        w_des = 3.0 * angle_diff

        return v_des, w_des

    def lidar_scan(self, robot, obstacles, max_range=10.0):
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
                if obstacle.get('type') == 'rectangle':
                    # rectangle obstacle
                    # check intersection with all four edges
                    rect_x = obstacle['x']
                    rect_y = obstacle['y']
                    width = obstacle['width']
                    height = obstacle['height']

                    # four edges of the rectangle
                    edges = [
                        # (x1, y1, x2, y2) for each edge
                        (rect_x - width / 2, rect_y - height / 2, rect_x + width / 2, rect_y - height / 2),  # Bottom
                        (rect_x + width / 2, rect_y - height / 2, rect_x + width / 2, rect_y + height / 2),  # Right
                        (rect_x + width / 2, rect_y + height / 2, rect_x - width / 2, rect_y + height / 2),  # Top
                        (rect_x - width / 2, rect_y + height / 2, rect_x - width / 2, rect_y - height / 2),  # Left
                    ]

                    for edge in edges:
                        x1, y1, x2, y2 = edge

                        # ray-line segment intersection
                        # ray: P = robot_pos + t * ray_dir
                        # line: P = P1 + s * (P2 - P1), where 0 <= s <= 1

                        dx_edge = x2 - x1
                        dy_edge = y2 - y1

                        # solve: robot + t*ray_dir = P1 + s*(P2-P1)
                        denominator = ray_dx * dy_edge - ray_dy * dx_edge

                        if abs(denominator) > 1e-10:  # not parallel
                            dx_to_edge = x1 - robot.x
                            dy_to_edge = y1 - robot.y

                            t = (dx_to_edge * dy_edge - dy_to_edge * dx_edge) / denominator
                            s = (dx_to_edge * ray_dy - dy_to_edge * ray_dx) / denominator

                            if t > 0 and 0 <= s <= 1 and t < closest_distance:
                                closest_distance = t
                                hit_x = robot.x + t * ray_dx
                                hit_y = robot.y + t * ray_dy
                                closest_hit = {
                                    'angle': ray_angle,
                                    'distance': t,
                                    'point': (hit_x, hit_y),
                                    'obstacle': obstacle
                                }
                else:
                    # circular obstacle (original code)
                    to_obs_x = obstacle['x'] - robot.x
                    to_obs_y = obstacle['y'] - robot.y

                    a = ray_dx ** 2 + ray_dy ** 2
                    b = -2 * (ray_dx * to_obs_x + ray_dy * to_obs_y)
                    c = to_obs_x ** 2 + to_obs_y ** 2 - obstacle['radius'] ** 2

                    discriminant = b ** 2 - 4 * a * c

                    if discriminant >= 0:
                        t1 = (-b - np.sqrt(discriminant)) / (2 * a)
                        t2 = (-b + np.sqrt(discriminant)) / (2 * a)
                        t = t1 if t1 > 0 else t2

                        if t > 0 and t < closest_distance:
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

    def cluster_detections(self, detections, cluster_distance=0.3):
        if not detections:
            return []

        # Sort detections by angle for sequential clustering
        sorted_dets = sorted(detections, key=lambda d: d['angle'])

        clusters = []
        current_cluster = [sorted_dets[0]]

        for i in range(1, len(sorted_dets)):
            prev_point = current_cluster[-1]['point']
            curr_point = sorted_dets[i]['point']

            # Distance between consecutive detection points
            dist = np.sqrt((curr_point[0] - prev_point[0]) ** 2 +
                           (curr_point[1] - prev_point[1]) ** 2)

            if dist <= cluster_distance:
                # Same object
                current_cluster.append(sorted_dets[i])
            else:
                # New object
                clusters.append(current_cluster)
                current_cluster = [sorted_dets[i]]

        # Don't forget the last cluster
        clusters.append(current_cluster)

        # Check if first and last clusters should be merged (wrap-around at 0°/360°)
        if len(clusters) > 1:
            first_point = clusters[0][0]['point']
            last_point = clusters[-1][-1]['point']
            wrap_dist = np.sqrt((first_point[0] - last_point[0]) ** 2 +
                                (first_point[1] - last_point[1]) ** 2)

            if wrap_dist <= cluster_distance:
                # merge first and last clusters
                clusters[0] = clusters[-1] + clusters[0]
                clusters.pop()

        return clusters

        # Handle single obstacle (dict) or multiple obstacles (list)
        if isinstance(obstacles, dict):
            obstacles = [obstacles]

        detections = []

        for i in range(self.lidar_rays):
            # Ray angle in global frame
            ray_angle = 2 * np.pi * i / self.lidar_rays

            # ray direction
            ray_dx = np.cos(ray_angle)
            ray_dy = np.sin(ray_angle)

            # find the closest intersection across all obstacles
            closest_hit = None
            closest_distance = max_range

            for obstacle in obstacles:
                if obstacle.get('type') == 'rectangle':
                    # Rectangle obstacle
                    # Check intersection with all four edges
                    rect_x = obstacle['x']
                    rect_y = obstacle['y']
                    width = obstacle['width']
                    height = obstacle['height']

                    # Four edges of the rectangle
                    edges = [
                        # (x1, y1, x2, y2) for each edge
                        (rect_x - width / 2, rect_y - height / 2, rect_x + width / 2, rect_y - height / 2),  # Bottom
                        (rect_x + width / 2, rect_y - height / 2, rect_x + width / 2, rect_y + height / 2),  # Right
                        (rect_x + width / 2, rect_y + height / 2, rect_x - width / 2, rect_y + height / 2),  # Top
                        (rect_x - width / 2, rect_y + height / 2, rect_x - width / 2, rect_y - height / 2),  # Left
                    ]

                    for edge in edges:
                        x1, y1, x2, y2 = edge

                        dx_edge = x2 - x1
                        dy_edge = y2 - y1

                        # solve: robot + t*ray_dir = P1 + s*(P2-P1)
                        denominator = ray_dx * dy_edge - ray_dy * dx_edge

                        if abs(denominator) > 1e-10:  # Not parallel
                            dx_to_edge = x1 - robot.x
                            dy_to_edge = y1 - robot.y

                            t = (dx_to_edge * dy_edge - dy_to_edge * dx_edge) / denominator
                            s = (dx_to_edge * ray_dy - dy_to_edge * ray_dx) / denominator

                            if t > 0 and 0 <= s <= 1 and t < closest_distance:
                                closest_distance = t
                                hit_x = robot.x + t * ray_dx
                                hit_y = robot.y + t * ray_dy
                                closest_hit = {
                                    'angle': ray_angle,
                                    'distance': t,
                                    'point': (hit_x, hit_y),
                                    'obstacle': obstacle
                                }
                else:
                    to_obs_x = obstacle['x'] - robot.x
                    to_obs_y = obstacle['y'] - robot.y

                    a = ray_dx ** 2 + ray_dy ** 2
                    b = -2 * (ray_dx * to_obs_x + ray_dy * to_obs_y)
                    c = to_obs_x ** 2 + to_obs_y ** 2 - obstacle['radius'] ** 2

                    discriminant = b ** 2 - 4 * a * c

                    if discriminant >= 0:
                        t1 = (-b - np.sqrt(discriminant)) / (2 * a)
                        t2 = (-b + np.sqrt(discriminant)) / (2 * a)
                        t = t1 if t1 > 0 else t2

                        if t > 0 and t < closest_distance:
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
        # Scan all obstacles (with occlusion handling built-in)
        all_detections = self.lidar_scan(robot, obstacles)

        if not all_detections:
            # No detections, return safe value
            return 10.0, None, all_detections

        # Cluster detections into separate objects
        clusters = self.cluster_detections(all_detections, cluster_distance=0.3)

        # Store clusters for analysis
        self.last_clusters = clusters

        # find the single closest detection point across ALL obstacles
        closest_detection = min(all_detections, key=lambda d: d['distance'])

        obs_x, obs_y = closest_detection['point']
        dx = robot.x - obs_x
        dy = robot.y - obs_y
        dist_sq = dx ** 2 + dy ** 2

        # Different safety buffers based on obstacle type and priority
        # Check if this detection is from another robot (has 'radius' and small radius)
        obstacle_source = closest_detection['obstacle']
        is_robot = (obstacle_source.get('radius') is not None and
                    obstacle_source.get('radius') <= 0.25 and  # Robot-sized
                    obstacle_source.get('type') != 'rectangle')  # Not a rectangle

        # Check if this robot obstacle is marked as low priority
        is_low_priority = obstacle_source.get('low_priority', False)

        if is_robot and is_low_priority:
            safety_buffer = -0.05  # NEGATIVE buffer = can get closer than actual collision!
        elif is_robot:
            safety_buffer = 0.0  # No buffer for normal robot-robot
        else:
            safety_buffer = 0.03  # 3cm buffer for static obstacles

        h = dist_sq - (robot.radius + safety_buffer) ** 2

        # Create a single critical point dictionary
        critical_point = {
            'point': closest_detection['point'],
            'distance': closest_detection['distance'],
            'angle': closest_detection['angle'],
            'h': h,
            'dx': dx,
            'dy': dy,
            'obstacle': closest_detection['obstacle'],
            'is_robot': is_robot
        }

        # Store for visualization
        self.last_detections = all_detections

        return h, critical_point, all_detections

    def compute_cbf_derivative(self, robot, dx, dy, v, w):
        h_dot = 2 * (dx * v * np.cos(robot.theta) + dy * v * np.sin(robot.theta))
        return h_dot

    def safe_controller(self, robot, goal, obstacles):
        v_des, w_des = self.nominal_controller(robot, goal)

        # Handle both single obstacle (dict) and multiple obstacles (list)
        if isinstance(obstacles, dict):
            obstacles = [obstacles]

        # Compute CBF value for the single closest point
        h_min, critical_point, detections = self.compute_cbf_all_obstacles(robot, obstacles)

        # Store detections for visualization
        self.last_detections = detections

        # only activate CBF if close to any obstacle
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

        # SINGLE-POINT CBF CONSTRAINT with discrete-time compensation
        def constraint(u):
            v, w = u
            h_dot = self.compute_cbf_derivative(robot, dx, dy, v, w)
            # Add small term to account for discrete-time overshoot
            dt = 0.05
            discrete_margin = 0.5 * robot.max_a * dt * dt  # ≈ 0.000625
            return h_dot + self.gamma * (h_min - discrete_margin)

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

        # clamp initial guess to bounds
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
            # QP failed - provide detailed debugging info
            # Check if constraint is even satisfiable at bounds
            test_points = [
                (v_min, 0.0),
                (v_max, 0.0),
                (0.0, w_min),
                (0.0, w_max),
                (v_min, w_min),
                (v_max, w_max),
            ]

            print(f"\n{'=' * 60}")
            print(f"QP FAILED - Debug Information")
            print(f"{'=' * 60}")
            print(f"h_min: {h_min:.4f}")
            print(f"Robot state:")
            print(f"  Position: ({robot.x:.3f}, {robot.y:.3f})")
            print(f"  Heading: {np.degrees(robot.theta):.1f}°")
            print(f"  Current velocities: v={robot.v:.3f} m/s, w={robot.w:.3f} rad/s")
            print(f"Constraint evaluation at test points:")
            for i, (v_test, w_test) in enumerate(test_points):
                h_dot_test = self.compute_cbf_derivative(robot, dx, dy, v_test, w_test)
                dt = 0.05
                discrete_margin = 0.5 * robot.max_a * dt * dt
                constraint_value = h_dot_test + self.gamma * (h_min - discrete_margin)
                print(
                    f"  v={v_test:.3f}, w={w_test:.3f}: constraint={constraint_value:.6f} {'✓' if constraint_value >= 0 else '✗ VIOLATED'}")
            print(f"Angles:")
            print(f"  Angle to obstacle: {np.degrees(angle_to_obs):.1f}°")
            print(f"  Angle diff: {np.degrees(angle_diff):.1f}°")
            print(f"  Heading toward obstacle: {heading_toward_obstacle}")
            print(f"QP solver message: {result.message}")
            print(f"{'=' * 60}\n")

            # Emergency fallback - intelligently escape
            # Test which turn direction increases h more
            h_dot_left = self.compute_cbf_derivative(robot, dx, dy, 0.0, robot.max_w * 0.8)
            h_dot_right = self.compute_cbf_derivative(robot, dx, dy, 0.0, -robot.max_w * 0.8)

            # Choose better direction
            if h_dot_left > h_dot_right:
                w_escape = robot.max_w * 0.8
            else:
                w_escape = -robot.max_w * 0.8

            # Check if we can add small forward velocity while still satisfying constraint
            # Try v=0.05 with the best turn direction
            h_dot_with_v = self.compute_cbf_derivative(robot, dx, dy, 0.05, w_escape)
            dt = 0.05
            discrete_margin = 0.5 * robot.max_a * dt * dt
            constraint_with_v = h_dot_with_v + self.gamma * (h_min - discrete_margin)

            if constraint_with_v >= 0:
                # Safe to move forward slowly
                return 0.05, w_escape, True
            else:
                # Pure rotation only
                return 0.0, w_escape, True


def run_simulation():
    # Initialize two robots - robot 2 starts behind
    robot1 = TurtleBot(0.0, 0.0, 0.0)
    robot2 = TurtleBot(-0.5, 0.5, 0.0)  # Starts 0.5m behind robot 1

    goal1 = np.array([5.0, 1.0])
    goal2 = np.array([5.0, 1.5])  # Goals also closer together

    # Static circular obstacle - positioned to block robot 1's direct path
    static_obstacles = [
        {
            'x': 2.5,  # Center x
            'y': 1.3,  # Center y - moved up by 0.3 (was 1.0)
            'radius': 0.4  # Radius
        }
    ]

    # Simulation parameters
    dt = 0.05
    max_time = 60.0
    time = 0.0

    controller1 = CBFController(gamma=2.0, lidar_rays=72)
    controller2 = CBFController(gamma=2.0, lidar_rays=72)

    # Run simulation
    print("Starting two-robot simulation...")
    print(f"Robot max acceleration: {robot1.max_a} m/s²")
    print(f"Robot max angular acceleration: {robot1.max_alpha} rad/s²")
    print(f"Number of static obstacles: {len(static_obstacles)}")
    print(f"Obstacle type: {'Rectangle' if static_obstacles[0].get('type') == 'rectangle' else 'Circle'}")
    if static_obstacles[0].get('type') != 'rectangle':
        print(
            f"Circle at ({static_obstacles[0]['x']}, {static_obstacles[0]['y']}) with radius {static_obstacles[0]['radius']}")

    qp_failure_count1 = 0
    qp_failure_count2 = 0

    while time < max_time:
        # Check if both goals reached
        dist_to_goal1 = np.sqrt((robot1.x - goal1[0]) ** 2 + (robot1.y - goal1[1]) ** 2)
        dist_to_goal2 = np.sqrt((robot2.x - goal2[0]) ** 2 + (robot2.y - goal2[1]) ** 2)

        if dist_to_goal1 < 0.15 and dist_to_goal2 < 0.15:
            print(f"Both goals reached at t={time:.2f}s")
            break

        # Create dynamic obstacle list for each robot (includes the other robot)
        # Robot 1 sees: static obstacles + robot 2
        obstacles1 = static_obstacles + [
            {'x': robot2.x, 'y': robot2.y, 'radius': robot2.radius}
        ]

        # Robot 2 sees: static obstacles + robot 1
        obstacles2 = static_obstacles + [
            {'x': robot1.x, 'y': robot1.y, 'radius': robot1.radius}
        ]

        # Compute safe control for both robots
        # first, determine which robot is closer to static obstacles
        # this robot gets priority (lower gamma for other robot = less avoidance)

        # Find closest static obstacle for each robot
        h1_static = 10.0
        h2_static = 10.0

        for obs in static_obstacles:
            if obs.get('type') == 'rectangle':
                # For rectangles, use distance to center as approximation
                dist1 = np.sqrt((robot1.x - obs['x']) ** 2 + (robot1.y - obs['y']) ** 2)
                dist2 = np.sqrt((robot2.x - obs['x']) ** 2 + (robot2.y - obs['y']) ** 2)
                h1_static = min(h1_static, dist1 ** 2 - robot1.radius ** 2)
                h2_static = min(h2_static, dist2 ** 2 - robot2.radius ** 2)
            else:
                # For circles
                dist1 = np.sqrt((robot1.x - obs['x']) ** 2 + (robot1.y - obs['y']) ** 2)
                dist2 = np.sqrt((robot2.x - obs['x']) ** 2 + (robot2.y - obs['y']) ** 2)
                h1_static = min(h1_static, (dist1 - obs['radius']) ** 2 - robot1.radius ** 2)
                h2_static = min(h2_static, (dist2 - obs['radius']) ** 2 - robot2.radius ** 2)

        # Robot closer to static obstacle gets priority
        robot1_has_priority = h1_static < h2_static

        # Modify obstacles based on priority
        if robot1_has_priority:
            # Robot 1 has priority - make it less cautious about robot 2
            # Mark robot 2 as "low priority" for robot 1
            obstacles1_modified = static_obstacles + [
                {'x': robot2.x, 'y': robot2.y, 'radius': robot2.radius, 'low_priority': True}
            ]
            obstacles2_modified = static_obstacles + [
                {'x': robot1.x, 'y': robot1.y, 'radius': robot1.radius}
            ]
        else:
            # Robot 2 has priority - make it less cautious about robot 1
            obstacles1_modified = static_obstacles + [
                {'x': robot2.x, 'y': robot2.y, 'radius': robot2.radius}
            ]
            obstacles2_modified = static_obstacles + [
                {'x': robot1.x, 'y': robot1.y, 'radius': robot1.radius, 'low_priority': True}
            ]

        v1, w1, cbf_active1 = controller1.safe_controller(robot1, goal1, obstacles1_modified)
        v2, w2, cbf_active2 = controller2.safe_controller(robot2, goal2, obstacles2_modified)

        # Track QP failures
        if cbf_active1 and v1 == 0.0:
            qp_failure_count1 += 1
        if cbf_active2 and v2 == 0.0:
            qp_failure_count2 += 1

        # Compute CBF values for logging
        h1, critical1, _ = controller1.compute_cbf_all_obstacles(robot1, obstacles1)
        h2, critical2, _ = controller2.compute_cbf_all_obstacles(robot2, obstacles2)

        # Debug: Check if robots are detecting each other using clusters
        if time > 0 and int(time * 10) % 50 == 0:  # Print every 5 seconds
            # Distance between robots
            dist_between = np.sqrt((robot1.x - robot2.x) ** 2 + (robot1.y - robot2.y) ** 2)
            print(f"\nt={time:.1f}s: Distance between robots: {dist_between:.3f}m")

            # Analyze robot1's clusters
            if hasattr(controller1, 'last_clusters') and controller1.last_clusters:
                print(f"  Robot 1 detects {len(controller1.last_clusters)} object(s):")
                for i, cluster in enumerate(controller1.last_clusters):
                    # Calculate cluster center
                    cluster_x = np.mean([det['point'][0] for det in cluster])
                    cluster_y = np.mean([det['point'][1] for det in cluster])
                    cluster_dist = np.sqrt((cluster_x - robot1.x) ** 2 + (cluster_y - robot1.y) ** 2)

                    # Check if this cluster is near robot2
                    dist_to_r2 = np.sqrt((cluster_x - robot2.x) ** 2 + (cluster_y - robot2.y) ** 2)
                    is_robot2 = dist_to_r2 < 0.3

                    print(
                        f"    Cluster {i + 1}: {len(cluster)} points, center at ({cluster_x:.2f}, {cluster_y:.2f}), dist={cluster_dist:.3f}m {'[ROBOT 2]' if is_robot2 else ''}")

            # Analyze robot2's clusters
            if hasattr(controller2, 'last_clusters') and controller2.last_clusters:
                print(f"  Robot 2 detects {len(controller2.last_clusters)} object(s):")
                for i, cluster in enumerate(controller2.last_clusters):
                    # Calculate cluster center
                    cluster_x = np.mean([det['point'][0] for det in cluster])
                    cluster_y = np.mean([det['point'][1] for det in cluster])
                    cluster_dist = np.sqrt((cluster_x - robot2.x) ** 2 + (cluster_y - robot2.y) ** 2)

                    # Check if this cluster is near robot1
                    dist_to_r1 = np.sqrt((cluster_x - robot1.x) ** 2 + (cluster_y - robot1.y) ** 2)
                    is_robot1 = dist_to_r1 < 0.3

                    print(
                        f"    Cluster {i + 1}: {len(cluster)} points, center at ({cluster_x:.2f}, {cluster_y:.2f}), dist={cluster_dist:.3f}m {'[ROBOT 1]' if is_robot1 else ''}")

        # Update both robots
        robot1.update(v1, w1, dt, time, cbf_active1, h1)
        robot2.update(v2, w2, dt, time, cbf_active2, h2)

        time += dt

    print(f"\nSimulation completed at t={time:.2f}s")
    print(f"Robot 1 - QP failures: {qp_failure_count1}, Min CBF: {min(robot1.history['h_values']):.3f}")
    print(f"Robot 2 - QP failures: {qp_failure_count2}, Min CBF: {min(robot2.history['h_values']):.3f}")

    return robot1, robot2, static_obstacles, goal1, goal2, controller1, controller2


# Run simulation
robot1, robot2, obstacles, goal1, goal2, controller1, controller2 = run_simulation()

# Get trajectories from both robots
traj1 = robot1.history
traj2 = robot2.history

# Create figure with four subplots
fig = plt.figure(figsize=(16, 10))

# Top left: Trajectory
ax1 = fig.add_subplot(221)
ax1.set_xlim(-0.5, 6)
ax1.set_ylim(-0.5, 4)
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3)
ax1.set_xlabel('X (m)', fontsize=12)
ax1.set_ylabel('Y (m)', fontsize=12)
ax1.set_title('Two TurtleBots with CBF Safety (Mutual Avoidance)', fontsize=14, fontweight='bold')

# Plot static obstacles
for i, obstacle in enumerate(obstacles):
    if obstacle.get('type') == 'rectangle':
        # Draw rectangle
        from matplotlib.patches import Rectangle

        rect = Rectangle(
            (obstacle['x'] - obstacle['width'] / 2, obstacle['y'] - obstacle['height'] / 2),
            obstacle['width'],
            obstacle['height'],
            color='red', alpha=0.7, label='Static Obstacle'
        )
        ax1.add_patch(rect)
    else:
        # Draw circle
        circle = Circle((obstacle['x'], obstacle['y']), obstacle['radius'],
                        color='red', alpha=0.7, label='Static Obstacle' if i == 0 else '')
        ax1.add_patch(circle)

# Plot Robot 1 trajectory (blue)
ax1.plot(traj1['x'], traj1['y'], 'b-', linewidth=2, alpha=0.8, label='Robot 1')
ax1.plot(goal1[0], goal1[1], 'b*', markersize=15, label='Goal 1')
ax1.plot(traj1['x'][0], traj1['y'][0], 'bo', markersize=10, label='Start 1')
ax1.plot(traj1['x'][-1], traj1['y'][-1], 'bs', markersize=10, label='End 1')

# Plot Robot 2 trajectory (green)
ax1.plot(traj2['x'], traj2['y'], 'g-', linewidth=2, alpha=0.8, label='Robot 2')
ax1.plot(goal2[0], goal2[1], 'g*', markersize=15, label='Goal 2')
ax1.plot(traj2['x'][0], traj2['y'][0], 'go', markersize=10, label='Start 2')
ax1.plot(traj2['x'][-1], traj2['y'][-1], 'gs', markersize=10, label='End 2')

# Plot robot orientations for both robots
step = max(1, len(traj1['x']) // 15)
for i in range(0, len(traj1['x']), step):
    x, y, theta = traj1['x'][i], traj1['y'][i], traj1['theta'][i]
    dx = 0.2 * np.cos(theta)
    dy = 0.2 * np.sin(theta)
    ax1.arrow(x, y, dx, dy, head_width=0.1, head_length=0.08,
              fc='darkblue', ec='darkblue', alpha=0.5)

step = max(1, len(traj2['x']) // 15)
for i in range(0, len(traj2['x']), step):
    x, y, theta = traj2['x'][i], traj2['y'][i], traj2['theta'][i]
    dx = 0.2 * np.cos(theta)
    dy = 0.2 * np.sin(theta)
    ax1.arrow(x, y, dx, dy, head_width=0.1, head_length=0.08,
              fc='darkgreen', ec='darkgreen', alpha=0.5)

# Visualize LIDAR detections at final position for robot 1
if hasattr(controller1, 'last_detections') and controller1.last_detections:
    for det in controller1.last_detections:
        hit_x, hit_y = det['point']
        ax1.plot([robot1.x, hit_x], [robot1.y, hit_y], 'b-', alpha=0.15, linewidth=0.5)
        ax1.plot(hit_x, hit_y, 'bo', markersize=1.5, alpha=0.3)

    closest_det1 = min(controller1.last_detections, key=lambda d: d['distance'])
    cp_x, cp_y = closest_det1['point']
    ax1.plot(cp_x, cp_y, 'ro', markersize=6, alpha=0.8)
    ax1.plot([robot1.x, cp_x], [robot1.y, cp_y], 'r-', linewidth=1.5, alpha=0.6)

# Visualize LIDAR detections at final position for robot 2
if hasattr(controller2, 'last_detections') and controller2.last_detections:
    for det in controller2.last_detections:
        hit_x, hit_y = det['point']
        ax1.plot([robot2.x, hit_x], [robot2.y, hit_y], 'g-', alpha=0.15, linewidth=0.5)
        ax1.plot(hit_x, hit_y, 'go', markersize=1.5, alpha=0.3)

    closest_det2 = min(controller2.last_detections, key=lambda d: d['distance'])
    cp_x, cp_y = closest_det2['point']
    ax1.plot(cp_x, cp_y, 'mo', markersize=6, alpha=0.8)
    ax1.plot([robot2.x, cp_x], [robot2.y, cp_y], 'm-', linewidth=1.5, alpha=0.6)

ax1.legend(loc='upper left', fontsize=9)

# Top right: CBF value over time for both robots
ax2 = fig.add_subplot(222)
time_array1 = np.array(traj1['time'])
time_array2 = np.array(traj2['time'])
ax2.plot(time_array1, traj1['h_values'], 'b-', linewidth=2, label='Robot 1 h(x)')
ax2.plot(time_array2, traj2['h_values'], 'g-', linewidth=2, label='Robot 2 h(x)')
ax2.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Safety Boundary (h=0)')
ax2.fill_between(time_array1, -1, 0, alpha=0.2, color='red', label='Unsafe Region')
ax2.grid(True, alpha=0.3)
ax2.set_xlabel('Time (s)', fontsize=12)
ax2.set_ylabel('CBF Value h(x)', fontsize=12)
ax2.set_title('Control Barrier Functions Over Time', fontsize=14, fontweight='bold')
ax2.legend(loc='upper right', fontsize=10)
ax2.set_ylim(-0.5, max(max(traj1['h_values']), max(traj2['h_values'])) + 0.5)

# Bottom left: Linear velocity over time for both robots
ax3 = fig.add_subplot(223)
ax3.plot(time_array1, traj1['v'], 'b-', linewidth=2, label='Robot 1 velocity')
ax3.plot(time_array2, traj2['v'], 'g-', linewidth=2, label='Robot 2 velocity')
ax3.axhline(y=robot1.max_v, color='r', linestyle='--', linewidth=1, alpha=0.5, label='Max velocity')
ax3.grid(True, alpha=0.3)
ax3.set_xlabel('Time (s)', fontsize=12)
ax3.set_ylabel('Linear Velocity (m/s)', fontsize=12)
ax3.set_title('Linear Velocities Over Time', fontsize=14, fontweight='bold')
ax3.legend(loc='upper right', fontsize=10)
ax3.set_ylim(-0.1, robot1.max_v + 0.1)

# Bottom right: Angular velocity over time for both robots
ax4 = fig.add_subplot(224)
ax4.plot(time_array1, traj1['w'], 'b-', linewidth=2, label='Robot 1 angular velocity')
ax4.plot(time_array2, traj2['w'], 'g-', linewidth=2, label='Robot 2 angular velocity')
ax4.axhline(y=robot1.max_w, color='r', linestyle='--', linewidth=1, alpha=0.5, label='Max velocity')
ax4.axhline(y=-robot1.max_w, color='r', linestyle='--', linewidth=1, alpha=0.5)
ax4.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
ax4.grid(True, alpha=0.3)
ax4.set_xlabel('Time (s)', fontsize=12)
ax4.set_ylabel('Angular Velocity (rad/s)', fontsize=12)
ax4.set_title('Angular Velocities Over Time', fontsize=14, fontweight='bold')
ax4.legend(loc='upper right', fontsize=10)
ax4.set_ylim(-robot1.max_w - 0.2, robot1.max_w + 0.2)

plt.tight_layout()
plt.show()

print(f"\n{'=' * 50}")
print(f"SIMULATION RESULTS")
print(f"{'=' * 50}")
print(f"ROBOT 1:")
print(f"  Trajectory points: {len(traj1['x'])}")
print(f"  Start: ({traj1['x'][0]:.2f}, {traj1['y'][0]:.2f})")
print(f"  End: ({traj1['x'][-1]:.2f}, {traj1['y'][-1]:.2f})")
print(f"  Goal: ({goal1[0]:.2f}, {goal1[1]:.2f})")
print(f"  Min CBF: {min(traj1['h_values']):.4f}")
print(f"  Avg velocity: {np.mean(traj1['v']):.3f} m/s")
print(f"  Max velocity: {np.max(traj1['v']):.3f} m/s")
print(f"\nROBOT 2:")
print(f"  Trajectory points: {len(traj2['x'])}")
print(f"  Start: ({traj2['x'][0]:.2f}, {traj2['y'][0]:.2f})")
print(f"  End: ({traj2['x'][-1]:.2f}, {traj2['y'][-1]:.2f})")
print(f"  Goal: ({goal2[0]:.2f}, {goal2[1]:.2f})")
print(f"  Min CBF: {min(traj2['h_values']):.4f}")
print(f"  Avg velocity: {np.mean(traj2['v']):.3f} m/s")
print(f"  Max velocity: {np.max(traj2['v']):.3f} m/s")
print(f"\nSAFETY:")
print(f"  Both robots safe: {min(traj1['h_values']) >= 0 and min(traj2['h_values']) >= 0}")
print(f"{'=' * 50}")