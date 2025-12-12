import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle, Arrow
import matplotlib.patches as patches
import math


class DifferentialDriveRobot:
    """Class representing a differential drive robot"""

    def __init__(self, x=0, y=0, theta=0, width=0.5, wheel_radius=0.1, wheel_distance=0.3):
        """
        Initialize a differential drive robot

        Parameters:
        - x, y: initial position
        - theta: initial orientation (radians)
        - width: robot body width
        - wheel_radius: radius of wheels
        - wheel_distance: distance between wheels
        """
        self.x = x
        self.y = y
        self.theta = theta  # orientation in radians
        self.width = width
        self.wheel_radius = wheel_radius
        self.wheel_distance = wheel_distance

        # Robot specifications
        self.max_speed = 2.0
        self.max_omega = 3.0  # max angular velocity

        # Controller gains
        self.k_rho = 1.5  # distance gain
        self.k_alpha = 2.0  # angle to goal gain
        self.k_beta = -0.5  # heading correction gain

        # State history for visualization
        self.trajectory = [(x, y)]

    def set_goal(self, goal_x, goal_y):
        """Set the goal position for the robot"""
        self.goal_x = goal_x
        self.goal_y = goal_y

    def get_pose(self):
        """Return the current pose (x, y, theta)"""
        return self.x, self.y, self.theta

    def move(self, v_left, v_right, dt):
        """
        Move the robot based on wheel velocities

        Parameters:
        - v_left: left wheel velocity
        - v_right: right wheel velocity
        - dt: time step
        """
        # Limit wheel velocities
        v_left = np.clip(v_left, -self.max_speed, self.max_speed)
        v_right = np.clip(v_right, -self.max_speed, self.max_speed)

        # Calculate linear and angular velocities
        v = (v_right + v_left) / 2.0
        omega = (v_right - v_left) / self.wheel_distance

        # Update pose using differential drive kinematics
        self.x += v * math.cos(self.theta) * dt
        self.y += v * math.sin(self.theta) * dt
        self.theta += omega * dt

        # Normalize theta to [-pi, pi]
        self.theta = ((self.theta + math.pi) % (2 * math.pi)) - math.pi

        # Record trajectory
        self.trajectory.append((self.x, self.y))

    def control_to_goal(self, dt):
        """
        Compute control inputs to move toward the goal

        Returns:
        - v_left, v_right: wheel velocities
        """
        # Calculate error to goal
        dx = self.goal_x - self.x
        dy = self.goal_y - self.y

        # Distance and angle to goal
        rho = math.sqrt(dx ** 2 + dy ** 2)  # distance to goal
        alpha = math.atan2(dy, dx) - self.theta  # angle to goal relative to robot heading
        beta = -self.theta - alpha  # final heading correction

        # Normalize alpha to [-pi, pi]
        alpha = ((alpha + math.pi) % (2 * math.pi)) - math.pi

        # Control law (based on polar coordinates)
        if rho > 0.05:  # Only apply control if we're not very close to goal
            v = self.k_rho * rho
            omega = self.k_alpha * alpha + self.k_beta * beta

            # Limit velocities
            v = np.clip(v, -self.max_speed, self.max_speed)
            omega = np.clip(omega, -self.max_omega, self.max_omega)

            # Convert to wheel velocities
            v_right = v + (omega * self.wheel_distance) / 2
            v_left = v - (omega * self.wheel_distance) / 2
        else:
            # Stop when close to goal
            v_left = 0
            v_right = 0

        return v_left, v_right

    def distance_to_goal(self):
        """Calculate distance to goal"""
        if hasattr(self, 'goal_x') and hasattr(self, 'goal_y'):
            return math.sqrt((self.goal_x - self.x) ** 2 + (self.goal_y - self.y) ** 2)
        return float('inf')

    def at_goal(self, tolerance=0.1):
        """Check if robot is at goal within tolerance"""
        return self.distance_to_goal() < tolerance


class RobotSimulation:
    """Main simulation class for managing robots and visualization"""

    def __init__(self, width=10, height=10):
        """
        Initialize the simulation environment

        Parameters:
        - width, height: dimensions of the simulation area
        """
        self.width = width
        self.height = height
        self.robots = []
        self.goals = []
        self.obstacles = []

        # Simulation parameters
        self.dt = 0.05  # time step
        self.max_time = 30.0  # maximum simulation time

        # Visualization
        self.fig, self.ax = plt.subplots(figsize=(10, 10))

    def add_robot(self, robot):
        """Add a robot to the simulation"""
        self.robots.append(robot)

    def add_goal(self, x, y):
        """Add a goal position"""
        self.goals.append((x, y))

    def add_obstacle(self, x, y, radius):
        """Add an obstacle to the environment"""
        self.obstacles.append((x, y, radius))

    def setup_visualization(self):
        """Set up the visualization"""
        self.ax.set_xlim(-self.width / 2, self.width / 2)
        self.ax.set_ylim(-self.height / 2, self.height / 2)
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_title('Differential Drive Robot Simulation')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_aspect('equal')

        # Draw boundaries
        boundary = Rectangle((-self.width / 2, -self.height / 2), self.width, self.height,
                             fill=False, edgecolor='black', linewidth=2)
        self.ax.add_patch(boundary)

        # Draw obstacles
        for obstacle in self.obstacles:
            x, y, radius = obstacle
            circle = Circle((x, y), radius, color='red', alpha=0.5)
            self.ax.add_patch(circle)

        # Draw goals
        for goal in self.goals:
            x, y = goal
            goal_marker = Circle((x, y), 0.2, color='green', alpha=0.7)
            self.ax.add_patch(goal_marker)

        # Draw robot initial positions
        self.robot_patches = []
        self.trajectory_lines = []

        for i, robot in enumerate(self.robots):
            # Robot body
            robot_patch = Circle((robot.x, robot.y), robot.width / 2,
                                 color=f'C{i}', alpha=0.7, label=f'Robot {i + 1}')
            self.ax.add_patch(robot_patch)

            # Direction indicator
            arrow = Arrow(robot.x, robot.y,
                          0.3 * math.cos(robot.theta), 0.3 * math.sin(robot.theta),
                          width=0.1, color='black')
            self.ax.add_patch(arrow)

            # Trajectory line
            line, = self.ax.plot([], [], color=f'C{i}', linewidth=1, alpha=0.5)

            self.robot_patches.append(robot_patch)
            self.trajectory_lines.append(line)

        self.ax.legend()

    def update(self, frame):
        """Update function for animation"""
        for i, robot in enumerate(self.robots):
            # Control robot to its goal (simple round-robin assignment for multiple robots)
            goal_idx = i % len(self.goals)
            robot.set_goal(self.goals[goal_idx][0], self.goals[goal_idx][1])

            # Compute control inputs
            v_left, v_right = robot.control_to_goal(self.dt)

            # Move robot
            robot.move(v_left, v_right, self.dt)

            # Update visualization
            self.robot_patches[i].center = (robot.x, robot.y)

            # Update trajectory
            if len(robot.trajectory) > 1:
                x_vals, y_vals = zip(*robot.trajectory)
                self.trajectory_lines[i].set_data(x_vals, y_vals)

            # Update direction arrow (remove old, add new)
            for patch in self.ax.patches:
                if isinstance(patch, Arrow) and patch.get_facecolor() == (0.0, 0.0, 0.0, 1.0):
                    patch.remove()

            # Add new direction arrows
            for robot in self.robots:
                arrow = Arrow(robot.x, robot.y,
                              0.3 * math.cos(robot.theta), 0.3 * math.sin(robot.theta),
                              width=0.1, color='black')
                self.ax.add_patch(arrow)

        # Check if all robots have reached their goals
        all_at_goal = all(robot.at_goal() for robot in self.robots)
        if all_at_goal:
            print("All robots have reached their goals!")

        return self.robot_patches + self.trajectory_lines

    def run(self):
        """Run the simulation"""
        self.setup_visualization()

        # Calculate number of frames based on max time
        num_frames = int(self.max_time / self.dt)

        # Create animation
        anim = animation.FuncAnimation(
            self.fig, self.update, frames=num_frames,
            interval=self.dt * 1000, blit=True, repeat=False
        )

        plt.show()

        return anim


def example_single_robot():
    """Example with a single robot"""
    print("Running single robot simulation...")

    # Create simulation environment
    sim = RobotSimulation(width=12, height=12)

    # Add a goal
    sim.add_goal(5, 5)

    # Add some obstacles
    sim.add_obstacle(1, 1, 0.5)
    sim.add_obstacle(-2, 3, 0.7)
    sim.add_obstacle(3, -1, 0.4)

    # Create and add a robot
    robot = DifferentialDriveRobot(x=-4, y=-4, theta=np.pi / 4)
    sim.add_robot(robot)

    # Run the simulation
    anim = sim.run()

    return anim


def example_multiple_robots():
    """Example with multiple robots"""
    print("Running multiple robot simulation...")

    # Create simulation environment
    sim = RobotSimulation(width=15, height=15)

    # Add multiple goals
    sim.add_goal(6, 6)
    sim.add_goal(-6, 6)
    sim.add_goal(6, -6)
    sim.add_goal(-6, -6)

    # Add some obstacles
    for i in range(5):
        x = np.random.uniform(-5, 5)
        y = np.random.uniform(-5, 5)
        radius = np.random.uniform(0.3, 0.8)
        sim.add_obstacle(x, y, radius)

    # Create and add multiple robots
    num_robots = 4
    for i in range(num_robots):
        x = np.random.uniform(-6, 6)
        y = np.random.uniform(-6, 6)
        theta = np.random.uniform(0, 2 * np.pi)
        robot = DifferentialDriveRobot(x=x, y=y, theta=theta)
        sim.add_robot(robot)

    # Run the simulation
    anim = sim.run()

    return anim


if __name__ == "__main__":
    # Choose which example to run
    print("Differential Drive Robot Simulation")
    print("=" * 40)
    print("1. Single robot")
    print("2. Multiple robots")

    choice = input("Select an option (1 or 2): ").strip()

    if choice == "1":
        anim = example_single_robot()
    elif choice == "2":
        anim = example_multiple_robots()
    else:
        print("Invalid choice, running single robot example.")
        anim = example_single_robot()