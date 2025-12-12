#!/bin/bash
PROJECT_DIR="$HOME/Mobileroboticsproject"
SDF_FILE="$PROJECT_DIR/cbf_holonomic_demo.sdf"
CONTROLLER="$PROJECT_DIR/SRC/cbf_holonomic_controller.py"

echo "=========================================="
echo "   CBF HOLONOMIC COLLISION AVOIDANCE"
echo "=========================================="
echo "Blue: (0,0) -> (3,3)"
echo "Red:  (3,0) -> (0,3)"
echo "Safety Distance: 0.5m"
echo "=========================================="

# Cleanup
echo "[1/4] Cleaning up..."
pkill -9 -f "gz sim" 2>/dev/null || true
pkill -9 ruby 2>/dev/null || true
pkill -9 -f parameter_bridge 2>/dev/null || true
sleep 3

# Dependencies
echo "[2/4] Checking dependencies..."
python3 -c "import cvxpy" 2>/dev/null || pip3 install cvxpy osqp -q

# Gazebo
echo "[3/4] Starting Gazebo..."
gz sim "$SDF_FILE" -r &
sleep 7

# ROS2 bridges
source /opt/ros/jazzy/setup.bash
echo "[4/4] Starting bridges..."

# Odom: Gazebo -> ROS2
ros2 run ros_gz_bridge parameter_bridge /robot0/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry &
ros2 run ros_gz_bridge parameter_bridge /robot1/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry &

# Cmd_vel: ROS2 -> Gazebo
ros2 run ros_gz_bridge parameter_bridge /robot0/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist &
ros2 run ros_gz_bridge parameter_bridge /robot1/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist &

sleep 3

echo ""
echo "=========================================="
echo "   SIMULATION RUNNING"
echo "=========================================="
python3 "$CONTROLLER"

# Cleanup on exit
pkill -9 -f "gz sim" 2>/dev/null || true
pkill -9 -f parameter_bridge 2>/dev/null || true
