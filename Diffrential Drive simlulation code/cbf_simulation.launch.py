"""
=============================================================================
WAREHOUSE CBF SIMULATION LAUNCH FILE - Two Robot Version
=============================================================================

This launch file starts:
1. The ros_gz_bridge (translates between Gazebo and ROS2)
2. The CBF controller node

Gazebo should already be running with your world.sdf file.

USAGE:
    ros2 launch cbf_collision_avoidance cbf_simulation.launch.py
=============================================================================
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Generate the launch description for the two-robot warehouse simulation."""
    
    # =========================================================================
    # LAUNCH ARGUMENTS
    # =========================================================================
    
    start_gazebo_arg = DeclareLaunchArgument(
        'start_gazebo',
        default_value='false',
        description='Set to true to start Gazebo (default: false, assumes Gazebo is running)'
    )
    
    world_file_arg = DeclareLaunchArgument(
        'world_file',
        default_value=os.path.expanduser('~/world.sdf'),
        description='Path to Gazebo world file'
    )
    
    # =========================================================================
    # GAZEBO (Optional)
    # =========================================================================
    
    gazebo = ExecuteProcess(
        condition=IfCondition(LaunchConfiguration('start_gazebo')),
        cmd=['gz', 'sim', '-r', LaunchConfiguration('world_file')],
        output='screen'
    )
    
    # =========================================================================
    # ROS-GAZEBO BRIDGE
    # =========================================================================
    # Bridge configuration for TWO robots only
    
    robot_names = ['turtlebot1', 'turtlebot2']  # Only two robots now
    
    bridge_args = []
    
    for robot_name in robot_names:
        # Velocity commands: ROS2 → Gazebo
        bridge_args.append(
            f'/model/{robot_name}/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist'
        )
        
        # Odometry: Gazebo → ROS2
        bridge_args.append(
            f'/model/{robot_name}/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry'
        )
    
    # Clock synchronization
    bridge_args.append('/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock')
    
    bridge_node = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=bridge_args,
        output='screen',
        parameters=[{'use_sim_time': True}]
    )
    
    # =========================================================================
    # CBF CONTROLLER NODE
    # =========================================================================
    
    cbf_controller = Node(
        package='cbf_collision_avoidance',
        executable='cbf_controller',
        name='cbf_controller',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )
    
    # =========================================================================
    # RETURN LAUNCH DESCRIPTION
    # =========================================================================
    
    return LaunchDescription([
        start_gazebo_arg,
        world_file_arg,
        gazebo,
        bridge_node,
        cbf_controller,
    ])