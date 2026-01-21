from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument
import os

def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    share_dir = get_package_share_directory('kachaka_nav2_bringup')
    rviz_config_file = '/home/ting/ros2_ws/src/kachaka_cartographer/rviz/demo_2d.rviz'
    # cartographer_ros_dir = get_package_share_directory('cartographer_ros')
    # rviz_config_file = os.path.join(
    #     cartographer_ros_dir, 'configuration_files', 'demo_2d.rviz')
    cartographer_config_dir = LaunchConfiguration('cartographer_config_dir',
                                                   default=os.path.join(share_dir, 'config'))
    configuration_basename = LaunchConfiguration(
        'configuration_basename', default='cartographer.lua')
    resolution = LaunchConfiguration('resolution', default='0.05')
    publish_period_sec = LaunchConfiguration(
        'publish_period_sec', default='1.0')
    
    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation time'),
        
        DeclareLaunchArgument(
            'resolution',
            default_value=resolution,
            description='Resolution of a grid cell in the published occupancy grid'),
        
        DeclareLaunchArgument(
            'publish_period_sec',
            default_value=publish_period_sec,
            description='OccupancyGrid publishing period'),
        
        # Static TF: odom -> base_footprint (temporary, until odometry provides it)
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='odom_to_base_footprint',
            output='screen',
            parameters=[{'use_sim_time': use_sim_time}],
            arguments=['0', '0', '0', '0', '0', '0', 'odom', 'base_footprint']
        ),
        
        # Static TF: base_footprint -> laser_frame
        # TODO: Replace 'laser_frame' with actual frame name from scan topic!
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='base_footprint_to_laser',
            output='screen',
            parameters=[{'use_sim_time': use_sim_time}],
            arguments=['0', '0', '0.1', '0', '0', '0', 'base_footprint', 'laser_frame']
        ),
        
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            parameters=[{'use_sim_time': use_sim_time}],
            arguments=['-d', rviz_config_file],
        ),
        
        Node(
            package='cartographer_ros',
            executable='cartographer_node',
            output='screen',
            parameters=[{'use_sim_time': use_sim_time}],
            arguments=['-configuration_directory', cartographer_config_dir,
                      '-configuration_basename', configuration_basename],
            remappings=[('scan', '/kachaka/lidar/scan'),
                       ('odom', '/kachaka/odometry/odometry'),
                       ('imu', '/kachaka/imu/imu')]
        ),
        
        Node(
            package='cartographer_ros',
            executable='cartographer_occupancy_grid_node',
            name='cartographer_occupancy_grid_node',
            parameters=[{'use_sim_time': use_sim_time}],
            arguments=['-resolution', resolution, '-publish_period_sec', publish_period_sec])
    ])