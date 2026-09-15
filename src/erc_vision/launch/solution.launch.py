#!/usr/bin/env python3
"""
solution.launch.py  --  ERC 2026 Library Assistant Robot, team entry point.

To lauch run command:

    ros2 launch erc_vision solution.launch.py shelf_column_number:=2 book_colour:=red

"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                            LogInfo, RegisterEventHandler, TimerAction)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

TEAM_PACKAGE = 'erc_vision'


def generate_launch_description():
    nav2_bringup = get_package_share_directory('nav2_bringup')
    pkg_share = get_package_share_directory(TEAM_PACKAGE)

    default_map = os.path.join(pkg_share, 'maps', 'arena_map_corrected.yaml')
    default_params = os.path.join(pkg_share, 'config', 'nav2_params.yaml')
    # Annotated images in folder /erc_images/
    default_images = '/erc_images'

    args = [

        DeclareLaunchArgument('shelf_column_number', default_value='0',
                              description='Target shelf column 1-5.'),
        # Accept any spelling.
        DeclareLaunchArgument('shelf_number', default_value='0',
                              description='Alias for shelf_column_number.'),
        DeclareLaunchArgument('book_colour', default_value='red',
                              description='Target book colour.'),
        DeclareLaunchArgument('map', default_value=default_map),
        DeclareLaunchArgument('params_file', default_value=default_params),
        DeclareLaunchArgument('erc_images_dir', default_value=default_images),

        DeclareLaunchArgument('purge_images_dir', default_value='false',
                              description='Clear /erc_images before the run. '
                                          'Defaults to false so past runs are kept.'),
        DeclareLaunchArgument('nav2_settle_sec', default_value='20.0',
                              description='Delay before seeding AMCL, to let Nav2 activate.'),
        DeclareLaunchArgument('save_debug_frames', default_value='false'),
        DeclareLaunchArgument('standoff_from_marker', default_value='0.85',
                              description='Metres from the marker plane to base_link. '
                                          'At 0.85 the base stops ~0.59 m clear of the shelf.'),

        DeclareLaunchArgument('initial_x', default_value='0.0'),
        DeclareLaunchArgument('initial_y', default_value='0.0'),
        DeclareLaunchArgument('initial_yaw', default_value='0.0'),
    ]


    column = PythonExpression([
        "str(", LaunchConfiguration('shelf_column_number'), ") "
        "if int(", LaunchConfiguration('shelf_column_number'), ") > 0 "
        "else (str(", LaunchConfiguration('shelf_number'), ") "
        "if int(", LaunchConfiguration('shelf_number'), ") > 0 else '1')"
    ])

    sim_time = {'use_sim_time': True}

    # --- Nav2 ----------------------------------------------------------
    localization = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav2_bringup, 'launch', 'localization_launch.py')),
        launch_arguments={
            'map': LaunchConfiguration('map'),
            'params_file': LaunchConfiguration('params_file'),
            'use_sim_time': 'true',
        }.items(),
    )

    navigation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav2_bringup, 'launch', 'navigation_launch.py')),
        launch_arguments={
            'params_file': LaunchConfiguration('params_file'),
            'use_sim_time': 'true',
        }.items(),
    )

    # --- Stage 1a: tuck arms in, before anything moves ----------------------
    tuck = Node(
        package=TEAM_PACKAGE, executable='tuck_arms_once', name='tuck_arms_once',
        output='screen', parameters=[sim_time],
    )

    # --- Stage 1b: tell AMCL the position ------------------------------
    initial_pose = Node(
        package=TEAM_PACKAGE, executable='set_initial_pose', name='set_initial_pose',
        output='screen',
        parameters=[sim_time, {
            'initial_x': ParameterValue(LaunchConfiguration('initial_x'), value_type=float),
            'initial_y': ParameterValue(LaunchConfiguration('initial_y'), value_type=float),
            'initial_yaw': ParameterValue(LaunchConfiguration('initial_yaw'), value_type=float),
        }],
    )

    # --- Stage 2: find the column --------------------------------------

    column_detector = Node(
        package=TEAM_PACKAGE, executable='column_detector', name='column_detector',
        output='screen',
        parameters=[sim_time, {
            'shelf_column_number': ParameterValue(column, value_type=int),
            'erc_images_dir': ParameterValue(
                LaunchConfiguration('erc_images_dir'), value_type=str),
            'purge_images_dir': ParameterValue(
                LaunchConfiguration('purge_images_dir'), value_type=bool),
            'standoff_from_marker': ParameterValue(
                LaunchConfiguration('standoff_from_marker'), value_type=float),
            'save_debug_frames': ParameterValue(
                LaunchConfiguration('save_debug_frames'), value_type=bool),
        }],
    )

    # --- Stage 3: park perpendicular to it ------------------------------
    approach = Node(
        package=TEAM_PACKAGE, executable='approach_column', name='approach_column',
        output='screen', parameters=[sim_time],
    )

    # --- Stage 4: book color --------------------------------------------
    book_detector = Node(
        package=TEAM_PACKAGE, executable='book_color_detector', name='book_color_detector',
        output='screen',
        parameters=[sim_time, {
            'book_colour': ParameterValue(
                LaunchConfiguration('book_colour'), value_type=str),
            'erc_images_dir': ParameterValue(
                LaunchConfiguration('erc_images_dir'), value_type=str),
            'purge_images_dir': False,
            'save_debug_frames': ParameterValue(
                LaunchConfiguration('save_debug_frames'), value_type=bool),
            'column_tolerance': 0.55,
        }],
    )

    # --- Stage 5: grasp the book ----------------------------------------
    grasp = Node(
        package=TEAM_PACKAGE, executable='grasp_book', name='grasp_book',
        output='screen',
        parameters=[sim_time, {
            'save_debug_frames': ParameterValue(
                LaunchConfiguration('save_debug_frames'), value_type=bool),
        }],
    )

    # --- Stage 6: return to start and place in the bin ------------------
    return_place = Node(
        package=TEAM_PACKAGE, executable='return_and_place', name='return_and_place',
        output='screen', parameters=[sim_time],
    )

    return LaunchDescription(args + [
        LogInfo(msg=['ERC solution starting - target column ', column,
                     ', target colour ', LaunchConfiguration('book_colour')]),
        localization,
        navigation,
        tuck,

        # AMCL needs a moment to activate before it will accept /initialpose.
        RegisterEventHandler(OnProcessExit(
            target_action=tuck,
            on_exit=[TimerAction(
                period=LaunchConfiguration('nav2_settle_sec'),
                actions=[initial_pose])],
        )),
        RegisterEventHandler(OnProcessExit(
            target_action=initial_pose, on_exit=[TimerAction(period=5.0, actions=[column_detector])])),
        RegisterEventHandler(OnProcessExit(
            target_action=column_detector, on_exit=[approach])),
        RegisterEventHandler(OnProcessExit(
            target_action=approach, on_exit=[book_detector])),
        RegisterEventHandler(OnProcessExit(
            target_action=book_detector, on_exit=[grasp])),
        RegisterEventHandler(OnProcessExit(
            target_action=grasp, on_exit=[return_place])),
    ])
