#!/usr/bin/env python3
"""
solution.launch.py  --  ERC 2026 Library Assistant Robot, team entry point.

Launched by the organisers as:

    ros2 launch YOUR_PACKAGE solution.launch.py shelf_column_number:=2 book_colour:=red

THREE THINGS THIS FILE HAS TO GET RIGHT, OR THE SUBMISSION IS NOT EVALUATED

  1. The filename and the argument names.  The Phase 1 document specifies
     shelf_column_number / book_colour; the organisers' own
     competition_run.launch.py declares shelf_number / book_colour.  Both
     spellings are accepted below and merged, because a mismatch means a
     zero, and accepting an extra alias costs nothing.

  2. Nav2 is NOT started by simulation.launch.py.  Nav2 is merely
     pre-installed in the image.  The organisers run simulation.launch.py
     and then this file, so this file has to bring up map_server, AMCL and
     the navigation stack itself.  (The trial timer starts with this
     launch, so that bringup time is on our clock - unavoidable.)

  3. use_sim_time on EVERY node.  Gazebo's real-time factor is below 1.0
     on most machines, so a node on the wall clock will mis-time waits and
     hand Nav2 pose stamps from a different clock than the one TF is on.

SEQUENCING
  Each stage is a separate process that exits 0 when done, and the next is
  started by OnProcessExit.  Stages are deliberately processes rather than
  one node so a crash is contained and so each piece can still be run by
  hand during development.

    1a tuck_arms_once      - arms in before ANY motion (the table is close)
    1b set_initial_pose    - seed AMCL from the known spawn pose
    2  column_detector     - rotate, read the markers, fit the shelf line
    3  approach_column     - Nav2 + precision align, perpendicular to it
    4  book_color_detector - (not wired in yet)
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

TEAM_PACKAGE = 'erc_vision'          # <-- your solution package name


def generate_launch_description():
    nav2_bringup = get_package_share_directory('nav2_bringup')
    pkg_share = get_package_share_directory(TEAM_PACKAGE)

    default_map = os.path.join(pkg_share, 'maps', 'arena_map_corrected.yaml')
    default_params = os.path.join(pkg_share, 'config', 'nav2_params.yaml')
    # The rules require annotated images in a folder named /erc_images/ in
    # the team's repository.  Override erc_images_dir if your checkout puts
    # it elsewhere; the detector clears this folder at the start of each run
    # so it contains only the current trial's output.
    default_images = '/erc_images'

    args = [
        # Spec spelling.
        DeclareLaunchArgument('shelf_column_number', default_value='0',
                              description='Target shelf column 1-5.'),
        # Organiser-script spelling. Whichever is supplied, we use it.
        DeclareLaunchArgument('shelf_number', default_value='0',
                              description='Alias for shelf_column_number.'),
        DeclareLaunchArgument('book_colour', default_value='red',
                              description='Target book colour.'),
        DeclareLaunchArgument('map', default_value=default_map),
        DeclareLaunchArgument('params_file', default_value=default_params),
        DeclareLaunchArgument('erc_images_dir', default_value=default_images),
        DeclareLaunchArgument('nav2_settle_sec', default_value='20.0',
                              description='Delay before seeding AMCL, to let Nav2 activate.'),
        DeclareLaunchArgument('save_debug_frames', default_value='false'),
        DeclareLaunchArgument('standoff_from_marker', default_value='0.85',
                              description='Metres from the marker plane to base_link. '
                                          'At 0.85 the base stops ~0.59 m clear of the shelf.'),
        # See the frame note in set_initial_pose.py before changing these.
        DeclareLaunchArgument('initial_x', default_value='0.0'),
        DeclareLaunchArgument('initial_y', default_value='0.0'),
        DeclareLaunchArgument('initial_yaw', default_value='0.0'),
    ]

    # Take shelf_number when shelf_column_number was left at its default,
    # otherwise take shelf_column_number. Falls back to 1 if neither given.
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

    # --- Stage 1a: arms in, before anything moves ----------------------
    tuck = Node(
        package=TEAM_PACKAGE, executable='tuck_arms_once', name='tuck_arms_once',
        output='screen', parameters=[sim_time],
    )

    # --- Stage 1b: tell AMCL where we are ------------------------------
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
    # ParameterValue with an explicit value_type is required: launch
    # substitutions are strings, and declare_parameter() on the node side
    # declares an int, so passing the raw substitution throws a parameter
    # type exception at startup.
    column_detector = Node(
        package=TEAM_PACKAGE, executable='column_detector', name='column_detector',
        output='screen',
        parameters=[sim_time, {
            'shelf_column_number': ParameterValue(column, value_type=int),
            'erc_images_dir': ParameterValue(
                LaunchConfiguration('erc_images_dir'), value_type=str),
            'purge_images_dir': True,
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
            'purge_images_dir': False,   # stage 2 already cleared it
            'save_debug_frames': ParameterValue(
                LaunchConfiguration('save_debug_frames'), value_type=bool),
            'column_tolerance':0.55,
        }],
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
    ])
