from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from smac_benchmark.batch_config import load_batch_benchmark_config


def _resolve_launch_arguments(context):
    config_path_raw = LaunchConfiguration("config").perform(context).strip()
    map_yaml_raw = LaunchConfiguration("map").perform(context).strip()
    planner_params_raw = LaunchConfiguration("planner_params").perform(context).strip()
    costmap_params_raw = LaunchConfiguration("costmap_params").perform(context).strip()
    autostart_raw = LaunchConfiguration("autostart").perform(context).strip()

    if config_path_raw:
        batch_config = load_batch_benchmark_config(Path(config_path_raw))
        if not map_yaml_raw:
            map_yaml_raw = str(batch_config.map_yaml)
        if not planner_params_raw and batch_config.planner_params is not None:
            planner_params_raw = str(batch_config.planner_params)
        if not costmap_params_raw and batch_config.costmap_params is not None:
            costmap_params_raw = str(batch_config.costmap_params)

    if not map_yaml_raw:
        raise RuntimeError("Either 'map' or 'config' must provide a map YAML path.")

    planner_parameters = []
    if planner_params_raw:
        planner_parameters.append(planner_params_raw)
    if costmap_params_raw:
        planner_parameters.append(costmap_params_raw)

    return [
        Node(
            package="nav2_map_server",
            executable="map_server",
            name="map_server",
            output="screen",
            parameters=[
                {"yaml_filename": map_yaml_raw},
            ],
        ),
        Node(
            package="nav2_planner",
            executable="planner_server",
            name="planner_server",
            output="screen",
            parameters=planner_parameters,
        ),
        Node(
            package="nav2_lifecycle_manager",
            executable="lifecycle_manager",
            name="lifecycle_manager_planner_only",
            output="screen",
            parameters=[
                {"autostart": autostart_raw.lower() == "true"},
                {"node_names": ["map_server", "planner_server"]},
            ],
        ),
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="static_tf_map_to_base",
            arguments=["0", "0", "0", "0", "0", "0", "map", "base_link"],
            output="screen",
        ),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "config",
            default_value="",
            description="Path to batch benchmark config YAML used to fill map/planner/costmap paths.",
        ),
        DeclareLaunchArgument(
            "map",
            default_value="",
            description="Path to map yaml (e.g., .../maps/your_map.yaml)",
        ),
        DeclareLaunchArgument(
            "planner_params",
            default_value="",
            description="Path to planner_server params yaml (e.g., .../config/planner_server.yaml)",
        ),
        DeclareLaunchArgument(
            "costmap_params",
            default_value="",
            description="Path to costmap params yaml (e.g., .../config/costmap_inflation_on.yaml)",
        ),
        DeclareLaunchArgument(
            "autostart",
            default_value="true",
            description="Autostart lifecycle nodes",
        ),
        OpaqueFunction(function=_resolve_launch_arguments),
    ])


