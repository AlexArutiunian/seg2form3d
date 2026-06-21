from argparse import Namespace

from sku_rgbd.cli import public_run_config


def test_run_config_redacts_robot_connection_details():
    config = public_run_config(
        Namespace(robot_ip="private-host", robot_api_path="/private/path", camera="robot")
    )
    assert config["robot_ip"] == "<redacted>"
    assert config["robot_api_path"] == "<redacted>"
    assert config["camera"] == "robot"
