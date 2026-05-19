#!/usr/bin/env python3
"""Run the robot client for remote inference.

Mirror of imitation_learning/run_robot_client.py @ 9a18027d.
Tenet T1: client scripts are invariant. Only mechanical adaptations are
permitted (import path rewrites, config spec string). Behavioral logic is
unchanged. Every deviation is logged in brief-201-edge-client closeout.md.

Modifications require the PR label ``tenet-override-t1`` and a reason entry
in the PR description (e.g. "tracking upstream run_robot_client.py @ <sha>").

Example:
    uv run -m nt._client.run_robot_client \\
        --config nt._client.imitation_mirror.configs.client_defaults

Robot-specific settings (camera serials, extrinsics, arm IPs) are loaded
from ``~/.config/nt/nt.toml`` (required). Override the path with
``site_config_path='/path/to/nt.toml'``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from urllib.parse import urlparse

from ml_collections import config_dict

from nt._client.imitation_mirror.config_loader import load_base_cfg_overrides
from nt._client.imitation_mirror import NetworkClient, run_inference_client
from nt._client.imitation_mirror.config import (
    InferenceCameraConfig,
    InferenceClientConfig,
    TrossenBimanualConfig,
    TrossenSingleArmConfig,
    build_default_inference_client_config,
)
from nt._client.imitation_mirror.runtime import normalize_camera_name
from nt._client.imitation_mirror.site_config import SiteConfig, load_site_config


_DEFAULT_CLIENT_CONFIG_SPEC = "nt._client.imitation_mirror.configs.client_defaults"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for robot client runtime."""
    parser = argparse.ArgumentParser(description="Run robot inference client.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional config module/path applied on top of built-in client profiles.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Dotted key=value overrides (e.g. action_interval=0.1).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Camera name resolution (moved from runtime.py)
# ---------------------------------------------------------------------------


def _resolve_camera_names(
    server_names: list[str],
    local_names: list[str],
    camera_mapping: dict[str, str] | None = None,
) -> list[str]:
    """Resolve server camera names to local camera names.

    When ``camera_mapping`` is provided, it is used as an explicit
    server-name -> local-name mapping.  Otherwise, cameras are matched by
    name identity and a ``ValueError`` is raised on any mismatch.

    Returns:
        list[str]: Local camera names in server camera order.
    """
    local_set = set(local_names)

    if camera_mapping is not None:
        resolved: list[str] = []
        for server_name in server_names:
            local_name = camera_mapping.get(server_name)
            if local_name is None:
                raise ValueError(
                    f"camera_mapping is missing an entry for server camera '{server_name}'. "
                    f"Server cameras: {server_names}, mapping keys: {list(camera_mapping.keys())}"
                )
            if local_name not in local_set:
                raise ValueError(
                    f"camera_mapping maps '{server_name}' -> '{local_name}', "
                    f"but '{local_name}' is not a local camera. "
                    f"Available local cameras: {local_names}"
                )
            resolved.append(local_name)
        return resolved

    missing = [n for n in server_names if n not in local_set]
    if not missing:
        return list(server_names)

    raise ValueError(
        f"Server cameras {missing} not found in local cameras {local_names}. "
        f"Set camera_mapping in the client config to map server camera names to local names, "
        f"e.g. camera_mapping={{'{missing[0]}': '<local_camera_name>'}}."
    )


# ---------------------------------------------------------------------------
# Partial config loading (before server connection)
# ---------------------------------------------------------------------------


def _load_partial_client_config(args: argparse.Namespace) -> InferenceClientConfig:
    """Load client config before server connection.

    Applies: built-in defaults → optional user module → CLI overrides.
    """
    config = build_default_inference_client_config()

    config = load_base_cfg_overrides(
        base_config=config,
        spec=_DEFAULT_CLIENT_CONFIG_SPEC,
        overrides=None,
    )

    if args.config is not None and str(args.config).strip() != "":
        config = load_base_cfg_overrides(
            base_config=config,
            spec=args.config,
            overrides=None,
        )

    if args.overrides:
        config = load_base_cfg_overrides(
            base_config=config,
            spec=None,
            overrides=list(args.overrides),
        )

    if not isinstance(config, InferenceClientConfig):
        raise TypeError(
            f"Client config must resolve to InferenceClientConfig, "
            f"got {type(config).__name__}."
        )
    return config


# ---------------------------------------------------------------------------
# Site config application
# ---------------------------------------------------------------------------


def _apply_site_config(
    config: InferenceClientConfig,
    site: SiteConfig,
) -> InferenceClientConfig:
    """Apply site config (nt.toml) — camera serials, arm IPs."""
    if not site.cameras:
        raise ValueError(
            "Site config (nt.toml) must define at least one camera in "
            "[[camera_config.cameras]]. Camera serials and extrinsics are "
            "always read from the site config."
        )

    existing_cameras = config.robot.cameras
    camera_map = {}
    for cam in site.cameras:
        existing = existing_cameras.get(cam.id)
        if existing is not None:
            merged = InferenceCameraConfig(
                serial_number_or_name=cam.serial_number,
                is_wrist_camera=cam.is_wrist_camera,
                fps=int(existing.fps),
                width=int(existing.width),
                height=int(existing.height),
                use_depth=bool(existing.use_depth),
                use_depth_postprocessing=bool(existing.use_depth_postprocessing),
            )
        else:
            merged = InferenceCameraConfig(
                serial_number_or_name=cam.serial_number,
                is_wrist_camera=cam.is_wrist_camera,
            )
        camera_map[cam.id] = merged
    config.robot.cameras = config_dict.ConfigDict(camera_map)

    if not site.arms:
        raise ValueError(
            "Site config (nt.toml) must define at least one arm in "
            "[[robot_config.arms]]. Arm IPs are always read from the site config."
        )

    arm = config.robot.arm
    if isinstance(arm, TrossenSingleArmConfig):
        arm_name = arm.arm_name
        site_arm = site.arm_by_id(arm_name)
        if site_arm is None:
            available = [a.id for a in site.arms]
            raise ValueError(
                f"Arm '{arm_name}' not found in site config. "
                f"Available arms: {available}"
            )
        if not site_arm.follower_ip:
            raise ValueError(
                f"Site config arm '{site_arm.id}' is missing follower_ip "
                "(required for single-arm setup)."
            )
        arm.ip = site_arm.follower_ip
    elif isinstance(arm, TrossenBimanualConfig):
        available_ids = [a.id for a in site.arms]
        for site_arm in site.arms:
            if "left" in site_arm.id.lower():
                if not site_arm.follower_ip:
                    raise ValueError(
                        f"Site config arm '{site_arm.id}' is missing follower_ip."
                    )
                arm.left_arm_ip = site_arm.follower_ip
            elif "right" in site_arm.id.lower():
                if not site_arm.follower_ip:
                    raise ValueError(
                        f"Site config arm '{site_arm.id}' is missing follower_ip."
                    )
                arm.right_arm_ip = site_arm.follower_ip
        if not arm.left_arm_ip:
            raise ValueError(
                f"No arm with 'left' in its ID found in site config. "
                f"Available arm IDs: {available_ids}. "
                f"Bimanual setup requires arm IDs containing 'left' and 'right'."
            )
        if not arm.right_arm_ip:
            raise ValueError(
                f"No arm with 'right' in its ID found in site config. "
                f"Available arm IDs: {available_ids}. "
                f"Bimanual setup requires arm IDs containing 'left' and 'right'."
            )

    return config


# ---------------------------------------------------------------------------
# Full config resolution (after server connection)
# ---------------------------------------------------------------------------


def _resolve_full_config(
    partial_config: InferenceClientConfig,
    args: argparse.Namespace,
    server_config: dict,
) -> InferenceClientConfig:
    """Resolve the final client config using server metadata.

    Applies in order:
        1. Inline arm profile (based on server arm_count)
        2. Site config (nt.toml — camera serials, arm IPs)
        3. Re-apply user config module (so user values win)
        4. Re-apply CLI overrides
        5. Resolve camera names (server → local mapping)
        6. Filter cameras to only those needed by the policy
        7. Validate
    """
    config = partial_config

    # 1. Extract server metadata.
    arm_count = server_config.get("arm_count")
    if arm_count is None:
        raise ValueError("Server config missing arm_count.")
    arm_count = int(arm_count)

    camera_keys = server_config.get("camera_keys")
    if not isinstance(camera_keys, list) or len(camera_keys) == 0:
        raise ValueError("Server config missing non-empty camera_keys.")

    # 2. Inline arm profile.
    config.robot.type = "trossen"
    if arm_count == 1:
        config.robot.arm = TrossenSingleArmConfig()
    elif arm_count == 2:
        config.robot.arm = TrossenBimanualConfig()
    else:
        raise ValueError(f"Unsupported server arm_count={arm_count}.")

    # 3. Apply site config.
    site_path = getattr(config, "site_config_path", None)
    if site_path == "":
        raise ValueError(
            "site_config_path is empty — site config (nt.toml) is required. "
            "Camera serials and arm IPs must be defined in ~/.config/nt/nt.toml."
        )
    site = load_site_config(site_path)
    if site is None:
        from nt._client.imitation_mirror.site_config import DEFAULT_SITE_CONFIG_PATH

        resolved = site_path or str(DEFAULT_SITE_CONFIG_PATH)
        raise FileNotFoundError(
            f"Site config not found at '{resolved}'. "
            "Camera serials and arm IPs must be defined in ~/.config/nt/nt.toml."
        )
    config = _apply_site_config(config, site)

    # 4. Re-apply user config module so user overrides win.
    if args.config is not None and str(args.config).strip() != "":
        config = load_base_cfg_overrides(
            base_config=config,
            spec=args.config,
            overrides=None,
        )

    # 5. Re-apply CLI overrides.
    if args.overrides:
        config = load_base_cfg_overrides(
            base_config=config,
            spec=None,
            overrides=list(args.overrides),
        )

    # 6. Resolve camera names and filter to only needed cameras.
    server_names = [normalize_camera_name(key) for key in camera_keys]
    local_names = [str(name) for name in config.robot.cameras.keys()]
    camera_mapping = getattr(config, "camera_mapping", None)
    if camera_mapping is not None:
        camera_mapping = dict(camera_mapping)
    camera_names = _resolve_camera_names(server_names, local_names, camera_mapping)

    filtered_cameras = {
        name: config.robot.cameras[name]
        for name in camera_names
        if name in config.robot.cameras
    }
    config.robot.cameras = config_dict.ConfigDict(filtered_cameras)

    # 7. Validate.
    if not isinstance(config, InferenceClientConfig):
        raise TypeError(
            f"Final config must be InferenceClientConfig, got {type(config).__name__}."
        )
    config.assert_resolved()

    if float(config.action_interval) <= 0.0:
        raise ValueError(f"action_interval must be > 0, got {config.action_interval}.")
    if int(config.max_actions_per_chunk) < 0:
        raise ValueError(
            f"max_actions_per_chunk must be >= 0, got {config.max_actions_per_chunk}."
        )
    if bool(config.ssh.forward_viewer) and int(config.ssh.viewer_remote_port) <= 0:
        raise ValueError(
            "ssh.viewer_remote_port must be > 0 when ssh.forward_viewer=True, "
            f"got {config.ssh.viewer_remote_port}."
        )

    return config


# ---------------------------------------------------------------------------
# Server config display
# ---------------------------------------------------------------------------


def _print_server_config(server_config: dict[str, object]) -> None:
    """Pretty-print the server runtime config payload."""
    rendered = json.dumps(server_config, indent=2, sort_keys=True, default=str)
    print("Server /config payload:")
    print(rendered)


# ---------------------------------------------------------------------------
# Session runner (connects server once, resolves config, runs loop)
# ---------------------------------------------------------------------------


async def _run_session(
    partial_config: InferenceClientConfig,
    args: argparse.Namespace,
) -> None:
    """Single server connection → config resolution → inference loop."""
    client = NetworkClient(partial_config.server_url)
    await client.connect()
    try:
        server_config = await client.get_config()
        if not isinstance(server_config, dict):
            raise TypeError(
                f"Server /config response must be a dict, got {type(server_config).__name__}."
            )
        _print_server_config(server_config)

        final_config = _resolve_full_config(partial_config, args, server_config)
        await run_inference_client(final_config, client, server_config)
    finally:
        await client.disconnect()


# ---------------------------------------------------------------------------
# Tunnel / direct runner
# ---------------------------------------------------------------------------


def _run_with_optional_tunnel(
    args: argparse.Namespace, config: InferenceClientConfig
) -> None:
    """Run client loop directly or through an SSH tunnel."""
    if not bool(config.ssh.enabled):
        asyncio.run(_run_session(config, args))
        return

    parsed_url = urlparse(config.server_url)
    remote_port = parsed_url.port or 8128
    remote_host = parsed_url.hostname or "localhost"
    viewer_forward_enabled = bool(config.ssh.forward_viewer)
    viewer_remote_port = int(config.ssh.viewer_remote_port)
    ssh_host = str(config.ssh.host).strip() if config.ssh.host is not None else ""
    if ssh_host == "":
        ssh_host = str(remote_host)

    if ssh_host == "":
        raise ValueError("Unable to resolve SSH host from ssh.host or server_url.")

    try:
        from sshtunnel import SSHTunnelForwarder
    except ImportError as exc:
        raise ImportError(
            "'sshtunnel' is required for SSH forwarding. Install with 'pip install sshtunnel'."
        ) from exc

    print(f"Establishing SSH tunnel to {ssh_host}...")

    ssh_args = {
        "ssh_address_or_host": str(ssh_host),
        "ssh_username": str(config.ssh.user),
    }

    if viewer_forward_enabled:
        ssh_args["remote_bind_addresses"] = [
            (str(remote_host), int(remote_port)),
            (str(remote_host), int(viewer_remote_port)),
        ]
    else:
        ssh_args["remote_bind_address"] = (str(remote_host), int(remote_port))

    if config.ssh.identity_file is not None:
        ssh_args["ssh_pkey"] = str(config.ssh.identity_file)

    tunnel = SSHTunnelForwarder(**ssh_args)
    try:
        tunnel.start()

        local_ports = list(getattr(tunnel, "local_bind_ports", []))
        if not local_ports and getattr(tunnel, "local_bind_port", None) is not None:
            local_ports = [int(tunnel.local_bind_port)]
        if len(local_ports) == 0:
            raise RuntimeError("SSH tunnel failed to start (missing local bind port).")
        inference_local_port = int(local_ports[0])

        print(
            f"Tunnel established: localhost:{inference_local_port} -> {remote_host}:{remote_port}"
        )

        tunneled_server_url = parsed_url._replace(
            netloc=f"localhost:{inference_local_port}"
        ).geturl()
        config.server_url = tunneled_server_url

        if viewer_forward_enabled:
            if len(local_ports) < 2:
                raise RuntimeError(
                    "SSH viewer forwarding enabled but viewer local bind port is missing."
                )
            viewer_local_port = int(local_ports[1])
            print(
                f"Viewer tunnel established: http://localhost:{viewer_local_port} "
                f"-> {remote_host}:{viewer_remote_port}"
            )

        asyncio.run(_run_session(config, args))
    finally:
        tunnel.stop()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    """Entrypoint for robot-side inference runtime."""
    args = _parse_args()
    partial_config = _load_partial_client_config(args)
    _run_with_optional_tunnel(args, partial_config)


if __name__ == "__main__":
    main()
