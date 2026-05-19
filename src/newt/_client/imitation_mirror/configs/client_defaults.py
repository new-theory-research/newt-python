"""Default robot-client runtime profile shared across setups."""

from __future__ import annotations

from nt._client.imitation_mirror.config import InferenceClientConfig


def get_config(config: InferenceClientConfig) -> InferenceClientConfig:
    """Populate generic client defaults shared by all robot setups.

    Args:
        config: Base client runtime config object.

    Returns:
        InferenceClientConfig: Mutated client runtime config.
    """

    # Set default websocket endpoint for local serving.
    config.server_url = "ws://localhost:8128/ws/inference"

    # Set default control-loop cadence. Default 15fps.
    config.action_interval = 0.0667

    # Replan at the same cadence by default (i.e. no chunking).
    # Users can set max_actions_per_chunk > 0 to enable chunking and
    # decouple action horizon from replan horizon.
    config.max_actions_per_chunk = 0

    # Set default profiling interval.
    config.profile_every = 50

    # Keep follower proprio read disabled by default.
    config.read_proprio_from_follower = False

    # Keep dry-run disabled by default.
    config.dry_run = False

    # Keep server-side recording requests disabled in the shared default
    # profile. Some servers (including the default local websocket endpoint)
    # may not support episode control messages, so recording should be
    # enabled only by profiles targeting compatible deployments.
    config.recording.request_recording = False

    # Keep language prompt unset by default.
    config.language_text = None

    # Enable SSH tunneling by default for remote-inference usage.
    config.ssh.enabled = True

    # Use default SSH user.
    config.ssh.user = "ubuntu"

    # Forward viewer by default when SSH is enabled.
    config.ssh.forward_viewer = True

    # Use standard viewer port.
    config.ssh.viewer_remote_port = 8079

    # Return resolved default client config.
    return config
