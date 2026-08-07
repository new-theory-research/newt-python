"""Regenerate the source-text-under-protobuf validator regression fixture."""

from __future__ import annotations

import json
from pathlib import Path

from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
from mcap.reader import make_reader
from mcap.writer import Writer

FIXTURE_DIR = Path(__file__).parent
PROTO_SOURCE = b"""syntax = "proto3";

package robot;

message RobotState {
  repeated double joint_positions = 1;
  repeated double joint_velocities = 2;
  uint32 output_id = 14;
  uint64 output_timestamp = 15;
}
"""


def robot_state_class():
    """Build the message class used to produce a genuine protobuf payload."""
    file_descriptor = descriptor_pb2.FileDescriptorProto(
        name="robot_state.proto",
        package="robot",
        syntax="proto3",
    )
    message = file_descriptor.message_type.add(name="RobotState")
    repeated = descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED
    optional = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    double = descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE
    uint32 = descriptor_pb2.FieldDescriptorProto.TYPE_UINT32
    uint64 = descriptor_pb2.FieldDescriptorProto.TYPE_UINT64
    for name, number, field_type, label in (
        ("joint_positions", 1, double, repeated),
        ("joint_velocities", 2, double, repeated),
        ("output_id", 14, uint32, optional),
        ("output_timestamp", 15, uint64, optional),
    ):
        message.field.add(name=name, number=number, type=field_type, label=label)

    pool = descriptor_pool.DescriptorPool()
    pool.Add(file_descriptor)
    return message_factory.GetMessageClass(
        pool.FindMessageTypeByName("robot.RobotState")
    )


def generate() -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    (FIXTURE_DIR / "episode.json").write_text(
        json.dumps(
            {"format_version": "0.0.3", "camera_config": {"cameras": []}},
            indent=2,
        )
        + "\n"
    )

    robot_state_type = robot_state_class()
    robot_state = robot_state_type(
        joint_positions=[0.25, -0.5],
        joint_velocities=[1.5, -2.0],
        output_id=7,
        output_timestamp=1_000_000_000,
    )
    payload = robot_state.SerializeToString()
    assert robot_state_type.FromString(payload) == robot_state

    with (FIXTURE_DIR / "data.mcap").open("wb") as stream:
        writer = Writer(stream)
        writer.start()
        schema_id = writer.register_schema(
            name="robot.RobotState",
            encoding="protobuf",
            data=PROTO_SOURCE,
        )
        channel_id = writer.register_channel(
            topic="robot_state/test-arm",
            message_encoding="protobuf",
            schema_id=schema_id,
        )
        writer.add_message(
            channel_id=channel_id,
            log_time=1_000_000_000,
            publish_time=1_000_000_000,
            data=payload,
            sequence=0,
        )
        writer.finish()

    with (FIXTURE_DIR / "data.mcap").open("rb") as stream:
        records = list(make_reader(stream).iter_messages())
    assert len(records) == 1
    schema, channel, message = records[0]
    assert (schema.name, schema.encoding, schema.data) == (
        "robot.RobotState",
        "protobuf",
        PROTO_SOURCE,
    )
    assert channel.topic == "robot_state/test-arm"
    assert robot_state_type.FromString(message.data) == robot_state


if __name__ == "__main__":
    generate()
