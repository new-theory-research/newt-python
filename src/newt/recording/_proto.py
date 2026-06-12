"""The ``robot.RobotState`` protobuf message for NT v0.0.3 episodes.

Built clean-room from the NT episode-format spec (nt-platform
``docs/data-format/index.md`` — read as reference, NEVER imported). The message
is constructed from a hand-built ``FileDescriptorProto`` so no ``protoc``
toolchain is needed: ``protobuf`` compiles the descriptor at first use.

``protobuf`` ships with the ``recording`` extra; this module is imported lazily
by the writer, never at ``import newt`` time. The build is wrapped in a function
so the import cost (and the lantern, if the dep is absent) lands at call time.
"""
from __future__ import annotations

from newt.recording._lantern import require

_CACHE: dict = {}


def _build():
    """Compile the RobotState message type once and cache it.

    Returns ``(RobotState_class, file_descriptor_set_bytes)``.
    """
    if "robot_state" in _CACHE:
        return _CACHE["robot_state"]

    # Lazy + guarded: protobuf is a recording-extra dep.
    protobuf = require("google.protobuf.descriptor_pb2", "protobuf")
    descriptor_pb2 = protobuf
    descriptor_pool = require("google.protobuf.descriptor_pool", "protobuf")
    message_factory = require("google.protobuf.message_factory", "protobuf")

    pool = descriptor_pool.DescriptorPool()

    def build_file_descriptor():
        fdp = descriptor_pb2.FileDescriptorProto()
        fdp.name = "robot_state.proto"
        fdp.package = "robot"
        fdp.syntax = "proto3"

        msg = fdp.message_type.add()
        msg.name = "RobotState"

        double = descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE
        uint32 = descriptor_pb2.FieldDescriptorProto.TYPE_UINT32
        uint64 = descriptor_pb2.FieldDescriptorProto.TYPE_UINT64
        repeated = descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED
        optional = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL

        # (name, number, type, label) — spec table order, gripper-last arrays.
        fields = [
            ("joint_positions", 1, double, repeated),
            ("joint_velocities", 2, double, repeated),
            ("joint_accelerations", 3, double, repeated),
            ("joint_efforts", 4, double, repeated),
            ("joint_external_efforts", 5, double, repeated),
            ("joint_compensation_efforts", 6, double, repeated),
            ("cartesian_velocities", 8, double, repeated),
            ("rotor_temperatures", 12, double, repeated),
            ("driver_temperatures", 13, double, repeated),
            ("output_id", 14, uint32, optional),
            ("output_timestamp", 15, uint64, optional),
        ]
        for name, number, ftype, label in fields:
            f = msg.field.add()
            f.name = name
            f.number = number
            f.type = ftype
            f.label = label
        return fdp

    file_desc_proto = build_file_descriptor()
    pool.Add(file_desc_proto)
    robot_state_desc = pool.FindMessageTypeByName("robot.RobotState")
    robot_state_cls = message_factory.GetMessageClass(robot_state_desc)

    fds = descriptor_pb2.FileDescriptorSet()
    fds.file.add().CopyFrom(file_desc_proto)
    schema_bytes = fds.SerializeToString()

    _CACHE["robot_state"] = (robot_state_cls, schema_bytes)
    return _CACHE["robot_state"]


def robot_state_class():
    """The compiled ``robot.RobotState`` message class."""
    return _build()[0]


def file_descriptor_set() -> bytes:
    """The MCAP protobuf schema payload: a serialized ``FileDescriptorSet`` holding
    the one RobotState file, so a reader can rebuild the message type with no
    external ``.proto``."""
    return _build()[1]
