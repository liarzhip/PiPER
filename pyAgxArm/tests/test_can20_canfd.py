"""Unit tests for CAN 2.0 -> CAN FD tunnel framing (no hardware)."""

from pyAgxArm.protocols.can_protocol.comms.can20_canfd import (
    DEFAULT_RESPONSE_ID,
    Can20CanfdRx,
    CanFdMessage,
    _crc16_ccitt_false,
    _canfd_dlc_slot,
    _pad_to_canfd_dlc,
)

# Sample tunneled CAN FD frame (8-byte Modbus read, same shape as wrist tunnel traffic).
_SAMPLE_ARB_ID = 0x007E0808
_SAMPLE_PAYLOAD = bytes.fromhex("7e0407d000067b4a")


def _sample_canfd() -> CanFdMessage:
    return CanFdMessage(_SAMPLE_ARB_ID, _SAMPLE_PAYLOAD)


def _response_fragments(canfd: CanFdMessage):
    response = CanFdMessage(
        canfd.arbitration_id,
        canfd.data,
        tunnel_id=DEFAULT_RESPONSE_ID,
    )
    return response.msgs


def _reassemble(fragments):
    reassembler = Can20CanfdRx._Reassembler()
    for message in fragments:
        result = reassembler.feed(message.arbitration_id, bytes(message.data))
        if result is not None:
            return result
    raise ValueError("incomplete CAN FD tunnel message")


def test_canfd_tx_header_and_rx_roundtrip():
    canfd = _sample_canfd()

    first_fragment = canfd.msgs[0].data
    assert first_fragment[0] >> 4 == len(canfd.msgs)
    assert first_fragment[1:3] == b"\x55\xAA"
    assert len(first_fragment) == 7
    assert all(m.arbitration_id == 0x711 for m in canfd.msgs)

    result = _reassemble(_response_fragments(canfd))
    assert result.arbitration_id == canfd.arbitration_id
    assert result.data == canfd.data


def test_tunnel_crc_mismatch_warns_and_still_returns_payload(capsys):
    payload = b"\x7e\x04\x00\x00\x00\x06"
    arb_id = 0x007E0806
    reassembler = Can20CanfdRx._Reassembler()
    reassembler._frame_group = 3
    reassembler._arbitration_id = arb_id
    reassembler._payload = bytearray(payload)
    reassembler._next_sequence = 3

    result = reassembler.feed(0x712, bytes((0x33, 0x00, 0x00)))
    assert result is not None
    assert result.data == payload
    out = capsys.readouterr().out
    assert "[WARNING] tunnel CRC mismatch" in out


def test_rx_ignores_can_id_low_byte_for_length():
    """Tunnel reassembly uses frame count + tail CRC, not arb_id & 0xFF."""
    payload = bytes.fromhex("7e040000000c010203040506")
    canfd = CanFdMessage(0x007E0806, payload)  # low byte 0x06, payload 12 bytes
    result = _reassemble(_response_fragments(canfd))
    assert result.data == payload


def test_collector_feed_and_wait():
    canfd = _sample_canfd()
    collector = Can20CanfdRx()
    collector.reset()
    for response in _response_fragments(canfd):
        collector.feed(response)

    canfd_out = collector.wait(timeout=1.0)

    assert canfd_out is not None
    assert canfd_out.arbitration_id == canfd.arbitration_id
    assert canfd_out.data == canfd.data


def test_wait_matching_skips_non_matching_reply():
    wrong = CanFdMessage(0x007F0807, bytes.fromhex("7f03020001518e"))
    right = _sample_canfd()
    collector = Can20CanfdRx()
    collector.reset()
    for response in _response_fragments(wrong):
        collector.feed(response)
    for response in _response_fragments(right):
        collector.feed(response)

    got = collector.wait_matching(
        1.0,
        lambda msg: msg.arbitration_id == _SAMPLE_ARB_ID,
    )
    assert got is not None
    assert got.arbitration_id == _SAMPLE_ARB_ID
    assert got.data == _SAMPLE_PAYLOAD


def test_reassemble_real_device_712_trace():
    """Candump shape from Nero wrist tunnel (0x7F hand reply)."""
    from can.message import Message

    frames = [
        bytes.fromhex("31 55 AA 00 7F 08 07"),
        bytes.fromhex("32 7F 03 02 00 01 51 8E"),
        bytes.fromhex("33 DD 0F"),
    ]
    collector = Can20CanfdRx()
    collector.reset()
    for data in frames:
        collector.feed(Message(arbitration_id=0x712, is_extended_id=False, data=data))

    got = collector.wait(timeout=1.0)
    assert got is not None
    assert got.arbitration_id == 0x007F0807
    assert got.data.hex() == "7f03020001518e"


def test_tx_pads_to_canfd_dlc_slot():
    """Unpadded SDK-length payload is padded before tunnel CRC/framing."""
    payload = bytes.fromhex(
        "7f1003fe000c1803e803e803e803e803e803e803e803e803e803e803e803e8d97e"
    )
    assert len(payload) == 33
    assert _canfd_dlc_slot(len(payload)) == 48

    msg = CanFdMessage(0x007F0821, payload)
    assert msg.data == payload
    assert _pad_to_canfd_dlc(payload) == payload + bytes(15)

    tail = msg.msgs[-1].data
    expected_crc = _crc16_ccitt_false(_pad_to_canfd_dlc(payload))
    assert tail[1:3] == expected_crc.to_bytes(2, "little")


def test_reassemble_device_712_trace_with_stripped_tail_padding():
    """712 reply: board omits 11 trailing 0x00; CRC is over 64-byte CAN FD slot."""
    from can.message import Message

    frames = [
        bytes.fromhex("A1 55 AA 00 7F 08 35"),
        bytes.fromhex("A2 7F 03 30 00 00 00 00"),
        bytes.fromhex("A3 00 00 00 00 00 00 00"),
        bytes.fromhex("A4 00 00 3C 00 5A 00 51"),
        bytes.fromhex("A5 00 51 00 51 00 51 00"),
        bytes.fromhex("A6 91 00 A0 00 82 00 82"),
        bytes.fromhex("A7 00 82 00 82 03 E8 03"),
        bytes.fromhex("A8 E8 03 E8 03 E8 03 E8"),
        bytes.fromhex("A9 03 E8 83 91"),
        bytes.fromhex("AA 59 0C"),
    ]
    expected = bytes.fromhex(
        "7f033000000000000000000000000000"
        "3c005a005100510051005100"
        "9100a0008200820082008203e8"
        "03e803e803e803e803e88391"
    )
    assert len(expected) == 53
    assert _canfd_dlc_slot(len(expected)) == 64

    collector = Can20CanfdRx()
    collector.reset()
    for data in frames:
        collector.feed(Message(arbitration_id=0x712, is_extended_id=False, data=data))

    got = collector.wait(timeout=1.0)
    assert got is not None
    assert got.arbitration_id == 0x007F0835
    assert got.data == expected
