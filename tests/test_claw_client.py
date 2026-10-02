from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from controller.motion.claw_client import ClawUsbCdcClient


class ClawUsbCdcClientTests(unittest.TestCase):
    def test_commands_are_written_as_the_usb_cdc_protocol_frames(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            device = Path(directory) / "usb-cdc"
            device.touch()
            client = ClawUsbCdcClient(device)

            self.assertEqual(client.open(), "O")
            self.assertEqual(device.read_bytes(), b"O")
            self.assertEqual(client.close(), "C")
            self.assertEqual(device.read_bytes(), b"C")
            self.assertEqual(client.calibrate(), "H")
            self.assertEqual(device.read_bytes(), b"H")
            self.assertEqual(client.turn_to_position(45), "T45")
            self.assertEqual(device.read_bytes(), b"T45")
            self.assertEqual(client.turn_to_position(-90), "T-90")
            self.assertEqual(device.read_bytes(), b"T-90")

    def test_turn_requires_a_whole_number_position(self) -> None:
        client = ClawUsbCdcClient(Path("/dev/null"))
        for invalid in (True, 12.5, float("inf"), "45"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    client.turn_to_position(invalid)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
