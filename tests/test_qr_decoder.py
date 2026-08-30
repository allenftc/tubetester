from __future__ import annotations

import io
import unittest

import qrcode

from controller.vision.qr_decoder import QrDecoder


class QrDecoderTests(unittest.TestCase):
    def test_decode_generated_qr_code(self) -> None:
        payload = "tube-12345"
        image = qrcode.make(payload, box_size=6, border=2)

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")

        decoder = QrDecoder(library="opencv")
        result = decoder.decode(buffer.getvalue(), frame_id=7)

        self.assertEqual(result.payload, payload)
        self.assertGreaterEqual(result.confidence, 0.1)
        self.assertEqual(result.frame_id, 7)


if __name__ == "__main__":
    unittest.main()
