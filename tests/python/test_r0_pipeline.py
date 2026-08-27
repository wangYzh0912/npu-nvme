import binascii
import os
import sys
import unittest

sys.path.insert(0, "python")
from r0_pipeline import crc32_combine  # noqa: E402


class R0PipelineTests(unittest.TestCase):
    def test_crc_combine(self):
        for left, right in ((b"abc", b"def"),
                            (os.urandom(17), os.urandom(4099)),
                            (b"", b"suffix")):
            combined = crc32_combine(binascii.crc32(left) & 0xffffffff,
                                     binascii.crc32(right) & 0xffffffff,
                                     len(right))
            self.assertEqual(combined, binascii.crc32(left + right) & 0xffffffff)


if __name__ == "__main__":
    unittest.main()
