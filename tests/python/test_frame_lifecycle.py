import threading
import time
import unittest

import sys
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))

from frame_lifecycle import FrameBufferPool, FrameState


class FrameLifecycleTests(unittest.TestCase):
    def test_unacknowledged_payload_cannot_be_reused_or_mutated(self):
        pool = FrameBufferPool(1)
        slot = pool.acquire(1, 10)
        source = bytearray(b"first-frame")
        pool.publish(slot, source)
        source[:] = b"XXXXXXXXXXX"
        payload, checksum = pool.begin_write(slot)
        self.assertEqual(payload, b"first-frame")
        with self.assertRaises(TimeoutError):
            pool.acquire(2, 11, timeout=0.01)
        with self.assertRaises(RuntimeError):
            pool.publish(slot, b"overwrite-before-ack")
        pool.ack(slot, checksum)
        self.assertEqual(pool.records[0].state, FrameState.FREE)

    def test_slow_writer_releases_backpressure_after_ack(self):
        pool = FrameBufferPool(1)
        first = pool.acquire(1, 1)
        pool.publish(first, b"generation-1")
        payload, checksum = pool.begin_write(first)
        self.assertEqual(payload, b"generation-1")
        result = {}

        def producer():
            second = pool.acquire(2, 2, timeout=1.0)
            pool.publish(second, b"generation-2")
            result["slot"] = second

        thread = threading.Thread(target=producer)
        thread.start()
        time.sleep(0.03)
        self.assertTrue(thread.is_alive())
        pool.ack(first, checksum)
        thread.join(timeout=1.0)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result["slot"], first)

    def test_bad_ack_and_generation_are_hard_failures(self):
        pool = FrameBufferPool(2)
        slot = pool.acquire(7, 70)
        with self.assertRaises(ValueError):
            pool.acquire(7, 71)
        pool.publish(slot, b"payload")
        pool.begin_write(slot)
        with self.assertRaises(ValueError):
            pool.ack(slot, "wrong")
        self.assertEqual(pool.records[slot].state, FrameState.WRITING)
        pool.fail(slot, TimeoutError("injected"))
        self.assertEqual(pool.records[slot].state, FrameState.FREE)


if __name__ == "__main__":
    unittest.main()
