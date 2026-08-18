from __future__ import annotations

from datetime import datetime, timedelta, timezone
import unittest

from controller.web.events import EventStore, classify_level, sanitize_message


class EventStoreTests(unittest.TestCase):
    def test_classification_and_redaction(self) -> None:
        self.assertEqual(classify_level("Unknown command: TEST"), "error")
        self.assertEqual(classify_level("timeout while retrying"), "warning")
        self.assertEqual(classify_level("healthy"), "info")
        self.assertEqual(sanitize_message("Authorization: Bearer secret"), "Authorization: [REDACTED]")

    def test_consecutive_repeats_collapse_inside_window(self) -> None:
        store = EventStore()
        start = datetime(2026, 8, 18, tzinfo=timezone.utc)
        first = store.publish("Unknown   command: TUBE_SET_YAW", source="klipper", timestamp=start)
        second = store.publish("Unknown command: TUBE_SET_YAW", source="klipper", timestamp=start + timedelta(seconds=20))
        self.assertEqual(first["id"], second["id"])
        self.assertEqual(second["repeat_count"], 2)
        self.assertEqual(len(store.history()), 1)

        store.publish("An intervening message", source="klipper", timestamp=start + timedelta(seconds=21))
        store.publish("Unknown command: TUBE_SET_YAW", source="klipper", timestamp=start + timedelta(seconds=22))
        self.assertEqual(len(store.history()), 3)

    def test_retention_and_newest_first(self) -> None:
        store = EventStore(max_events=500)
        for index in range(510):
            store.publish(f"message {index}")
        history = store.history(500)
        self.assertEqual(len(history), 500)
        self.assertEqual(history[0]["message"], "message 509")
        self.assertEqual(history[-1]["message"], "message 10")


if __name__ == "__main__":
    unittest.main()
