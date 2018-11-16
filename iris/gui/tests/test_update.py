import socket
import unittest
from contextlib import suppress

from ..update import update_available


def connection_available():
    """ Returns whether or not an internet connection is available """
    with suppress(OSError):
        try:
            socket.create_connection(("www.google.com", 80))
        except:
            return False
        else:
            return True
    return False


@unittest.skipUnless(connection_available(), "Internet connection is required.")
class TestUpdateAvailable(unittest.TestCase):
    def test_version(self):
        """ Test that update_available is workign as intended"""
        is_outdated, latest_version = update_available()
        self.assertIsInstance(is_outdated, bool)
        self.assertIsInstance(latest_version, str)


if __name__ == "__main__":
    unittest.main()
