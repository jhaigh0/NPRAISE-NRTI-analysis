import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from endf_database_util.endf_database import ENDFDatabase, TotalCrossSectionData


class ENDFDatabaseTests(unittest.TestCase):
    def test_get_total_cross_section_uses_cache(self):
        sample_text = ""
        with open(Path(__file__).parent / "6-Li.txt") as f:
            sample_text = f.read_text(encoding="utf-8")

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            db = ENDFDatabase()
            db.cache_dir = cache_dir
            cache_dir.mkdir(parents=True, exist_ok=True)

            # Response must be a context manager with a read() method
            fake_response = type(
                "Response",
                (),
                {
                    "read": lambda self: sample_text.encode("utf-8"),
                    "__enter__": lambda self: self,
                    "__exit__": lambda self, exc, val, tb: None,
                },
            )()

            with patch(
                "endf_database_util.endf_database.urlopen", return_value=fake_response
            ) as mocked_urlopen:
                first = db.get_total_cross_section("Li6")
                second = db.get_total_cross_section("Li6")

            self.assertIsInstance(first, TotalCrossSectionData)
            # isotope is a periodictable.Isotope
            self.assertEqual(
                f"{first.isotope.element.symbol}-{first.isotope.isotope}", "Li-6"
            )
            self.assertEqual(first.x, [1.00000e-05, 1.03125e-05, 1.06250e-05])
            self.assertEqual(first.y, [47203.4, 46482.7, 45794.0])
            self.assertEqual(second.x, first.x)
            self.assertEqual(second.y, first.y)
            self.assertEqual(mocked_urlopen.call_count, 1)

            cache_file = cache_dir / "6-Li.txt"
            self.assertTrue(cache_file.exists())


if __name__ == "__main__":
    unittest.main()
