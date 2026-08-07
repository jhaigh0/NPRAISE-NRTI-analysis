import html
import json
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

import periodictable


@dataclass
class TotalCrossSectionData:
    isotope: periodictable.core.Isotope
    x: list[float]
    y: list[float]


class ENDFDatabase:
    """Retrieve total neutron cross-section data from the KAERI ENDF/B-VIII.0 source."""

    def __init__(self):
        self.cache_dir = Path(__file__).parent / ".cache" / "endf"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._isotope_pattern = re.compile(r"^(?P<symbol>[A-Za-z]+)(?P<mass>\d+)$")
        self._known_isotopes = json.loads(
            (Path(__file__).parent / "isotopes.json").read_text()
        )

    def get_total_cross_section(self, isotope: str) -> TotalCrossSectionData:
        isotope_obj = self._parse_isotope(isotope)
        cache_file = self.cache_dir / f"{isotope_obj!s}.txt"
        if cache_file.exists():
            raw_text = cache_file.read_text(encoding="utf-8")
        else:
            url = self._build_url(isotope_obj)
            with urlopen(url) as response:
                raw_text = response.read().decode("utf-8")
            cache_file.write_text(raw_text, encoding="utf-8")

        return self._parse_total_cross_section(raw_text, isotope_obj)

    def _build_url(self, isotope: periodictable.core.Isotope) -> str:
        isotopes = self._known_isotopes[isotope.element.symbol]
        isotope_idx = isotopes.index(str(isotope.isotope).zfill(3))
        return (
            "https://atom.kaeri.re.kr/nuchart/getData.jsp"
            f"?target=endfb8.0,{isotope.element.number},{isotope.isotope},{isotope.element.number * 100 + 25 + (3 * isotope_idx)},3,1"
        )

    def _parse_isotope(self, isotope: str) -> periodictable.core.Isotope:
        if not isotope:
            raise ValueError("isotope must not be empty")

        normalized = isotope.strip()
        match = self._isotope_pattern.fullmatch(normalized)
        if match is None:
            raise ValueError(f"Unsupported isotope format: {isotope}")

        symbol, mass = match.groups()
        try:
            return getattr(periodictable, symbol)[int(mass)]
        except AttributeError:
            raise ValueError(f"Unknown element symbol: {symbol}")

    def _parse_total_cross_section(
        self, raw_text: str, isotope: periodictable.core.Isotope
    ) -> TotalCrossSectionData:
        html_span_pattern = re.compile(
            r"<span\s+class=['\"]text_s['\"]>(.*?)</span>", re.DOTALL | re.IGNORECASE
        )
        span_match = html_span_pattern.search(html.unescape(raw_text))
        if span_match:
            data = span_match.group(1).strip()
        else:
            print("Warning: No <span> tag found.")
            return None

        expected_header = f"{isotope.element.symbol}-{isotope.isotope}(n,tot) ENDFB-8.0"
        self._verify_data_header(data, expected_header)

        lines = [
            line.replace("<br>", "").strip()
            for line in data.splitlines()
            if line.strip()
        ]
        x_values: list[float] = []
        y_values: list[float] = []

        for line in lines[2:]:
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                x_values.append(float(parts[0]))
                y_values.append(float(parts[1]))
            except ValueError:
                continue

        return TotalCrossSectionData(isotope=isotope, x=x_values, y=y_values)

    @staticmethod
    def _verify_data_header(data: str, expected_header: str) -> None:
        first_line = data.splitlines()[0].strip() if data else ""
        if not first_line.startswith(expected_header):
            raise ValueError(
                f"ENDF data header mismatch: expected '{expected_header}', got '{first_line}'"
            )
