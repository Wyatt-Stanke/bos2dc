"""Static configuration: endpoints, corridor geometry, and service filters."""

from __future__ import annotations

import re
from dataclasses import dataclass

API_BASE = "https://api.mobilitydatabase.org/v1"


@dataclass(frozen=True)
class Place:
    name: str
    lat: float
    lon: float


# Coordinates of the station buildings themselves; bus stops are attached
# to these by walking links (see graph.attach_place).
SOUTH_STATION = Place("Boston South Station", 42.35225, -71.05530)
UNION_STATION = Place("Washington Union Station", 38.89746, -77.00637)

# States whose feeds are queried from the catalog. The bbox query alone is
# not enough: feeds whose latest dataset has no bounding box are only found
# through their location metadata.
CORRIDOR_STATES = (
    "Massachusetts",
    "Rhode Island",
    "Connecticut",
    "New York",
    "New Jersey",
    "Pennsylvania",
    "Delaware",
    "Maryland",
    "District of Columbia",
    "Virginia",
)

# Rough I-95 corridor spine. A feed (or stop) is kept if it lies within
# CORRIDOR_BUFFER_KM of this polyline. The default buffer is wide enough to
# admit inland detours (Worcester, Hartford, Lehigh Valley, Frederick).
CORRIDOR_SPINE = (
    (42.3523, -71.0552),  # Boston
    (41.8240, -71.4128),  # Providence
    (41.3557, -72.0995),  # New London
    (41.3083, -72.9279),  # New Haven
    (41.1865, -73.1952),  # Bridgeport
    (41.0534, -73.5387),  # Stamford
    (40.7506, -73.9935),  # Manhattan
    (40.7357, -74.1724),  # Newark
    (40.2206, -74.7597),  # Trenton
    (39.9526, -75.1652),  # Philadelphia
    (39.7391, -75.5398),  # Wilmington
    (39.6837, -75.7497),  # Newark DE
    (39.5387, -76.0936),  # Havre de Grace / Perryville
    (39.2904, -76.6122),  # Baltimore
    (38.8973, -77.0063),  # Washington
)
CORRIDOR_BUFFER_KM = 60.0

# Catalog feed statuses worth downloading. "deprecated" feeds have been
# superseded by another catalog entry; "inactive" ones are no longer
# refreshed but are sometimes the only data for an agency (GATRA, Bee-Line,
# Harford Transit), so they are kept and flagged as stale.
USABLE_STATUSES = ("active", "future", "inactive")

# Feeds/agencies that are not fixed-fare local transit: intercity coaches
# with reservation or dynamic pricing, private commuter coaches, rail,
# ferries, tourist and campus shuttles that are not open to the public.
EXCLUDED_PROVIDER_PATTERNS = [
    r"water\s*taxi", r"airtrain", r"streetcar",
    # private commuter coaches / employer shuttles
    r"bloom\s*bus", r"coach\s*company", r"business\s*council", r"logan\s*express",
    r"greyhound", r"peter\s*pan", r"flix", r"megabus", r"trailways",
    r"coach\s*usa", r"academy", r"ourbus", r"trans-?bridge", r"bolt\s*bus",
    r"vamoose", r"tripper", r"lucky\s*star", r"jefferson\s*lines",
    r"martz", r"virginia\s*breeze", r"vermont\s*translines", r"boxcar",
    r"blue\s*apple", r"dattco", r"yankee\s*line", r"leprechaun",
    r"plymouth\s*&?\s*brockton", r"boston\s*express", r"miller\s*transportation",
    r"amtrak", r"via\s*rail", r"metro-?north", r"long\s*island\s*rail",
    r"ferry", r"ferries", r"cruise", r"waterway", r"seastreak", r"party\s*boats",
    r"statue\s*of\s*liberty", r"harbor\s*island", r"shore\s*line\s*east",
    r"virginia\s*railway\s*express", r"patco", r"\bpath\b", r"tramway",
    r"university", r"college", r"renfe", r"pays\s*de\s*la\s*loire",
]
EXCLUDED_FEED_NAME_PATTERNS = [
    r"\brail\b", r"subway", r"light\s*rail", r"marc", r"train", r"ferry",
    r"hartford\s*line",
]

# Agencies inside otherwise-included feeds that should be dropped.
EXCLUDED_AGENCY_PATTERNS = [r"amtrak", r"greyhound", r"peter\s*pan", r"megabus"]

# GTFS route_type values treated as "bus". 200-209 (coach) is deliberately
# excluded: in the corridor those are intercity services.
BUS_ROUTE_TYPES = frozenset({3, 11} | set(range(700, 717)))


def compile_patterns(patterns: list[str]) -> re.Pattern:
    return re.compile("|".join(f"(?:{p})" for p in patterns), re.IGNORECASE)


# Demand-response zones published outside GTFS. RIPTA's Flex On Demand zones
# (regular $2 fare, bookable by app or phone) are only available as the
# ArcGIS layer behind its route-map dashboard; the 204 Westerly zone is what
# bridges Rhode Island's fixed routes to SEAT in Pawcatuck, CT.
FLEX_ARCGIS_SOURCES = [
    {
        "agency": "Rhode Island Public Transit Authority",
        "agency_id": "RIPTA",
        "url": "https://services6.arcgis.com/H2Zm7udJq46gqzt2/arcgis/rest/services/FlexZones/FeatureServer/0",
        "id_field": "Route_ID",
        "name_field": "Name",
        "info_field": "MoreInfo",
        "type_field": "TripType",
        "keep_types": ["Flex Zone"],  # drops the naval-base and expired temporary polygons
        "hours_fields": {"weekday": "WkServ", "saturday": "SatServ", "sunday": "SunServ"},
    },
]


# Transitland Atlas feeds the Mobility Database does not carry. NECTD and
# WRTD (published with UConn's campus routes) are the only links between
# SEAT/CTtransit and north-eastern Connecticut; the New Jersey county systems
# are local alternatives to NJ Transit's long-distance routes.
ATLAS_EXTRA_FEEDS = [
    "f-northeastern~connecticut~transit~district",
    "f-university~of~connecticut",
    "f-atlantic~county~nj",
    "f-cumberland~county~nj",
    "f-gloucester~county~nj",
    "f-hunterdon~county~link",
    "f-warren~county~nj",
    "f-southjerseytransportationauthority",
    "f-hoboken~nj",
]
# Used only when git is unavailable to clone the atlas.
ATLAS_FALLBACK_FILES = [
    "cadavl.com.dmfr.json", "westchestergov.com.dmfr.json", "passio3.com.dmfr.json",
    "hosted-gtfs-feeds.s3.amazonaws.com.dmfr.json", "nj-transit.dmfr.json", "cttransit.com.dmfr.json",
    "ripta.com.dmfr.json", "trilliumtransit.com.dmfr.json", "massdot.state.ma.us.dmfr.json",
]
