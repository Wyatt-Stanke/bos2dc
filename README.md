# bos2dc

Finds the most usable way to get from **Boston South Station** to
**Washington Union Station** using only local, fixed-fare buses: no Greyhound,
Peter Pan, FlixBus or other intercity coaches, and no subway, commuter rail or
Amtrak. "Most usable" is about how often the buses run, not about the
fastest timetable. A chain of 30+ buses across a dozen agencies is only as
good as its rarest link, and a leg that runs once a day makes the whole trip
a once-a-day proposition.

## Usage

```sh
pip install -e .
export REFRESH_TOKEN=...            # Mobility Database refresh token
bos2dc fetch                        # discover + download feeds (~330 MB into ./data)
bos2dc route                        # walk-bridged answer (no demand-response)
bos2dc route --flex                 # also allow RIPTA Flex on-demand zones
bos2dc route --itinerary 07:00      # plus a timed itinerary leaving at 07:00
```

`route` prints a report and writes `out/route.json` and `out/route.geojson`.
Useful options:

| option | meaning |
|---|---|
| `--day saturday` / `--date 2026-10-07` | representative service day (default: next Wednesday) |
| `--window 06:00-22:00` | hours in which frequency is measured |
| `--city-penalty 1 --express-penalty 3` | minutes added per mile ridden at city / express stop density |
| `--max-walk 3` | longest walk in km (default: 2.5 km, raised automatically if nothing connects) |
| `--no-auto-walk` | fail with a gap report instead of raising the walk limit |
| `--flex --flex-headway 60` | allow Flex zones; treat on-demand service like a bus every 60 min |
| `--exclude-agency REGEX`, `--exclude-feed ID` | drop services |
| `--max-stale-days N` | ignore feeds whose newest data expired more than N days ago |

## Data

* **Mobility Database** (`api.mobilitydatabase.org`). Every GTFS feed located
  in MA, RI, CT, NY, NJ, PA, DE, MD, DC or VA, or whose data overlaps the
  corridor. Intercity and private coaches, rail, ferries and campus shuttles
  are excluded by name (`config.py`), and only bus route types are kept. The
  exclusions cover Bloom, P&B, Coach Company and Logan Express.
* **Transitland Atlas** (`github.com/transitland/transitland-atlas`, shallow
  clone into `data/`). This is used two ways:
  * Stale catalog feeds are matched to atlas records, first by a shared URL,
    then by operator name with an area check. The atlas' current URL is used
    when its calendar is current. This refreshes GATRA, WRTA, Harford Transit,
    Charm City Circulator, Anne Arundel County and others.
  * Feeds missing from the catalog are added. This covers Northeastern CT
    Transit District, Windham Region Transit District (published in UConn's
    feed), and the Atlantic, Cumberland, Gloucester, Hunterdon and Warren
    County, SJTA and Hoboken systems.
* **RIPTA Flex zones**. These come from the ArcGIS layer behind RIPTA's route
  maps: zone polygons plus weekday/Saturday/Sunday hours. They are only used
  with `--flex`. GTFS-Flex zones (`locations.geojson`) inside catalog feeds are
  also read, and paratransit zones are skipped.

Feeds whose newest data is still expired are used anyway: the same weekday is
taken from their calendar, and every route that relies on them is flagged.
At the time of writing this applies to the Westchester Bee-Line, whose
current producer URL returns 503.

## Method

1. **Timetables.** Each feed is compiled for the representative date into
   trip patterns: unique stop sequences, with the times of every trip.
   `frequencies.txt` is expanded, and duplicate trips are dropped even when
   they appear across two feeds.
2. **Effective frequency.** For departures that split the window into gaps
   `g_i`, a rider who turns up at a random moment waits `Σg²/2W` on average.
   The *effective headway* is twice that wait. For evenly spaced service it
   equals the plain headway. Four peak-only trips come out at about 13 hours,
   and a daily bus at 16 hours (the whole window).
3. **Graph.**
   * Nodes are stops. There is a ride edge for every ordered stop pair on a
     pattern where you may board at the first stop and alight at the second.
   * Patterns serving the same pair are pooled by summing their frequencies
     (the *common-lines* effect), but only within one stop-density class.
   * Walk edges link stops within 400 m, the nearest few stops of each other
     agency within 2.5 km, and the single nearest one up to 8 km.
4. **Stop density.** Along each pattern, stops per mile are measured over a
   rolling one-mile window. **Local** means 4 or more stops per mile, **city**
   2–4, and **express** below 2. Every ride edge carries its local, city and
   express mileage, so a route can be local in town and express on the
   highway.
5. **Search.** The objectives are applied in order:
   1. **Shortest longest walk.** If nothing connects with 2.5 km walks, the
      smallest walk limit that does is found by binary search.
   2. **Widest path.** Maximise the effective frequency of the least frequent
      leg, using binary search over frequency levels with a reachability test.
   3. **Generalised cost** among routes with that bottleneck:
      * in-vehicle time;
      * ½ × effective headway per boarding (the expected wait);
      * 5 minutes per boarding;
      * walking time × 1.5 (× 3 beyond 1 km);
      * 1 min per city mile and 3 min per express mile.

      The express-over-city step is larger than the city-over-local step.

   A combined (bottleneck, cost) label can't be optimised in one Dijkstra
   pass, so the two phases are run separately. That is exact.
6. **Routes reported.**
   * The recommended route.
   * The **most local** route, with penalties of 5 and 15 min/mile, at the same
     bottleneck and at any frequency.
   * Trade-off routes that accept a weaker bottleneck for a lower cost.
7. **Timetable check.** Every reported route is replayed against the real
   trips for departures across a day. This gives the distinct end-to-end
   connections per day and the actual door-to-door time. `--itinerary` prints
   one of these replays in full.

## Known gaps in the data

* **Rhode Island → Connecticut.** No fixed-route, fixed-fare bus crosses the
  state line. The closest options are:
  * RIPTA 95X to Westerly, which runs once each weekday, then a ~4 km walk to
    SEAT 108 in Pawcatuck;
  * RIPTA 204 Flex on-demand, which reaches the state line.
  
  CTtransit's PPB Hartford/Providence is Peter Pan's route. The GTFS ends it at
  a non-boarding "RI State Line" point, and the Providence end is sold as a
  Peter Pan ticket.
* **Massachusetts → Connecticut inland.** PVTA and CTtransit meet at Enfield,
  but there are still walks of 4.4 km or more to get to Worcester, and 5.9 km
  from WRTA (Dudley) to NECTD (Thompson).
* **New Jersey.** No all-local chain exists between New York and Philadelphia:
  New Brunswick and Princeton aren't linked by public fixed routes. Every
  route therefore includes 70+ miles of express-density riding.
* **Stop density** uses straight-line distances between consecutive stops.
