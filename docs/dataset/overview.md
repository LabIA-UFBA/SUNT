# Dataset Overview

SUNT is organized into two layers: **raw data** (as collected from the field)
and **processed data** (derived via the Trip Chaining pipeline).

## Raw Data

| Table | System | Records/month | Key content |
|---|---|---|---|
| AVL-lines | Bus, BRT | Static | Stop sequences and coordinates per line |
| AVL-vehicles | Bus, BRT | High-frequency | GPS positions per vehicle per second |
| AFC | Bus, BRT, Subway | ~35 million | Fare validations with card ID and timestamp |
| GTFS | All | Static | Standard schedules (5 tables) |
| LTI | All | ~700,000 | Actual trip start/end per vehicle |

## Processed Data

| Table | Rows/day (approx.) | Key content |
|---|---|---|
| Boarding | ~700,000 | Identified boarding stop per passenger per trip |
| Alighting | ~700,000 | Inferred alighting stop per passenger per trip |
| OD | Varies | Aggregated boarding/alighting/load per stop per trip |

## Graph

| Element | Count |
|---|---|
| Nodes (stops/stations) | 2,871 |
| Edges (route segments) | 4,526 |
| Temporal resolution | 5 minutes |

See each sub-section for detailed schemas and the
[Data Dictionary](../data-dictionary.md) for full column references.
