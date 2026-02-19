# Data Dictionary

Complete reference for all columns across every table in the SUNT dataset.

!!! info "Conventions"
    - **Type**: Python/Pandas dtype (`str`, `int`, `float`, `datetime`, `bool`).
    - **Nullable** ✓: field may contain `NaN`/`NaT` values.

---

## Raw Data

### AVL-lines

| Column | Type | Description | Nullable |
|---|---|---|:---:|
| `route_short_name` | `str` | Bus line identifier (e.g., `"0116"`) | |
| `pt_sequence` | `int` | Stop sequence order along the line | |
| `direction_id` | `int` | `1` = one-way, `0` = return trip | |
| `longitude` | `float` | Longitude of the bus stop | |
| `latitude` | `float` | Latitude of the bus stop | |
| `stop_id` | `str` | Unique stop identifier | |
| `route_long_name` | `str` | Full name/address of the bus stop | |
| `service_code` | `str` | Trip identifier along the line | |

---

### AVL-vehicles

| Column | Type | Description | Nullable |
|---|---|---|:---:|
| `vehicle` | `str` | Vehicle identifier code | |
| `route_short_name` | `str` | Bus line identifier | |
| `direction_id` | `int` | `1` = one-way, `0` = return | |
| `gps_datetime` | `datetime` | Arrival/departure timestamp at the stop | |
| `longitude` | `float` | Longitude where GPS was recorded | |
| `latitude` | `float` | Latitude where GPS was recorded | |
| `stop_id` | `str` | Stop where the record was captured | |
| `service_code` | `str` | Trip identifier along the line | |

!!! warning "Timestamp order"
    For each stop: `arrival(N)` < `departure(N)` < `arrival(N+1)`

---

### AFC

| Column | Type | Description | Nullable |
|---|---|---|:---:|
| `cod_card` | `str` | Anonymized passenger card ID (hash-based) | |
| `afc_datetime` | `datetime` | Timestamp of fare validation | |
| `integration` | `bool` | `True` if part of an integrated transfer | |
| `route_short_name` | `str` | Bus/BRT line identifier | |
| `direction_id` | `str` | `"I"` = one-way, `"V"` = return | |
| `value` | `float` | Fare paid (BRL). `0.0` for free/integrated trips | |
| `vehicle` | `str` | Vehicle where fare was validated | |

!!! note "Volume"
    ~35 million records per month.

---

### GTFS Agency

| Column | Type | Description | Nullable |
|---|---|---|:---:|
| `agency_id` | `int` | Unique agency identifier | |
| `agency_name` | `str` | Company name | |
| `agency_url` | `str` | Company website | ✓ |
| `agency_timezone` | `str` | Timezone (e.g., `"America/Sao_Paulo"`) | |
| `agency_lang` | `str` | Primary language (e.g., `"pt"`) | |
| `agency_phone` | `str` | Contact phone number | ✓ |

---

### GTFS Routes

| Column | Type | Description | Nullable |
|---|---|---|:---:|
| `route_id` | `int` | Unique route identifier | |
| `agency_id` | `int` | Foreign key → GTFS Agency | |
| `route_short_name` | `str` | Short code for the route | |
| `route_long_name` | `str` | Full name (origin × destination) | |
| `route_type` | `int` | `3` = Bus (GTFS standard) | |

---

### GTFS Trips

| Column | Type | Description | Nullable |
|---|---|---|:---:|
| `route_id` | `int` | Foreign key → GTFS Routes | |
| `service_id` | `str` | Groups trips on same operating days | |
| `trip_id` | `str` | Unique trip identifier | |
| `direction_id` | `int` | `0` = one-way, `1` = return | |
| `block_id` | `str` | Block-links consecutive trips | ✓ |
| `shape_id` | `str` | Geographic shape of the route | ✓ |

---

### GTFS Stop Times

| Column | Type | Description | Nullable |
|---|---|---|:---:|
| `trip_id` | `str` | Foreign key → GTFS Trips | |
| `arrival_time` | `str` | Scheduled arrival (`HH:MM:SS`) | |
| `departure_time` | `str` | Scheduled departure (`HH:MM:SS`) | |
| `stop_id` | `str` | Foreign key → GTFS Stops | |
| `stop_sequence` | `int` | Order of the stop within the trip | |
| `pickup_type` | `int` | Pickup behavior (`0` = regular) | |
| `drop_off_type` | `int` | Drop-off behavior (`0` = regular) | |

---

### GTFS Stops

| Column | Type | Description | Nullable |
|---|---|---|:---:|
| `stop_id` | `str` | Unique stop identifier. IDs ending `_S` = parent station | |
| `stop_name` | `str` | Human-readable stop name | |
| `latitude` | `float` | Latitude of the stop | |
| `longitude` | `float` | Longitude of the stop | |
| `location_type` | `int` | `0` = stop, `1` = station | |
| `parent_station` | `str` | Parent station `stop_id`, if applicable | ✓ |

---

### LTI — Local Trip Information

| Column | Type | Description | Nullable |
|---|---|---|:---:|
| `route_short_name` | `str` | Bus line identifier | |
| `service_code` | `str` | Trip service identifier | |
| `direction_id` | `str` | `"I"` = one-way, `"V"` = return | |
| `vehicle` | `int` | Vehicle identifier | |
| `start_trip` | `datetime` | Trip start timestamp | |
| `end_trip` | `datetime` | Trip end timestamp | |
| `activity` | `str` | `"Normal"`, `"Leaving the garage"`, or `"Returning to the garage"` | |

!!! note "Volume"
    ~700,000 records per month.

---

## Processed Data

### Boarding

| Column | Type | Description | Nullable |
|---|---|---|:---:|
| `tripuserid` | `str` | Composite ID: `{cod_card}_{register_time}` | |
| `type_bus` | `str` | `"bus"`, `"brt"`, or `"subway"` | |
| `user_type` | `str` | Passenger category (e.g., `"driver"`, `"regular"`) | |
| `set` | `str` | Operating company subset ID | |
| `registers` | `int` | Number of validations in this trip | |
| `trip_id` | `str` | `{vehicle}_{route}_{sequence}` | |
| `start_trip` | `datetime` | Trip start timestamp | |
| `end_trip` | `datetime` | Trip end timestamp | |
| `tolerance` | `datetime` | Tolerance window for delay analysis | ✓ |
| `integration` | `bool` | Whether this is an integrated transfer | |
| `cod_card` | `str` | Anonymized card ID | |
| `stop_time` | `datetime` | Vehicle arrival at the boarding stop | |
| `register_time` | `datetime` | Fare validation timestamp | |
| `service_code` | `str` | Trip service code | |
| `route_short_name` | `str` | Bus line identifier | |
| `vehicle_afc` | `int` | Vehicle ID from AFC records | |
| `vehicle` | `int` | Vehicle ID from AVL records | |
| `stop_id` | `str` | Stop where boarding occurred | |
| `order` | `float` | Stop sequence order | |
| `direction_id` | `str` | `"I"` or `"V"` | |
| `trip_em` | `float` | Sequential trip number for this vehicle | |
| `dif_boarding` | `float` | Minutes between `register_time` and `stop_time` | |
| `trip` | `str` | `"frt_trip"`, `"mid_trip"`, or `"last_trip"` | |
| `classification` | `str` | `"regular"` or `"irregular"` | |
| `motive` | `str` | Reason for irregular flag (e.g., `"excessive time"`) | ✓ |
| `set_nb` | `str` | Neighboring trip company ID | ✓ |
| `stop_time_nb` | `datetime` | Neighboring trip stop timestamp | ✓ |
| `route_short_name_nb` | `str` | Neighboring trip route | ✓ |
| `vehicle_nb` | `int` | Neighboring trip vehicle ID | ✓ |
| `stop_id_nb` | `str` | Neighboring trip boarding stop | ✓ |
| `diff_nb` | `float` | Minutes to the next boarding event | ✓ |
| `motive_pe` | `str` | Quality of neighboring boarding | ✓ |
| `target_boarding` | `str` | Final target label | |

**Validity thresholds:**

| Location | Max `dif_boarding` |
|---|---|
| Bus station | 20 minutes |
| Regular stop | 5 minutes |

---

### Alighting

!!! warning "Inferred data"
    All alighting locations are **estimated** via Trip Chaining.
    See [Methodology → Trip Chaining](../methodology/trip-chaining.md).

| Column | Type | Description | Nullable |
|---|---|---|:---:|
| `tripuserid` | `str` | Links to the corresponding Boarding row | |
| `stop_time_ali` | `datetime` | Estimated alighting timestamp | |
| `stop_id_ali` | `str` | Stop where alighting is inferred | |
| `order_ali` | `float` | Stop sequence at alighting point | |
| `walk_target` | `str` | Walking behavior: `"regular"` or `"excessive"` | |
| `trip_ali` | `float` | Number of trips associated with this alighting | |
| `walk_dis` | `float` | Estimated walking distance to next stop (km) | |
| `walk_time` | `float` | Estimated walking time (minutes) | |
| `walk_speed` | `float` | Estimated walking speed (km/h) | |
| `diff_de_pe` | `float` | Difference between estimated and observed walking | |
| `wait_time` | `float` | Estimated waiting time at the stop (minutes) | |
| `trip_dis` | `float` | Distance covered during the bus trip (km) | |
| `trip_time` | `float` | Bus trip duration (minutes) | |
| `vel_media` | `float` | Average vehicle speed during the trip (km/h) | |
| `bridge` | `bool` | Whether a walking bridge was used | |
| `bridge_type` | `str` | Type of bridge connection | |
| `bridge_id` | `str` | Bridge identifier | ✓ |
| `chain` | `str` | Transport mode sequence (e.g., `"bus-bus"`) | |
| `target_ws` | `str` | Walking speed classification | |
| `target_avs` | `str` | Average speed classification | |
| `target_tt` | `str` | Trip time classification | |
| `target_td` | `str` | Trip distance classification | |
| `target_alighting` | `str` | Final target label | |

**Quality filters:**

| Filter | Threshold |
|---|---|
| Max walking distance | 1.1 km |
| Max average vehicle speed | 80 km/h |
| Max trip duration | 2 hours |

---

### Origin-Destination (OD)

| Column | Type | Description | Nullable |
|---|---|---|:---:|
| `route_short_name` | `str` | Bus line identifier | |
| `register_code` | `int` | Service registration code | |
| `direction_id` | `str` | `"I"` or `"V"` | |
| `pt_sequence` | `int` | Stop sequence number in the trip | |
| `stop_id` | `int` | Stop identifier | |
| `vehicle` | `int` | Vehicle identifier | |
| `trip_number` | `int` | Sequential trip number | |
| `trip_id` | `str` | `{vehicle}_{route}_{sequence}` | |
| `start_trip` | `datetime` | Trip start timestamp | |
| `end_trip` | `datetime` | Trip end timestamp | |
| `stop_time` | `datetime` | Vehicle arrival at this stop | |
| `n_boardings` | `float` | Passengers boarding at this stop | |
| `n_alighting` | `float` | Passengers alighting at this stop | |
| `lag_loading` | `int` | Passenger load before vehicle arrives | |
| `balance` | `int` | Load after alightings: `lag_loading − n_alighting` | |
| `loading` | `int` | Load after boardings: `balance + n_boardings` | |

!!! info "First stop"
    At `pt_sequence = 1`: `lag_loading = balance = n_alighting = 0`.

---

## Graph Data

### Edge Features

| Column | Type | Description | Nullable |
|---|---|---|:---:|
| `src` | `int` | Origin node (stop/station ID) | |
| `dst` | `int` | Destination node (stop/station ID) | |
| `distance` | `float` | Distance between stops (km) | |
| `src_lat` | `float` | Latitude of origin stop | |
| `dst_lat` | `float` | Latitude of destination stop | |
| `src_lon` | `float` | Longitude of origin stop | |
| `dst_lon` | `float` | Longitude of destination stop | |
| `average_speed` | `float` | Mean vehicle speed on this segment (km/h) | |
| `trip_time` | `int` | Mean travel time on this segment (minutes) | |
| `loading` | `int` | Mean passenger load on this segment | |

---

### Node Features (Static)

| Column | Type | Description | Nullable |
|---|---|---|:---:|
| `node` | `int` | Stop/station identifier | |
| `loading` | `float` | Average passengers passing through | |
| `n_alighting` | `float` | Average alightings per vehicle visit | |
| `n_routes` | `float` | Average distinct routes serving this node | |
| `n_boarding` | `float` | Average boardings per vehicle visit | |
| `n_trips` | `float` | Average trips per day | |
| `n_vehicles` | `float` | Average distinct vehicles per day | |
| `average_speed` | `float` | Average vehicle speed on arrival (km/h) | |

---

### Node Features (Temporal)

Aggregated into **5-minute intervals**.

| Column | Type | Description | Nullable |
|---|---|---|:---:|
| `interval` | `datetime` | Start of the 5-minute observation window | |
| `node` | `int` | Stop/station identifier | |
| `loading` | `float` | Passengers at the node in this interval | |
| `n_boarding` | `float` | Boardings in this interval | |
| `n_alighting` | `float` | Alightings in this interval | |
| `n_vehicles` | `float` | Distinct vehicles in this interval | |
| `n_routes` | `float` | Distinct routes in this interval | |
| `n_trips` | `float` | Distinct trips in this interval | |
| `average_speed` | `float` | Average vehicle speed (km/h) | |
