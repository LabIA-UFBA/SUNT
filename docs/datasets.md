# Dataset Tables

## Automatic Vehicle Location (AVL)
AVL (Automatic Vehicle Location) technology records real-time vehicles’ geographical location. The Global Positioning System (GPS) is commonly used for this purpose. In the SUNT dataset, AVL records can be divided into two datasets: AVL-lines and AVL-vehicles.

### AVL-lines
Features information concerning vehicles’ routes and bus schedules.

| route_short_name | pt_sequence | direction_id | longitude   | latitude   | stop_id  | route_long_name                        | service_code |
|------------------|-------------|--------------|-------------|------------|----------|-----------------------------------------|--------------|
| 0116             | 1           | 1            | -38.51123   | -12.983389 | 43768720 | Avenida Vale Do Tororo                  | 53786        |
| 0116             | 2           | 1            | -38.511097  | -12.986428 | 45832898 | Avenida Vale do Tororo, 291            | 53786        |
| 0116             | 3           | 1            | -38.511448  | -12.990091 | 44782328 | Praça Dr. João Mangabeira              | 53786        |
| 0116             | 4           | 1            | -38.504387  | -12.990533 | 44784448 | Av. Vaco da Gama, S/N - 5378           | 53786        |

- `route_short_name`: (str) Identifies the bus line.  
- `pt_sequence`: (int) Presents the stop sequences of the bus line.  
- `direction_id`: (int) Gives the direction of the line, where 1 stands for one-way and 0 for return trip.  
- `longitude`/`latitude`: (float) Geographical coordinates of the bus stop identified in `stop_id`.  
- `stop_id`: (str) Identifies the bus stop.  
- `route_long_name`: (str) Stores the bus stop's name.  
- `service_code`: (str) Identifies the trip along the line.  

### AVL-vehicles
Features information concerning vehicles’ routes and bus schedules over time.

| vehicle | route_short_name | direction_id | gps_datetime         | longitude   | latitude    | stop_id  | service_code |
|---------|------------------|--------------|----------------------|-------------|-------------|----------|--------------|
| 20001   | 0310             | 0            | 2024-03-01 05:53:20  | -38.512428  | -12.978642  | 45834426 | 45546        |
| 20001   | 0310             | 0            | 2024-03-01 05:53:53  | -38.509964  | -12.975935  | 45834425 | 45546        |
| 20001   | 0310             | 0            | 2024-03-01 05:53:57  | -38.509964  | -12.975935  | 45834425 | 45546        |
| 20001   | 0310             | 0            | 2024-03-01 05:54:02  | -38.508957  | -12.975689  | 44782954 | 45546        |

- `vehicle`: (str) Code identifying the vehicle.  
- `route_short_name`: (str) Identifies the bus line.  
- `direction_id`: (int) Indicates the travel direction (1 = one-way, 0 = return).  
- `gps_datetime`: (datetime) Timestamp(s) of arrival and/or departure from a stop.  
- `longitude`/`latitude`: (float) Geographical coordinates where the GPS was recorded.  
- `stop_id`: (str) ID of the stop where the record occurred.  
- `service_code`: (str) Identifies the trip along the line.

## Automatic Fare Collection (AFC)
Automated Fare Collection (AFC) is a payment system using smart cards or smartphones. In the SUNT dataset, AFC data includes information on fare validations at turnstiles, but does not include exact boarding or alighting locations for buses.

### AFC
Features information regarding ticket validations by passengers across the public transportation network.

| cod_card       | afc_datetime         | integration | route_short_name | direction_id | value | vehicle |
|----------------|----------------------|-------------|------------------|---------------|--------|---------|
| 02310034266847 | 2024-03-01 06:22:03  | False       | 1386             | I             | 0.0    | 20390   |
| 02310034266847 | 2024-03-01 06:22:10  | False       | 1386             | I             | 0.0    | 20390   |
| 02310033002113 | 2024-03-01 06:22:57  | False       | 1386             | I             | 0.0    | 20390   |
| 02310032345960 | 2024-03-01 08:12:25  | False       | 1386             | I             | 0.0    | 20390   |

- `cod_card`: (str) Anonymized passenger card ID.  
- `afc_datetime`: (datetime) Time of fare validation at the turnstile.  
- `integration`: (bool) Indicates if the fare is part of an integrated connection between transport modes.  
- `route_short_name`: (str) Identifies the bus line.  
- `direction_id`: (str) Direction of the line: 'I' (one-way) or 'V' (return).  
- `value`: (float) Fare paid for the trip.  
- `vehicle`: (str) Code identifying the vehicle where the fare was validated.

## General Transit Feed Specification (GTFS)
GTFS (General Transit Feed Specification) is an open standard for distributing relevant information about transit systems to users. In the SUNT dataset, GTFS provides five tablesthat describe the entire public transportation network and services related to bus companies.

### GTFS Agency
Information about the bus companies.

| agency_id | agency_name | agency_url | agency_timezone    | agency_lang | agency_phone |
|-----------|-------------|------------|---------------------|-------------|--------------|
| 1         | company_I   | www.       | America/Sao_Paulo   | pt          |              |
| 2         | company_II  | www.       | America/Sao_Paulo   | pt          |              |

- `agency_id`: (int) Identifies the bus agency.
- `agency_name`: (str) Name of the bus agency.
- `agency_url`: (str) URL of the agency's website.
- `agency_timezone`: (str) Time zone of the agency.
- `agency_lang`: (str) Language used by the agency.
- `agency_phone`: (str) Contact phone number of the agency.

### GTFS Routes
Information about bus lines.

| route_id | agency_id | route_short_name | route_long_name                                      | route_type |
|----------|-----------|------------------|------------------------------------------------------|------------|
| 4089     | 1         | 1230             | Sussuarana x Barra R1.                               | 3          |
| 4450     | 1         | 1321             | São Marcos x Barroquinha                             | 3          |
| 4518     | 1         | 1103             | Alto do Cruzeiro/Pernambués x Shop.Bela Vista/Term Ac.Norte | 3   |
| 4523     | 1         | 1405             | Estação Pirajá x Cajazeiras 8                        | 3          |
| 4524     | 1         | 1137             | Pernambués x Barra                                   | 3          |

- `route_id`: (int) Identifies the route.
- `agency_id`: (int) Foreign key that links to the GTFS Agency.
- `route_short_name`: (str) Short name or code for the route.
- `route_long_name`: (str) Full name or description of the route.
- `route_type`: (int) Type of transportation (e.g., 3 = bus).

### GTFS Trips
Information about trips and their corresponding paths.

| route_id | service_id         | trip_id          | direction_id | block_id   | shape_id  |
|----------|--------------------|------------------|--------------|------------|-----------|
| 4089     | 26082_D_1046761    | 1046761_D_1_0    | 0            | 4089_001M  | 26082_I   |
| 4089     | 26082_D_1046761    | 1046761_D_1_1    | 1            | 4089_001M  | 26082_V   |
| 4089     | 26082_D_1046761    | 1046761_D_2_0    | 0            | 4089_002M  | 26082_I   |
| 4089     | 26082_D_1046761    | 1046761_D_2_1    | 1            | 4089_002M  | 26082_V   |
| 4089     | 26082_D_1046761    | 1046761_D_3_0    | 0            | 4089_002T  | 26082_I   |

- `route_id`: (int) Identifier of the route.
- `service_id`: (str) Identifier for a group of trips on specific days.
- `trip_id`: (str) Identifier for the trip.
- `direction_id`: (int) Indicates direction: 0 = one way, 1 = return.
- `block_id`: (str) Used for blocking trips together.
- `shape_id`: (str) Identifier for the shape of the route.

### GTFS Stops Times
Chronological order of bus stops where each trip paused.

| trip_id         | arrival_time | departure_time | stop_id   | stop_sequence | pickup_type | drop_off_type |
|-----------------|--------------|----------------|-----------|----------------|--------------|----------------|
| 1046761_D_1_0   | 08:30:00     | 08:30:00       | 43968810  | 1              | 0            | 0              |
| 1046761_D_1_0   | 08:31:41     | 08:31:41       | 47566106  | 2              | 0            | 0              |
| 1046761_D_1_0   | 08:33:49     | 08:33:49       | 44782337  | 3              | 0            | 0              |
| 1046761_D_1_0   | 08:34:55     | 08:34:55       | 44784470  | 4              | 0            | 0              |
| 1046761_D_1_0   | 08:35:44     | 08:35:44       | 44784471  | 5              | 0            | 0              |

- `trip_id`: (str) Identifier for the trip.
- `arrival_time`: (str) Time the vehicle arrives at the stop.
- `departure_time`: (str) Time the vehicle departs from the stop.
- `stop_id`: (str) Identifier of the stop.
- `stop_sequence`: (int) Order of the stop in the trip.
- `pickup_type`: (int) Indicates pickup behavior.
- `drop_off_type`: (int) Indicates drop-off behavior.

### GTFS Stops
Information about each bus stop.

| stop_id     | stop_name                         | latitude        | longitude        | location_type | parent_station |
|-------------|-----------------------------------|------------------|------------------|----------------|----------------|
| 43968810_S  | R. São Cristóvão 2                | -12.931565284729| -38.444393157959 | 1              |                |
| 43968810    | R. São Cristóvão 2                | -12.931565284729| -38.444393157959 | 0              | 43968810_S     |
| 47566106_S  | Av. Ulysses Guimarães 4067        | -12.93385887146 | -38.4467735290527| 1              |                |
| 47566106    | Av. Ulysses Guimarães 4067        | -12.93385887146 | -38.4467735290527| 0              | 47566106_S     |
| 44782337    | Av. Ulysses Guimarães 4314-4322   | -12.9351501464844| -38.4405784606934| 0              |                |

- `stop_id`: (str) Unique identifier for the bus stop.
- `stop_name`: (str) Name or description of the stop.
- `latitude`/`longitude`: (float) Coordinates of the stop.
- `location_type`: (int) 0 for stop, 1 for station.
- `parent_station`: (str) ID of the parent station, if applicable.

## Local Trip Information (LTI)
Maps the start and end of each trip made by a vehicle on a specific route. Complements the AVL dataset, which lacks trip start/end information. The `activity` field indicates whether a vehicle is starting a trip normally, leaving the garage, or returning to it.

| route_short_name | service_code | direction_id | vehicle | start_trip          | end_trip            | activity               |
|------------------|--------------|--------------|---------|---------------------|---------------------|------------------------|
| T014             | 74335        | I            | 20401   | 01/03/2024 17:03:49 | 01/03/2024 17:10:45 | Leaving the garage     |
| T014             | 74335        | I            | 20516   | 01/03/2024 05:37:16 | 01/03/2024 05:40:36 | Leaving the garage     |
| T014             | 74335        | I            | 20516   | 01/03/2024 17:11:40 | 01/03/2024 17:20:58 | Normal                 |
| T014             | 74335        | I            | 20086   | 01/03/2024 05:39:27 | 01/03/2024 05:46:38 | Leaving the garage     |
| T014             | 74335        | I            | 20401   | 01/03/2024 12:37:47 | 01/03/2024 12:42:04 | Returning to the garage|

- `route_short_name`: (str) Short name/code for the route.  
- `service_code`: (str) Identifier for the specific service the trip belongs to.  
- `direction_id`: (str) Indicates travel direction: 'I' (one-way) or 'V' (return).  
- `vehicle`: (int) Identifier of the vehicle.  
- `start_trip`: (datetime) Timestamp of trip start.  
- `end_trip`: (datetime) Timestamp of trip end.  
- `activity`: (str) Classification of trip type: “Normal”, “Leaving the garage”, or “Returning to the garage”.

## Boarding
A comprehensive table integrating data from AFC, LTI, AVL, and GTFS sources. It includes trip metadata, fare details, vehicle IDs, timestamps, and derived metrics for analysis. The `type_bus`, `classification`, and `motive` columns support detailed investigation into operational patterns.

| tripuserid           | type_bus | user_type | set        | registers | trip_id       | start_trip          | end_trip            | tolerance | integration | cod_card       | stop_time           | register_time       | service_code | route_short_name | vehicle_afc | vehicle | stop_id  | order | direction_id | trip_em | dif_boarding | trip   | classification | motive          | trip       | set_nb     | stop_time_nb        | route_short_name_nb | vehicle_nb | stop_id_nb | diff_nb | motive_pe | target_boarding |
|----------------------|----------|-----------|------------|-----------|---------------|---------------------|---------------------|-----------|-------------|----------------|----------------------|----------------------|--------------|------------------|-------------|---------|----------|-------|---------------|---------|--------------|--------|----------------|------------------|------------|------------|----------------------|----------------------|------------|------------|---------|-----------|------------------|
| 02300033357538_...   | bus      | driver    | company_i  | 2         | 20097_0310_7  | 2024-03-01 17:56:43 | 2024-03-01 20:08:27 | NaT       | False       | 2300033357538  | 2024-03-01 19:36:35  | 2024-03-01 18:48:30  | 45546        | 0310             | 20097       | 20097   | 44782849 | 1.0   | I             | 7.0     | 48.083       | Inside | irregular      | excessive time   | firt_trip  | company_i  | 2024-03-01 20:04:39  | 1067                 | 20446      | 44164980   | 0.53    | regular   | irregular         |

- `tripuserid`: (str) Unique ID composed of the card number and registration time.  
- `type_bus`: (str) Type of the bus (e.g., regular, BRT, subway).  
- `user_type`: (str) Role of the user (e.g., driver).  
- `set`: (str) Company or data subset ID.  
- `registers`: (int) Number of validation events registered in the trip.  
- `trip_id`: (str) Unique trip ID including vehicle, route, and sequence.  
- `start_trip`: (datetime) Time when the trip started.  
- `end_trip`: (datetime) Time when the trip ended.  
- `tolerance`: (datetime or NaT) Tolerance timestamp for delay analysis.  
- `integration`: (bool) Whether the trip was part of an integration with other services.  
- `cod_card`: (str) Anonymized card code used by the passenger.  
- `stop_time`: (datetime) Timestamp when the user validated the fare.  
- `register_time`: (datetime) Time when the record was registered.  
- `service_code`: (str) Code identifying the service.  
- `route_short_name`: (str) Short name of the route.  
- `vehicle_afc`: (int) Vehicle ID from AFC dataset.  
- `vehicle`: (int) Main vehicle ID associated with the trip.  
- `stop_id`: (str) ID of the stop where boarding occurred.  
- `order`: (float) Order of stop along the route.  
- `direction_id`: (str) Direction of the route ('I' or 'V').  
- `trip_em`: (float) Trip sequence identifier.  
- `dif_boarding`: (float) Difference in minutes between register time and stop time.  
- `trip`: (str) Position of trip (e.g., first_trip).  
- `classification`: (str) Type of boarding (e.g., regular, irregular).  
- `motive`: (str) Reason for classification (e.g., excessive time).  
- `set_nb`: (str) Company or subset ID for neighboring trip.  
- `stop_time_nb`: (datetime) Neighboring stop timestamp.  
- `route_short_name_nb`: (str) Neighboring route short name.  
- `vehicle_nb`: (int) Vehicle ID for neighboring trip.  
- `stop_id_nb`: (str) Stop ID for neighboring boarding.  
- `diff_nb`: (float) Time difference (minutes) to neighboring boarding.  
- `motive_pe`: (str) Evaluation of neighboring motive (e.g., regular).  
- `target_boarding`: (str) Target classification for boarding analysis.
  
## Alighting

Features estimated alighting information inferred from passenger behavior, including walking metrics and transfer patterns.

| tripuserid             | stop_time_ali       | stop_id_ali | order_ali | walk_target | trip_ali | walk_dis | walk_time | walk_speed | diff_de_pe | wait_time | trip_dis | trip_time | vel_media | bridge | bridge_type | bridge_id | chain    | target_ws | target_avs | target_tt | target_td | target_alighting |
|------------------------|---------------------|-------------|-----------|-------------|----------|----------|-----------|------------|------------|-----------|----------|-----------|-----------|--------|-------------|-----------|----------|-----------|------------|-----------|-----------|------------------|
| 02300033520791_20240301104958 | 2024-03-01 10:55:27 | 44165441    | 6.0       | excessive   | 8.0      | 1.299    | 15.588    | 5.5        | 68.4       | 52.812    | 1.884    | 5         | 22        | False  | no bridge   | None      | bus-bus  | regular   | regular    | regular   | regular   | regular          |

- `tripuserid`: (str) Composite identifier combining the anonymized card ID and timestamp.
- `stop_time_ali`: (datetime) Estimated alighting time.
- `stop_id_ali`: (str) ID of the stop where alighting is inferred.
- `order_ali`: (float) Sequence order of the alighting event.
- `walk_target`: (str) Classification of walking behavior post-alighting.
- `trip_ali`: (float) Number of trips associated with the alighting.
- `walk_dis`: (float) Estimated walking distance in kilometers.
- `walk_time`: (float) Estimated walking time in minutes.
- `walk_speed`: (float) Estimated walking speed in km/h.
- `diff_de_pe`: (float) Difference between estimated and actual walking distance.
- `wait_time`: (float) Estimated waiting time at the stop in minutes.
- `trip_dis`: (float) Distance covered during the trip in kilometers.
- `trip_time`: (float) Duration of the trip in minutes.
- `vel_media`: (float) Average speed during the trip in km/h.
- `bridge`: (bool) Indicates if a bridge was used during the trip.
- `bridge_type`: (str) Type of bridge used, if any.
- `bridge_id`: (str) Identifier of the bridge, if applicable.
- `chain`: (str) Sequence of transport modes used.
- `target_ws`: (str) Target walking speed classification.
- `target_avs`: (str) Target average speed classification.
- `target_tt`: (str) Target trip time classification.
- `target_td`: (str) Target trip distance classification.
- `target_alighting`: (str) Target alighting behavior classification.


## Origin-Destination (OD)

Features estimated full trips by passengers, enabling analysis of bus loading at stops and along routes.

| route_short_name | register_code | direction_id | pt_sequence | stop_id   | vehicle | trip_number | trip_id           | start_trip          | end_trip            | stop_time           | n-boardings | n-alighting | lag_loading | balance | loading |
|------------------|---------------|--------------|-------------|-----------|---------|-------------|-------------------|---------------------|---------------------|---------------------|-------------|-------------|-------------|---------|---------|
| 1521             | 55037         | I            | 1           | 46021891  | 30661   | 1           | 30661_1521_1266   | 2024-03-01 06:59:11 | 2024-03-01 07:15:22 | 2024-03-01 06:59:11 | 42.0        | 0.0         | 0           | 0       | 42      |

- `route_short_name`: (str) Bus line identifier.
- `register_code`: (int) Service registration code.
- `direction_id`: (str) Direction of the trip: 'I' (one-way) or 'V' (return).
- `pt_sequence`: (int) Sequence number of the public transport segment.
- `stop_id`: (int) Identifier of the stop.
- `vehicle`: (int) Vehicle code.
- `trip_number`: (int) Sequential number of the trip.
- `trip_id`: (str) Unique identifier for the trip.
- `start_trip`: (datetime) Start time of the trip.
- `end_trip`: (datetime) End time of the trip.
- `stop_time`: (datetime) Time at the stop.
- `n-boardings`: (float) Number of boardings at the stop.
- `n-alighting`: (float) Number of alightings at the stop.
- `lag_loading`: (int) Loading from the previous stop.
- `balance`: (int) Net change in passenger count.
- `loading`: (int) Total passengers on board after the stop.

## Graph-based Dataset

### Edge Features

Features information about connections between stops/stations, including distance, geospatial data, average speed, trip time, and passenger loading.

| src       | dst       | distance | src_lat | dst_lat | src_lon | dst_lon | average_speed | trip_time | loading |
|-----------|-----------|----------|---------|---------|---------|---------|---------------|-----------|---------|
| 100009577 | 345936831 | 0.254    | -12.902 | -12.902 | -38.420 | -38.417 | 25.6          | 4         | 78      |
| 100722777 | 100722778 | 0.362    | -12.899 | -12.897 | -38.408 | -38.408 | 11.3          | 8         | 20      |
| 100722777 | 44782645  | 1.062    | -12.899 | -12.899 | -38.408 | -38.413 | 40.2          | 5         | 45      |
| 100722777 | 45833440  | 0.417    | -12.899 | -12.897 | -38.408 | -38.409 | 50.5          | 10        | 90      |
| 100722777 | 66771046  | 0.934    | -12.899 | -12.897 | -38.408 | -38.413 | 26.2          | 6         | 30      |

- `src`: (int) Origin stop/station ID.
- `dst`: (int) Destination stop/station ID.
- `distance`: (float) Distance between the two stops/stations in kilometers.
- `src_lat`, `dst_lat`: (float) Latitude of origin and destination.
- `src_lon`, `dst_lon`: (float) Longitude of origin and destination.
- `average_speed`: (float) Average vehicle speed between src and dst in km/h.
- `trip_time`: (int) Time taken for the trip in minutes.
- `loading`: (int) Average passenger load during the segment.

### Graph-based Node Features
Describes static features of transportation nodes (e.g., stops and stations), averaged across all data.

| node      | loading | n-alighting | n-routes | n-boarding | n-trips | n-vehicles | average_speed |
|-----------|---------|-------------|----------|------------|---------|-------------|----------------|
| 100009577 | 2.77    | 0.0         | 1.08     | 0.23       | 1.1     | 1.1         | 6.31           |
| 100722777 | 28.54   | 4.43        | 1.54     | 4.49       | 1.56    | 1.56        | 22.86          |
| 100722778 | 36.72   | 1.39        | 1.83     | 0.1        | 2.04    | 2.04        | 16.06          |
| 101214305 | 12.53   | 3.97        | 1.0      | 1.66       | 1.0     | 1.0         | 19.95          |
| 101269104 | 125.57  | 3.55        | 4.57     | 9.48       | 5.28    | 5.28        | 38.25          |

- `node`: (int) Unique identifier for the stop/station node.  
- `loading`: (float) Average number of passengers passing through the node.  
- `n-alighting`: (float) Average number of passengers alighting at the node.  
- `n-routes`: (float) Average number of routes passing through the node.  
- `n-boarding`: (float) Average number of boardings at the node.  
- `n-trips`: (float) Average number of trips involving the node.  
- `n-vehicles`: (float) Average number of vehicles passing through the node.  
- `average_speed`: (float) Average speed of vehicles when arriving at the node (km/h).  

### Graph-based Temporal Node Features
Describes node activity over time intervals (e.g., every 5 minutes).

| interval            | node      | loading | n-boarding | n-alighting | n-vehicles | n-routes | n-trips | average_speed |
|---------------------|-----------|---------|------------|-------------|------------|----------|---------|----------------|
| 2024-03-01 05:00:00 | 100009577 | 0.0     | 0.0        | 0.0         | 0.0        | 0.0      | 0.0     | 0.0            |
| 2024-03-01 05:05:00 | 100009577 | 14.0    | 1.0        | 0.0         | 2.0        | 2.0      | 2.0     | 0.0            |
| 2024-03-01 05:10:00 | 100009577 | 0.0     | 0.0        | 0.0         | 0.0        | 0.0      | 0.0     | 0.0            |
| 2024-03-01 05:15:00 | 100009577 | 0.0     | 0.0        | 0.0         | 0.0        | 0.0      | 0.0     | 0.0            |0.0     | 0.0            |

- `interval`: (datetime) 5-minute time interval of observation.  
- `node`: (int) Identifier of the node (stop/station).  
- `loading`: (float) Number of passengers at the node in the interval.  
- `n-boarding`: (float) Number of boardings in the interval.  
- `n-alighting`: (float) Number of alightings in the interval.  
- `n-vehicles`: (float) Number of distinct vehicles at the node.  
- `n-routes`: (float) Number of distinct routes at the node.  
- `n-trips`: (float) Number of distinct trips at the node.  
- `average_speed`: (float) Average speed of vehicles (km/h).
