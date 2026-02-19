# Data Collection

SUNT uses an **Automated Data Collection System (ADCS)** integrating four sources:

| Source | Coverage | Key information |
|---|---|---|
| AVL | Regular buses + BRT | Real-time GPS positions |
| AFC | Buses + BRT + Subway | Fare validations (card taps) |
| GTFS | All systems | Static schedules and stop sequences |
| LTI | All systems | Actual trip start/end times |

## Collection Period

March 1, 2024 – March 31, 2025 (13 months)  
Temporal granularity: **< 1 minute**

## Privacy

Passenger card IDs in AFC are anonymized using a **hash-based, collision-free**
function. No attribute in the dataset links individual users to personal information.
Elderly passengers (≥65 y/o) travel free without cards and are not individually
tracked; their presence is modeled via probability distributions.
