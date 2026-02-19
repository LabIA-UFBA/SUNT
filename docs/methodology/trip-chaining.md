# Trip Chaining

## The Problem

Salvador's public transportation system does **not** have physical alighting
validation — passengers only tap their card when boarding. SUNT uses the
**Trip Chaining** method to infer where each passenger exits: the next boarding
event of the same card is used to estimate the alighting point of the previous trip.

---

## Pipeline

```
AVL + AFC ──► Boarding identification
                    │
                    ▼
              Validity check
              (time thresholds)
                    │
          ┌─────────┴──────────┐
       Valid               Invalid
          │               (discard / redistribute)
          ▼
    User classification
          │
    ┌─────┴──────┐
  Regular     Elderly / single-trip
    │          (redistribute by distribution)
    ▼
  Alighting inference (Trip Chaining)
          │
    Quality filters
    (speed, distance, time)
          │
          ▼
    Origin-Destination (OD)
```

---

## Step 1 — Boarding Identification

AVL and AFC records are merged by matching each validation (`afc_datetime`) to
the nearest vehicle GPS position at that moment. The closest stop is assigned
as the boarding location.

| Location type | Max allowed gap |
|---|---|
| Bus station | 20 minutes |
| Regular stop | 5 minutes |

---

## Step 2 — User Classification

| User type | Handling |
|---|---|
| Regular (card) | Full Trip Chaining |
| Elderly (≥65 y/o) | Allocated via probability distribution |
| Single-trip-per-day | Boarding only; excluded from alighting inference |

---

## Step 3 — Alighting Inference

### Scenario I — Single Line

```
Trip 1:  b ──────────────► f   (8:00 AM boarding at b)
Trip 2:  f ──────────────► b   (6:00 PM boarding at f → infers alighting at f in trip 1)
```

### Scenario II — Connection

Two boardings per trip:

- Morning: board at **b** (8:00 AM), board at **d** (8:20 AM) → first alighting at **d**
- Evening: board at **j** (6:00 PM), board at **d** (6:10 PM)
- **Inference:** `b → j` (morning) and `j → b` (evening)

### Scenario III — Walking Connection

Board at **b** (8:00 AM), then at **x** (8:50 AM, different line).  
Stop **x** is within walking distance Δ of stop **f** on the first line.

**Inference:** `b → f (walk Δ to x) → u`

---

## Step 4 — Quality Filters

| Filter | Threshold |
|---|---|
| Max walking distance | 1.1 km |
| Max average vehicle speed | 80 km/h |
| Max trip duration | 2 hours |

Trips outside these bounds are flagged in `target_*` columns and retained in
the dataset — researchers can apply their own thresholds.
