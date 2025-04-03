# DATASETS

This fold presents all datasets explored in our investigation.

We utilized an Automated Data Collection System (ADCS) to gather data from multiple sources, resulting in two distinct raw datasets. 
The first dataset was obtained from the Automatic Vehicle Location (AVL) system, which monitors all regular and BRT buses, providing details about their geospatial positions over time. 
The second dataset, the Automatic Fare Collection (AFC) system, contains information from the ticketing systems, recording the time when users' contactless cards are used for payments. 
In addition to the exact time of card usage, it also includes details on the vehicles and their respective lines.

Additionally, we used static data based on the General Transit Feed Specification (GTFS) format, which defines a standard format for public transportation schedules associated with geographic information. 
Using this format, we provided geospatial details about stations and stops along with their sequential order, lines, and directions. 
Finally, we also provide a dataset containing Local Trip Information (LTI), which includes details about the expected and actual departure and arrival times for all vehicles on every line and in each direction. 
This issue can be easily addressed by combining redundant vehicle information from GTFS and LTI.


