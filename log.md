for weights of heuristics: do estimated time from segmentA to segmentB. speed can be 90% of max speed.
can adjust it for traffic and/or user driving dynamics like Waze (map app for driving) does.

CRS: EPSG:2326

use average speed for weight of a node
use speed limit for heuristics 110

lat, lon

[road_network link](https://data.gov.hk/en-data/dataset/hk-td-tis_15-road-network-v2/resource/38eec12e-6b66-489a-9a1b-736598ab16fd)

gdf layers:
['VEHICLE_RESTRICTION', 'TRAFFIC_FEATURES', 'SPEED_LIMIT', 'RUN_IN_OUT', 'ROUNDABOUT', 'PROHIBITION', 'PERMIT', 'PEDESTRIAN_ZONE', 'NSR', 'BUS_ONLY_LANE', 'CENTERLINE', 'TURN', 'INTERSECTION', 'TUN_BRIDGE_TOLL', 'ONSTREETPARK', 'GISP_ON_STREET_PARKING', 'TUN_BRIDGE_TV_TOLL']

centerline output format:
STREET_ENAME                                                      -99
STREET_CNAME                                                      –９９
ELEVATION                                                           0
ST_CODE                                                       49959.0
EXIT_NUM                                                         None
ROUTE_NUM                                                         NaN
REMARKS                                                          None
ROUTE_ID                                                          275
TRAVEL_DIRECTION                                                    1
CRE_DATE                                    2009-05-08 20:00:00+00:00
LAST_UPD_DATE_V                             2009-05-08 20:00:00+00:00
ALIAS_ENAME                                                      None
ALIAS_CNAME                                                      None
SHAPE_Length                                                20.367938
geometry            MULTILINESTRING ((834773.2070000004 814198.248...
Name: 0, dtype: object

interection output format:
CRE_DATE                             2009-05-08 20:00:00+00:00
INT_ID                                                   10707
INT_TYPE                                                     0
INT_ENAME                                                  -99
INT_CNAME                                                  –９９
RD_ID_1                                                94584.0
RD_ID_2                                                94492.0
RD_ID_3                                                94493.0
RD_ID_4                                                    NaN
RD_ID_5                                                    NaN
RD_ID_6                                                    NaN
REMARKS                                                   None
LAST_UPD_DATE_V                      2011-02-24 20:00:00+00:00
RD_ID_7                                                    NaN
RD_ID_8                                                    NaN
RD_ID_9                                                    NaN
RD_ID_10                                                   NaN
geometry           POINT (813766.0150000006 826148.5840000026)
Name: 0, dtype: object

traffic_features output format:
FEATURE_TYPE                                                 4
FEATURE_ID                                                 838
CRE_DATE                             2009-05-08 20:00:00+00:00
RD_ID_1                                                  93405
RD_ID_2                                                    NaN
RD_ID_3                                                    NaN
RD_ID_4                                                    NaN
RD_ID_5                                                    NaN
RD_ID_6                                                    NaN
REMARKS                                                   None
TUN_BRIDGE_ID                                              NaN
LAST_UPD_DATE_V                      2009-05-08 20:00:00+00:00
RD_ID_7                                                    NaN
RD_ID_8                                                    NaN
RD_ID_9                                                    NaN
geometry           POINT (828655.3828000007 826412.3034000024)
Name: 0, dtype: object

speed_limit output format:
SPEED_LIMIT_ID                                                  1881
ROAD_ROUTE_ID                                                 103706
SPEED_LIMIT                                                  80 km/h
REMARKS                                                         None
CRE_DATE                                   2009-05-08 20:00:00+00:00
LAST_UPD_DATE_V                            2009-05-08 20:00:00+00:00
BOUND                                                              0
Shape_Length                                               77.990943
geometry           MULTILINESTRING ((839825.8591300007 819787.095...
Name: 0, dtype: object

roundabout output format:
CRE_DATE                             2009-05-08 20:00:00+00:00
R_ABOUT_ID                                                 556
R_ABOUT_ENAME                                             None
R_ABOUT_CNAME                                             None
R_ABOUT_TYPE                                                 1
GRADED                                                       N
SIGNALIZED                                                   N
NO_OF_ARM                                                    3
RD_ID_1                                                  94929
RD_ID_2                                                94930.0
RD_ID_3                                                94931.0
RD_ID_4                                                94932.0
RD_ID_5                                                94928.0
RD_ID_6                                                    NaN
RD_ID_7                                                    NaN
RD_ID_8                                                    NaN
RD_ID_9                                                    NaN
RD_ID_10                                                   NaN
REMARKS                                                   None
LAST_UPD_DATE_V                      2009-05-08 20:00:00+00:00
geometry           POINT (817714.3608000008 825423.5574000012)
Name: 0, dtype: object

bus_only_lane output format:
ROAD_ROUTE_ID                                                  261305
TIME_ZONE                                                   0000-2400
EFFECTIVE_DAY                                                YYYYYYYY
REMARKS                                                          None
BUS_ONLY_LANE_ID                                               140195
CRE_DATE                                    2018-04-19 00:00:00+00:00
LAST_UPD_DATE_V                             2018-04-19 08:46:04+00:00
BOUND                                                               0
Shape_Length                                                90.555311
geometry            MULTILINESTRING ((832213.5457700007 826954.833...
Name: 0, dtype: object