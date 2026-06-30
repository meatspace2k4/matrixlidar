#!/usr/bin/env python3
"""Calibration test: the classifier must reproduce the operator's labels on
the four labeled captures (the screenshots) and on live empty data.

These oriented frames are transcribed from the labeled screenshots; blanks
(no-target) are 4000. Run from the project root:

  python3 tests/test_classify.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bathroom.classify import classify          # noqa: E402
from bathroom.zones import State                 # noqa: E402

# Screenshot 1 surrogate: a real live empty-room frame (door closed).
EMPTY = [
    [1070, 1186, 1297, 1397, 1337, 1156, 1182, 1083],
    [1193, 1344, 1440, 1487, 1290, 1160, 1180, 1072],
    [1347, 1495, 1616, 1437, 1311, 1143, 1190, 1088],
    [1393, 1534, 1656, 1461, 1311, 1166, 1177, 1077],
    [1286, 4000, 1625, 1515, 1300, 1128, 1179, 1079],
    [1258, 1347, 4000, 1446, 1298, 1145, 1161, 1094],
    [1349, 991, 1083, 1305, 1300, 1143, 1154, 1070],
    [1339, 1016, 946, 1387, 1278, 1127, 1155, 1069],
]

# Screenshot 2: user STANDING at the toilet (door closed).
STANDING = [
    [1068, 1208, 720, 808, 842, 1157, 1273, 1169],
    [1193, 1300, 813, 840, 908, 1157, 1219, 1144],
    [1316, 1465, 832, 863, 996, 1167, 1268, 1147],
    [1375, 1358, 850, 850, 1093, 1156, 1177, 1165],
    [1265, 1296, 839, 913, 1291, 1180, 1175, 1164],
    [1240, 1289, 4000, 916, 1234, 1134, 1120, 1158],
    [1257, 1023, 841, 4000, 1069, 1138, 1145, 1161],
    [1340, 973, 845, 891, 926, 1112, 1188, 1138],
]

# Screenshot 3: user SITTING on the toilet (door closed).
SITTING = [
    [1096, 1193, 1298, 1409, 1300, 1178, 1280, 1156],
    [1195, 1301, 1036, 1532, 1326, 1145, 1240, 1155],
    [1343, 4000, 1276, 1431, 1313, 1174, 1217, 1165],
    [1264, 1078, 1125, 1525, 1265, 1180, 1195, 1158],
    [1155, 1042, 1055, 1433, 1294, 1138, 1156, 1178],
    [1236, 986, 951, 4000, 1282, 1146, 1154, 1161],
    [1334, 969, 853, 773, 884, 1125, 1167, 1153],
    [1338, 965, 905, 769, 968, 1129, 1176, 1139],
]

# Screenshot 4: user standing in front of the sensor, crossing the scene.
WALKING = [
    [426, 385, 358, 356, 365, 1175, 1262, 1118],
    [1205, 381, 371, 349, 362, 1144, 1225, 1147],
    [1328, 397, 377, 358, 370, 1116, 1207, 1160],
    [1357, 428, 395, 356, 363, 1132, 1171, 1159],
    [1318, 455, 433, 360, 373, 1143, 1200, 1158],
    [1241, 458, 442, 369, 381, 1168, 1153, 1159],
    [1341, 444, 1177, 386, 387, 1151, 1174, 1158],
    [1330, 456, 929, 400, 403, 1131, 1189, 1156],
]

CASES = [
    ("empty (live)", EMPTY, State.EMPTY),
    ("standing", STANDING, State.STANDING_AT_TOILET),
    ("sitting", SITTING, State.SITTING_AT_TOILET),
    ("walking", WALKING, State.WALKING),
]


def main():
    ok = True
    for name, frame, expected in CASES:
        got, f = classify(frame)
        flag = "PASS" if got == expected else "FAIL"
        if got != expected:
            ok = False
        print("%-4s %-14s expected=%-20s got=%-20s  occ=%d vnear=%d up=%d low=%d"
              % (flag, name, expected.name, got.name,
                 f["occ_count"], f["vnear_count"], f["upper_count"], f["lower_count"]))
    print("ALL PASS" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
