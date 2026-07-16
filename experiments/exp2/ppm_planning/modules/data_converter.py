from __future__ import annotations

import math
from typing import List, Sequence

import numpy as np

TMP_RECORD_DTYPE = np.dtype(
    [
        ("count", np.int32),
        ("time", np.int32),
        ("loc_x", np.float32),
        ("loc_y", np.float32),
        ("prox_x", np.float32),
        ("prox_y", np.float32),
        ("prox_z", np.float32),
        ("prox_num", np.int32),
        ("prox_id", np.float32),
        ("prox_duration", np.float32),
        ("prox_velocity", np.float32),
        ("prox_theta", np.float32),
        ("prox_phi", np.float32),
        ("prox_distance", np.float32),
    ]
)


def _to_float(value: str | float | int | None) -> float:
    if value is None or value == "":
        return np.nan
    return float(value)


class ConvertBase:
    def _finalize_tmp_data(self, rows: Sequence[Sequence[float]]) -> np.ndarray:
        array = np.zeros(len(rows), dtype=TMP_RECORD_DTYPE)
        for idx, row in enumerate(rows):
            array[idx]["count"] = int(row[0])
            array[idx]["time"] = int(row[1])
            array[idx]["loc_x"] = _to_float(row[2])
            array[idx]["loc_y"] = _to_float(row[3])
            array[idx]["prox_x"] = _to_float(row[4])
            array[idx]["prox_y"] = _to_float(row[5])
            array[idx]["prox_z"] = _to_float(row[6])
            array[idx]["prox_num"] = int(_to_float(row[7]))
            array[idx]["prox_id"] = _to_float(row[8])
            array[idx]["prox_duration"] = _to_float(row[9])
            array[idx]["prox_velocity"] = _to_float(row[10])
            array[idx]["prox_theta"] = _to_float(row[11])
            array[idx]["prox_phi"] = _to_float(row[12])
            array[idx]["prox_distance"] = _to_float(row[13])
        return array


class OtoPP(ConvertBase):
    def __init__(self, omnia_loc: str, omnia_prox: str) -> None:
        self.omnia_loc = omnia_loc
        self.omnia_prox = omnia_prox

    def read_csv(self, file_path: str) -> list[list[str]]:
        with open(file_path, "r", encoding="utf-8") as file:
            return [
                line.strip().rstrip(",").split(",")
                for line in file
                if not line.startswith("count")
            ]

    def extract_loc(self) -> List[List[float]]:
        rows = self.read_csv(self.omnia_loc)
        loc_data: List[List[float]] = []
        for row in rows:
            if len(row) < 8:
                continue
            loc_data.append(
                [
                    int(row[0]),
                    int(row[1]),
                    float(row[2]),
                    float(row[3]),
                    float(row[4]),
                    float(row[5]),
                    float(row[6]),
                    float(row[7]),
                ]
            )
        return loc_data

    def extract_prox(self) -> List[List[float]]:
        rows = self.read_csv(self.omnia_prox)
        processed_prox_data: List[List[float]] = []
        for row in rows:
            if len(row) < 3:
                continue
            count = int(row[0])
            time = int(row[1])
            proximity_num = int(row[2])
            required_cols = 3 + proximity_num * 6
            if len(row) < required_cols:
                continue

            for i in range(proximity_num):
                offset = 3 + i * 6
                processed_prox_data.append(
                    [
                        count,
                        time,
                        proximity_num,
                        _to_float(row[offset]),
                        _to_float(row[offset + 1]),
                        _to_float(row[offset + 2]),
                        _to_float(row[offset + 3]),
                        _to_float(row[offset + 4]),
                        _to_float(row[offset + 5]),
                    ]
                )
        return processed_prox_data

    def rotation_matrix(self, yaw: float) -> np.ndarray:
        return np.array(
            [
                [np.cos(yaw), -np.sin(yaw), 0.0],
                [np.sin(yaw), np.cos(yaw), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

    def convert_coordinates(self, prox_data: list[float], loc_data: list[float]) -> list[float]:
        yaw = loc_data[7]
        rotation = self.rotation_matrix(yaw)

        theta, phi, radius = prox_data[6], prox_data[7], prox_data[8]
        tmp_x = radius * math.cos(phi) * math.cos(theta)
        tmp_y = radius * math.cos(phi) * math.sin(theta)
        tmp_z = radius * math.sin(phi)

        prox_rel = rotation @ np.array([tmp_x, tmp_y, tmp_z])
        prox_x = prox_rel[0] + loc_data[2]
        prox_y = prox_rel[1] + loc_data[3]
        prox_z = prox_rel[2]
        return [
            prox_data[0],
            prox_data[1],
            prox_data[2],
            prox_data[3],
            prox_data[4],
            prox_data[5],
            prox_x,
            prox_y,
            prox_z,
        ]

    def main_convert(self) -> np.ndarray:
        extracted_loc = self.extract_loc()
        extracted_prox = self.extract_prox()
        tmp_data = []

        for prox_row in extracted_prox:
            prox_count = prox_row[0]
            loc_row = next((row for row in extracted_loc if row[0] == prox_count), None)
            if loc_row is None:
                continue

            converted_prox = self.convert_coordinates(prox_row, loc_row)
            tmp_data.append(
                [
                    int(converted_prox[0]),
                    int(converted_prox[1]),
                    float(loc_row[2]),
                    float(loc_row[3]),
                    float(converted_prox[6]),
                    float(converted_prox[7]),
                    float(converted_prox[8]),
                    float(prox_row[2]),
                    float(prox_row[3]),
                    float(prox_row[4]),
                    float(prox_row[5]),
                    float(prox_row[6]),
                    float(prox_row[7]),
                    float(prox_row[8]),
                ]
            )

        return self._finalize_tmp_data(tmp_data)

