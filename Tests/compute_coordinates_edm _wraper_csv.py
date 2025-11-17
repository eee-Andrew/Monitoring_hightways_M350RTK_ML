#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Διαβάζει rows από Excel log (Only_Photo_isTrue_row.xlsx),
παίρνει OSD / GIMBAL δεδομένα και χρησιμοποιεί το dem_center_pixel_v9.py
για να βρει το σημείο τομής με το DEM (lat, lon, h) για ΚΑΘΕ row.
"""

import math
import numpy as np
import pandas as pd

# Βεβαιώσου ότι αυτό το αρχείο είναι στον ίδιο φάκελο με το dem_center_pixel_v9.py
from compute_coordinates_edm import (
    camera_forward_world,
    intersect_ray_with_dem,
    backproject_ray,
)

# ----------------- βοηθητικές συναρτήσεις -----------------
def decode_coord(raw):
    """
    Παίρνει κάτι τύπου 249153112047979 και το κάνει 24.9153112047979
    (δηλ. βάζει τελεία μετά τα 2 πρώτα ψηφία).
    Δουλεύει είτε είναι string είτε αριθμός.
    """
    if pd.isna(raw):
        return None
    s = str(raw).strip()
    # σβήνουμε τυχόν κενά / κόμματα
    s = s.replace(",", "")
    neg = s.startswith("-")
    if neg:
        s = s[1:]
    # βάζουμε τελεία μετά τα 2 πρώτα ψηφία
    if len(s) <= 2:
        val = float(s)
    else:
        val = float(s[:2] + "." + s[2:])
    return -val if neg else val


def feet_to_m(feet):
    if pd.isna(feet):
        return None
    return float(feet) * 0.3048


# ----------------- κύρια ρουτίνα -----------------
def process_excel(
    excel_path,
    dem_path,
    geoid_path,
    fx,
    fy,
    cx=2592,
    cy=1944,
    u=None,
    v=None,
    yaw_offset_deg=90.0,
    rcg="zforward_to_zdown",
    max_range_m=6000.0,
    step_m=50.0,
    max_iters=5,
    debug=False,
    output_csv="ground_points_from_excel.csv",
):
    # 1) διαβάζουμε το excel
    print(f"[info] Διαβάζω Excel: {excel_path}")
    df = pd.read_excel(excel_path)

    # αν υπάρχει στήλη CAMERA.isPhoto, κρατάμε μόνο τα True (just in case)
    if "CAMERA.isPhoto" in df.columns:
        before = len(df)
        df = df[df["CAMERA.isPhoto"] == True]
        print(f"[info] Φιλτράρισμα CAMERA.isPhoto=True: {before} → {len(df)} γραμμές")

    # 2) ετοιμάζουμε intrinsics (ίδια για όλα τα rows)
    if u is None:
        u = cx
    if v is None:
        v = cy

    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], float)
    d_cam = backproject_ray(u, v, K)  # προς το παρόν δεν χρησιμοποιείται στο R_wc αλλά το κρατάμε

    print(f"[info] Intrinsics fx={fx}, fy={fy}, cx={cx}, cy={cy}, u={u}, v={v}")
    print(f"[info] DEM: {dem_path}")
    print(f"[info] Geoid: {geoid_path}")

    results = []

    for idx, row in df.iterrows():
        try:
            # 3) παίρνουμε τα απαραίτητα πεδία από το Excel
            lat_raw = row["OSD.latitude"]
            lon_raw = row["OSD.longitude"]
            h_ft = row["OSD.height [ft]"]

            drone_pitch = float(row["OSD.pitch"])
            drone_roll = float(row["OSD.roll"])
            drone_yaw = float(row["OSD.yaw"]) + yaw_offset_deg  # +90 όπως ζήτησες

            gimbal_pitch = float(row["GIMBAL.pitch"])
            gimbal_roll = float(row["GIMBAL.roll"])
            gimbal_yaw = float(row["GIMBAL.yaw"])

            # 4) μετατροπές μονάδων / format
            lat0 = decode_coord(lat_raw)
            lon0 = decode_coord(lon_raw)
            alt_ortho = feet_to_m(h_ft)  # από ft → m (για να ταιριάζει με DEM που είναι σε m)

            if None in (lat0, lon0, alt_ortho):
                print(f"[warn] NaN τιμές στη γραμμή {idx}, την προσπερνάω.")
                continue

            # 5) κατεύθυνση κάμερας στον κόσμο
            d_world = camera_forward_world(
                drone_yaw_deg=drone_yaw,
                drone_pitch_deg=drone_pitch,
                drone_roll_deg=drone_roll,
                gimbal_yaw_deg=gimbal_yaw,
                gimbal_pitch_deg=gimbal_pitch,
                gimbal_roll_deg=gimbal_roll,
                rcg=rcg,
                yaw_is_heading=False,  # χρησιμοποιούμε ήδη OSD.yaw + 90
                debug=debug,
            )

            # 6) τομή ray με DEM
            lat_g, lon_g, h_g = intersect_ray_with_dem(
                lat0=lat0,
                lon0=lon0,
                h0_ortho=alt_ortho,
                d_world=d_world,
                dem_path=dem_path,
                geoid_path=geoid_path,
                max_iters=max_iters,
                max_range_m=max_range_m,
                step_m=step_m,
                debug=debug,
            )

            results.append(
                {
                    "row_index": idx,
                    "CUSTOM.date [local]": row.get("CUSTOM.date [local]", None),
                    "OSD.latitude_raw": lat_raw,
                    "OSD.longitude_raw": lon_raw,
                    "OSD.height_ft": h_ft,
                    "lat0_deg": lat0,
                    "lon0_deg": lon0,
                    "ground_lat_deg": lat_g,
                    "ground_lon_deg": lon_g,
                    "ground_h_m": h_g,
                }
            )

            if debug:
                print(
                    f"[ok] row {idx}: drone({lat0:.8f}, {lon0:.8f}) → "
                    f"ground({lat_g:.8f}, {lon_g:.8f}, h={h_g:.2f})"
                )

        except Exception as e:
            print(f"[err] πρόβλημα στη γραμμή {idx}: {e}")
            continue

    # 7) σώζουμε αποτελέσματα
    out_df = pd.DataFrame(results)
    out_df.to_csv(output_csv, index=False, float_format="%.8f")
    print(f"[info] Αποθήκευσα {len(results)} σημεία στο {output_csv}")


if __name__ == "__main__":
    # --- ΕΔΩ ΒΑΖΕΙΣ ΤΑ ΔΙΚΑ ΣΟΥ PATHS / ΠΑΡΑΜΕΤΡΟΥΣ ---
    EXCEL_PATH = r"C:\Users\User\OneDrive - Democritus University of Thrace\Έγγραφα\GitHub\Monitoring_hightways_M350RTK_ML\Tests\Highway_Footage_Greece\Only_Photo_isTrue_row.xlsx"

    DEM_PATH = "https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N41_00_E024_00_DEM/Copernicus_DSM_COG_10_N41_00_E024_00_DEM.tif"
    GEOID_PATH = "https://download.osgeo.org/proj/vdatum/egm96_15/egm96_15.gtx"

    # οι ίδιες τιμές που έβαζες με το χέρι πριν
    FX = 1601280
    FY = 1801440
    CX = 2592
    CY = 1944
    U = 1296   # μπορείς να τα αλλάξεις αν θέλεις
    V = 800

    process_excel(
        excel_path=EXCEL_PATH,
        dem_path=DEM_PATH,
        geoid_path=GEOID_PATH,
        fx=FX,
        fy=FY,
        cx=CX,
        cy=CY,
        u=U,
        v=V,
        yaw_offset_deg=90.0,   # +90 στο OSD.yaw
        rcg="zforward_to_zdown",
        max_range_m=6000.0,
        step_m=50.0,
        max_iters=5,
        debug=True,            # βάλε False αν σε κουράζουν τα logs
        output_csv="ground_points_from_excel.csv",
    )
