#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Διαβάζει rows από Excel log (Only_Photo_isTrue_row.xlsx),
παίρνει OSD / GIMBAL δεδομένα και χρησιμοποιεί το dem_center_pixel_v9.py
για να βρει το σημείο τομής με το DEM (lat, lon, h) για ΚΑΘΕ row.

ΠΡΟΣΟΧΗ:
Ο στόχος είναι να ΜΙΜΗΘΕΙ ΑΚΡΙΒΩΣ τη συμπεριφορά του CLI dem_center_pixel_v9.py,
άρα καλούμε την intersect_ray_with_dem με την ΙΔΙΑ ΣΕΙΡΑ POSITIONAL arguments.
"""

import math
import numpy as np
import pandas as pd
import time

from dem_center_pixel_v10_work import (
    camera_forward_world,
    intersect_ray_with_dem,
    backproject_ray,
)

# ----------------- βοηθητικές συναρτήσεις -----------------
def decode_coord(raw):
    """
    Μετατρέπει:
      - 411458353764122.00000000 → 41.1458353764122
      - 249153112047979.00000000 → 24.9153112047979
    αλλά αν η τιμή είναι ήδη κανονικό δεκαδικό (π.χ. 41.1458),
    την αφήνει όπως είναι.
    """
    if pd.isna(raw):
        return None

    s = str(raw).strip()
    s = s.replace(",", "")
    neg = False
    if s.startswith("-"):
        neg = True
        s = s[1:]

    # Αν έχει τελεία, δες αν είναι ήδη "λογικό" δεκαδικό
    if "." in s:
        left, right = s.split(".", 1)

        # περίπτωση 41.1458353764122 (ή 24.9153 κτλ)
        if len(left) <= 3 and len(right) <= 10:
            val = float(left + "." + right)
            return -val if neg else val

        # περίπτωση 411458353764122.00000000 → κρατάμε μόνο το αριστερό μέρος
        s = left

    # Από εδώ και κάτω θεωρούμε "κολλημένο" νούμερο: βάζουμε τελεία μετά τα 2 πρώτα ψηφία
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
    yaw_offset_deg=90.0,       # CLI: δίνεις OSD.yaw + 90
    rcg="zforward_to_zdown",
    max_range_m=6000.0,
    step_m=50.0,
    max_iters=2,
    debug=False,
    output_csv="ground_points_from_excel.csv",
):
    print(f"[info] Διαβάζω Excel: {excel_path}")
    df = pd.read_excel(excel_path)

    # Φιλτράρισμα φωτο (όπως πριν)
    if "CAMERA.isPhoto" in df.columns:
        before = len(df)
        df = df[df["CAMERA.isPhoto"] == True]
        print(f"[info] Φιλτράρισμα CAMERA.isPhoto=True: {before} → {len(df)} γραμμές")

    if u is None:
        u = cx
    if v is None:
        v = cy

    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], float)
    d_cam = backproject_ray(u, v, K)  # δεν το χρησιμοποιούμε άμεσα, αλλά το κρατάω

    print(f"[info] Intrinsics fx={fx}, fy={fy}, cx={cx}, cy={cy}, u={u}, v={v}")
    print(f"[info] DEM: {dem_path}")
    print(f"[info] Geoid: {geoid_path}")

    results = []

    for idx, row in df.iterrows():
        try:
            # -------- 1) Δεδομένα από Excel --------
            lat_raw = row["OSD.latitude"]
            lon_raw = row["OSD.longitude"]
            h_ft = row["OSD.height [ft]"]

            # Drone attitude από OSD
            drone_pitch = float(row["OSD.pitch"])
            drone_roll  = float(row["OSD.roll"])
            drone_yaw   = float(row["OSD.yaw"]) + yaw_offset_deg  # όπως στο CLI: OSD.yaw + 90

            # Gimbal: στο log GIMBAL.pitch είναι αρνητικό όταν κοιτάει κάτω
            # Στο CLI βάζεις θετικό -> άρα εδώ κάνουμε sign flip
            gimbal_pitch = -float(row["GIMBAL.pitch"])
            gimbal_roll  = float(row["GIMBAL.roll"])
            gimbal_yaw   = float(row["GIMBAL.yaw [360]"])

            # -------- 2) Μετατροπές μονάδων --------
            lat0 = decode_coord(lat_raw)
            lon0 = decode_coord(lon_raw)
            alt_ortho = feet_to_m(h_ft)

            if None in (lat0, lon0, alt_ortho):
                print(f"[warn] NaN τιμές στη γραμμή {idx}, την προσπερνάω.")
                continue

            # -------- 3) Διεύθυνση κάμερας (ίδια λογική με CLI) --------
            d_world = camera_forward_world(
                drone_yaw_deg=drone_yaw,
                drone_pitch_deg=drone_pitch,
                drone_roll_deg=drone_roll,
                gimbal_yaw_deg=gimbal_yaw,
                gimbal_pitch_deg=gimbal_pitch,
                gimbal_roll_deg=gimbal_roll,
                rcg=rcg,
                yaw_is_heading=True,   # ΠΟΛΥ ΣΗΜΑΝΤΙΚΟ: όπως στο CLI (--yaw_is_heading)
                debug=debug,
            )

            # -------- DEBUG: εκτύπωση predicted + CLI equivalent --------
            print(f"\n[PRED] row {idx}")
            print(f"  drone_lat={lat0:.8f}, drone_lon={lon0:.8f}, alt_ortho_m={alt_ortho:.2f}")
            print(f"  drone_yaw={drone_yaw:.2f}, pitch={drone_pitch:.2f}, roll={drone_roll:.2f}")
            print(f"  gimbal_yaw={gimbal_yaw:.2f}, pitch={gimbal_pitch:.2f}, roll={gimbal_roll:.2f}")

            cli_cmd = (
                f'python dem_center_pixel_v9.py '
                f'--lat {lat0:.8f} --lon {lon0:.8f} --alt_ortho {alt_ortho:.2f} '
                f'--drone_yaw {drone_yaw:.2f} --drone_pitch {drone_pitch:.2f} --drone_roll {drone_roll:.2f} '
                f'--gimbal_yaw {gimbal_yaw:.2f} --gimbal_pitch {gimbal_pitch:.2f} --gimbal_roll {gimbal_roll:.2f} '
                f'--yaw_is_heading --rcg {rcg} --fx {fx} --fy {fy} --cx {cx} --cy {cy} --u {u} --v {v} '
                f'--dem "{dem_path}" --max_range_m {max_range_m} --step_m {step_m} --max_iters {max_iters} '
                f'--geoid "{geoid_path}"'
            )
            print("  CLI equivalent:")
            print(f"  {cli_cmd}")

            # -------- 4) Τομή με DEM (ΕΔΩ ΚΑΝΟΥΜΕ ΤΟ ΙΔΙΟ ΜΕ ΤΟ MAIN ΤΟΥ v9) --------
            # ΠΡΟΣΟΧΗ: ΧΩΡΙΣ KEYWORDS, ΜΟΝΟ POSITIONAL, ΙΔΙΑ ΣΕΙΡΑ:
            # lat, lon, h = intersect_ray_with_dem(
            #     args.lat, args.lon, args.alt_ortho, d_world, args.dem,
            #     args.geoid, args.max_iters,
            #     args.max_range_m, args.step_m, args.debug
            # )
            lat_g, lon_g, h_g = intersect_ray_with_dem(
                lat0,           # lat0
                lon0,           # lon0
                alt_ortho,      # h0_ortho
                d_world,        # d_world
                dem_path,       # dem_path
                geoid_path,     # geoid_path
                max_iters,      # max_iters
                max_range_m,    # tol_m  (ΝΑΙ, όπως στο CLI)
                step_m,         # max_range_m (ΝΑΙ, όπως στο CLI)
                debug,          # step_m → debug (ΝΑΙ, όπως στο CLI καλείται)
            )

            print(
                f"  → ground_lat={lat_g:.8f}, ground_lon={lon_g:.8f}, ground_h={h_g:.2f} m"
            )

            # -------- 5) Αποθήκευση αποτελέσματος --------
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

            # αν θες λίγο "χαλάρωμα" για I/O geoid/DEM
            time.sleep(0.05)

        except Exception as e:
            print(f"[err] πρόβλημα στη γραμμή {idx}: {e}")
            continue

    out_df = pd.DataFrame(results)
    out_df.to_csv(output_csv, index=False, float_format="%.8f")
    print(f"[info] Αποθήκευσα {len(results)} σημεία στο {output_csv}")


if __name__ == "__main__":
    EXCEL_PATH = r"C:\Users\User\OneDrive - Democritus University of Thrace\Έγγραφα\GitHub\Monitoring_hightways_M350RTK_ML\Tests\Highway_Footage_Greece\compute_world_coordinates\Only_Photo_isTrue_row.xlsx"

    DEM_PATH = "https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N41_00_E024_00_DEM/Copernicus_DSM_COG_10_N41_00_E024_00_DEM.tif"
    GEOID_PATH = "https://download.osgeo.org/proj/vdatum/egm96_15/egm96_15.gtx"

    FX = 1601280
    FY = 1801440
    CX = 2592
    CY = 1944
    U = 1296
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
        yaw_offset_deg=90.0,
        rcg="zforward_to_zdown",
        max_range_m=6000.0,
        step_m=50.0,
        max_iters=2,
        debug=True,
        output_csv="ground_points_from_excel.csv",
    )
