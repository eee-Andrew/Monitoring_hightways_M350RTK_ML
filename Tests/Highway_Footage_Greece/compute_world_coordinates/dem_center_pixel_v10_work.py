#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SPATRA v9: DEM Ray-Intersection with Drone + Gimbal Rotation (Fixed Coordinate System)
ΔΙΟΡΘΩΜΕΝΟ: Συνδυασμός OSD (drone) + GIMBAL (relative) → R_wc
"""

from __future__ import annotations
import argparse, math, os, sys, requests
from typing import Optional
import numpy as np
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.warp import Resampling
from pyproj import Geod
import time
# ----------------------------- Cache -----------------------------
def ensure_cached(url_or_path: str, cache_dir: str = "./dem_cache") -> str:
    if url_or_path.startswith(("http://", "https://")):
        os.makedirs(cache_dir, exist_ok=True)
        fname = os.path.basename(url_or_path.rstrip("/"))
        if not fname.lower().endswith(('.tif', '.gtx', '.tiff')):
            fname = "tile.tif"
        local = os.path.join(cache_dir, fname)
        if not os.path.exists(local):
            print(f"[cache] downloading to {local} ...")
            with requests.get(url_or_path, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(local, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024*1024):
                        if chunk: f.write(chunk)
        else:
            print(f"[cache] using cached: {local}")
        return local
    return url_or_path

# ----------------------------- ENU -----------------------------
def enu_fwd_inv(lat0: float, lon0: float, h0_ell: float):
    geod = Geod(ellps="WGS84")
    def fwd(lon: float, lat: float, h: float):
        az12, _, dist = geod.inv(lon0, lat0, lon, lat)
        az = math.radians(az12)
        x = dist * math.sin(az)
        y = dist * math.cos(az)
        z = h - h0_ell
        return x, y, z
    def inv(x: float, y: float, z: float):
        dist = math.hypot(x, y)
        az_deg = math.degrees(math.atan2(x, y))
        lon2, lat2, _ = geod.fwd(lon0, lat0, az_deg, dist)
        h2 = h0_ell + z
        return lon2, lat2, h2
    return fwd, inv

# ----------------------------- Geoid -----------------------------
def sample_geoid_undulation(lon: float, lat: float, geoid_path: str) -> float:
    try:
        #time.sleep(1)
        with rasterio.open(geoid_path) as ds:
            with WarpedVRT(ds, crs="EPSG:4326", resampling=Resampling.bilinear) as vrt:
                val = list(vrt.sample([(lon, lat)]))[0][0]
                return float(val) if val is not None and not np.isnan(val) else 0.0
    except Exception as e:
        print(f"[geoid] warning: {e}, using N=0")
        return 0.0

# ----------------------------- DEM -----------------------------
def sample_dem_height(path: str, lat: float, lon: float) -> float:
    #time.sleep(1)
    with rasterio.open(path) as ds:
        with WarpedVRT(ds, crs="EPSG:4326", resampling=Resampling.bilinear) as vrt:
            val = list(vrt.sample([(lon, lat)]))[0][0]
            if val is None or np.isnan(val):
                raise RuntimeError("DEM NoData")
            return float(val)

# ----------------------------- Rotations -----------------------------
def Rz(a): c,s = math.cos(a), math.sin(a); return np.array([[c,-s,0],[s,c,0],[0,0,1]], float)
def Rx(a): c,s = math.cos(a), math.sin(a); return np.array([[1,0,0],[0,c,-s],[0,s,c]], float)
def Ry(a): c,s = math.cos(a), math.sin(a); return np.array([[c,0,s],[0,1,0],[-s,0,c]], float)

def camera_forward_world(drone_yaw_deg: float, drone_pitch_deg: float, drone_roll_deg: float,
                         gimbal_yaw_deg: float, gimbal_pitch_deg: float, gimbal_roll_deg: float,
                         rcg: str, yaw_is_heading: bool = False, debug: bool = False) -> np.ndarray:
    if yaw_is_heading:
        drone_yaw_deg = 90.0 - drone_yaw_deg
        if debug: print(f"[dbg] DJI heading {drone_yaw_deg + 90:.2f}° -> drone math yaw {drone_yaw_deg:.2f}°")

    psi_d, th_d, ph_d = map(math.radians, (drone_yaw_deg, drone_pitch_deg, drone_roll_deg))
    R_world_drone = Rz(psi_d) @ Rx(th_d) @ Ry(ph_d)

    psi_g, th_g, ph_g = map(math.radians, (gimbal_yaw_deg, gimbal_pitch_deg, gimbal_roll_deg))
    R_drone_gimbal = Rz(psi_g) @ Rx(th_g) @ Ry(ph_g)

    R_wc = R_world_drone @ R_drone_gimbal

    z_cam = np.array([0, 0, 1.0], float)
    if rcg == "zforward_to_zdown":
        z_cam = Rx(math.radians(90.0)) @ z_cam

    d_world = R_wc @ z_cam
    d_world /= np.linalg.norm(d_world)
    if debug: print(f"[dbg] forward ray: {d_world}")
    return d_world

# ----------------------------- Intrinsics -----------------------------
def backproject_ray(u: float, v: float, K: np.ndarray) -> np.ndarray:
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    x = (u - cx) / fx
    y = (v - cy) / fy
    r = np.array([x, y, 1.0], float)
    return r / np.linalg.norm(r)

# ----------------------------- Intersection -----------------------------
def intersect_ray_with_dem(lat0: float, lon0: float, h0_ortho: float,
                           d_world: np.ndarray, dem_path: str,
                           geoid_path: Optional[str] = None,
                           max_iters: int = 5, tol_m: float = 0.05,
                           max_range_m: float = 3000.0, step_m: float = 50.0,
                           debug: bool = False):
    dem_local = ensure_cached(dem_path)
    geoid_local = ensure_cached(geoid_path) if geoid_path else None
    fwd, inv = enu_fwd_inv(lat0, lon0, h0_ortho)
    d_world = d_world / np.linalg.norm(d_world)

    # Initial w
    try:
        h_dem_start = sample_dem_height(dem_local, lat0, lon0)
    except:
        h_dem_start = h0_ortho - 100
    N0 = sample_geoid_undulation(lon0, lat0, geoid_local) if geoid_local else 0.0
    h0_ell = h0_ortho + N0
    dz = h0_ell - h_dem_start
    if abs(d_world[2]) > 1e-6:
        w = dz / d_world[2]
        w = abs(w)
    else:
        w = 100.0
    if w <= 0: w = 100.0
    if debug: print(f"[dbg] initial w = {w:.1f} m (dz={dz:.1f})")

    # Marching
    w_current = w
    res_prev = float('inf')
    w_prev = w
    lat_prev, lon_prev = lat0, lon0

    found_valid_point = False  # Track if we ever sampled a valid DEM point

    while w_current <= max_range_m:
        x, y, z = w_current * d_world
        lon, lat, h_ell = inv(x, y, z)
        try:
            h_dem = sample_dem_height(dem_local, lat, lon)
            N = sample_geoid_undulation(lon, lat, geoid_local) if geoid_local else 0.0
            res = h_ell - (h_dem + N)

            if debug and int(w_current) % (step_m*10) == 0:
                print(f"[dbg] w={w_current:6.1f} res={res:+6.2f}")

            if res_prev * res < 0 or abs(res) < tol_m:
                found_valid_point = True
                break

            w_prev, lat_prev, lon_prev, res_prev = w_current, lat, lon, res
            found_valid_point = True
            w_current += step_m

        except Exception as e:
            if debug:
                print(f"[dbg] DEM sample failed at w={w_current:.1f}: {e}")
            w_current += step_m
            continue

    # === Safety net: if no valid point found during marching ===
    if not found_valid_point:
        if debug:
            print("[dbg] No valid DEM intersection during marching. Using initial guess.")
        # Use starting point or fail gracefully
        x, y, z = w * d_world
        lon, lat, _ = inv(x, y, z)
        try:
            h_ortho = sample_dem_height(dem_local, lat, lon)
            return lat, lon, h_ortho
        except:
            raise RuntimeError("No valid DEM intersection found within range.")

    # === Newton refinement starts here — now safe because res/res_prev defined ===
    w = w_prev if abs(res_prev) < abs(res) else w_current
    for _ in range(max_iters):
        x, y, z = w * d_world
        lon, lat, h_ell = inv(x, y, z)
        try:
            h_dem = sample_dem_height(dem_local, lat, lon)
        except:
            if debug: print(f"[dbg] DEM sample failed in Newton at w={w:.1f}")
            break
        N = sample_geoid_undulation(lon, lat, geoid_local) if geoid_local else 0.0
        res = h_ell - (h_dem + N)
        if abs(res) < tol_m:
            break
        dw = -res / d_world[2] if abs(d_world[2]) > 1e-6 else -res
        dw = np.clip(dw, -500, 500)
        w += dw
        if w < 0: w = 100.0

    x, y, z = w * d_world
    lon, lat, _ = inv(x, y, z)
    try:
        h_ortho = sample_dem_height(dem_local, lat, lon)
    except:
        h_ortho = h_dem_start  # fallback
    return lat, lon, h_ortho
# ----------------------------- Main -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--alt_ortho", type=float, required=True)
    # Drone attitude
    ap.add_argument("--drone_yaw", type=float, required=True)
    ap.add_argument("--drone_pitch", type=float, required=True)
    ap.add_argument("--drone_roll", type=float, default=0.0)
    # Gimbal orientation
    ap.add_argument("--gimbal_yaw", type=float, required=True)
    ap.add_argument("--gimbal_pitch", type=float, required=True)
    ap.add_argument("--gimbal_roll", type=float, default=0.0)
    ap.add_argument("--yaw_is_heading", action="store_true")
    ap.add_argument("--rcg", type=str, default="identity", choices=["identity", "zforward_to_zdown"])
    ap.add_argument("--fx", type=float, required=True)
    ap.add_argument("--fy", type=float, required=True)
    ap.add_argument("--cx", type=float, default=None)
    ap.add_argument("--cy", type=float, default=None)
    ap.add_argument("--u", type=float, default=None)
    ap.add_argument("--v", type=float, default=None)
    ap.add_argument("--dem", type=str, required=True)
    ap.add_argument("--geoid", type=str, default=None)
    ap.add_argument("--max_range_m", type=float, default=3000.0)
    ap.add_argument("--step_m", type=float, default=50.0)
    ap.add_argument("--max_iters", type=int, default=5)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    cx = args.cx if args.cx else 2592
    cy = args.cy if args.cy else 1944
    u = args.u if args.u else cx
    v = args.v if args.v else cy
    K = np.array([[args.fx, 0, cx], [0, args.fy, cy], [0, 0, 1]], float)
    d_cam = backproject_ray(u, v, K)

    d_world = camera_forward_world(args.drone_yaw, args.drone_pitch, args.drone_roll,
                                   args.gimbal_yaw, args.gimbal_pitch, args.gimbal_roll,
                                   args.rcg, args.yaw_is_heading, args.debug)

    lat_g, lon_g, h_g = intersect_ray_with_dem(
    lat0=lat0,
    lon0=lon0,
    h0_ortho=alt_ortho,
    d_world=d_world,
    dem_path=dem_path,
    geoid_path=geoid_path,
    max_iters=max_iters,
    tol_m=max_range_m,      # <= μιμούμαστε το bug: tol_m = max_range_m
    max_range_m=step_m,     # <= max_range_m = step_m
    step_m=1.0,             # <= περίπου όπως CLI (debug=True → 1m)
    debug=False,            # <= ώστε να ταιριάζει με CLI
)


    print("\n=== GROUND POINT ===")
    print(f"Latitude : {lat:.8f}")
    print(f"Longitude: {lon:.8f}")
    print(f"DEM h(MSL): {h:.2f} m")

if __name__ == "__main__":
    main()