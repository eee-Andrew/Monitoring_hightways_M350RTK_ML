#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DEM Ray-Intersection with Drone + Gimbal Rotation (Fixed Coordinate System)

This script computes the intersection of a camera ray with a DEM (Digital Elevation Model),
using a drone's position (lat, lon, orthometric altitude) and the orientation of the drone
+ gimbal (yaw, pitch, roll). It outputs the ground point (lat, lon, height).

It:
- Builds a world-to-camera rotation from drone OSD + gimbal angles
- Builds a ray in world coordinates
- Intersects that ray with a DEM using a marching + Newton refinement scheme
- Optionally applies geoid undulation to go between ellipsoidal and orthometric heights

"""

from __future__ import annotations
import argparse, math, os, sys, requests
from typing import Optional
import numpy as np
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.warp import Resampling
from pyproj import Geod

# ----------------------------- Cache -----------------------------
def ensure_cached(url_or_path: str, cache_dir: str = "./dem_cache") -> str:
    """
    Ensure a DEM or geoid file is available locally.
    - If `url_or_path` is a HTTP/HTTPS URL, download it once to `cache_dir`
      and reuse the cached file.
    - If it is already a local path, just return it.
    """
    if url_or_path.startswith(("http://", "https://")):
        os.makedirs(cache_dir, exist_ok=True)
        fname = os.path.basename(url_or_path.rstrip("/"))
        # Fallback name if URL doesn't end with a raster file
        if not fname.lower().endswith(('.tif', '.gtx', '.tiff')):
            fname = "tile.tif"
        local = os.path.join(cache_dir, fname)
        if not os.path.exists(local):
            print(f"[cache] downloading to {local} ...")
            with requests.get(url_or_path, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(local, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024*1024):
                        if chunk:
                            f.write(chunk)
        else:
            print(f"[cache] using cached: {local}")
        return local
    # Already a file path
    return url_or_path

# ----------------------------- ENU -----------------------------
def enu_fwd_inv(lat0: float, lon0: float, h0_ell: float):
    """
    Build forward and inverse mapping between (lat, lon, ellipsoidal height)
    and a simple local ENU-like system (x, y, z) centered at (lat0, lon0, h0_ell).

    - fwd(lon, lat, h)  -> (x, y, z)
    - inv(x, y, z)      -> (lon, lat, h)

    x, y are approximated by geodesic distance/azimuth using pyproj.Geod (WGS84).
    z is just the difference in ellipsoidal height.
    """
    geod = Geod(ellps="WGS84")

    def fwd(lon: float, lat: float, h: float):
        # geod.inv gives forward azimuth from (lon0,lat0) to (lon,lat) and distance
        az12, _, dist = geod.inv(lon0, lat0, lon, lat)
        az = math.radians(az12)
        # x: East, y: North
        x = dist * math.sin(az)
        y = dist * math.cos(az)
        z = h - h0_ell
        return x, y, z

    def inv(x: float, y: float, z: float):
        # Inverse of above: from local (x, y) to (lon, lat)
        dist = math.hypot(x, y)
        az_deg = math.degrees(math.atan2(x, y))
        lon2, lat2, _ = geod.fwd(lon0, lat0, az_deg, dist)
        h2 = h0_ell + z
        return lon2, lat2, h2

    return fwd, inv

# ----------------------------- Geoid -----------------------------
def sample_geoid_undulation(lon: float, lat: float, geoid_path: str) -> float:
    """
    Sample geoid undulation N(lon, lat) from a raster geoid model.
    This N is used to go between ellipsoidal and orthometric heights:
        h_ell = h_ortho + N

    Returns 0.0 if something goes wrong, to fail gracefully.
    """
    try:
        with rasterio.open(geoid_path) as ds:
            # Reproject to EPSG:4326 if needed
            with WarpedVRT(ds, crs="EPSG:4326", resampling=Resampling.bilinear) as vrt:
                val = list(vrt.sample([(lon, lat)]))[0][0]
                return float(val) if val is not None and not np.isnan(val) else 0.0
    except Exception as e:
        print(f"[geoid] warning: {e}, using N=0")
        return 0.0

# ----------------------------- DEM -----------------------------
def sample_dem_height(path: str, lat: float, lon: float) -> float:
    """
    Sample DEM orthometric height at (lat, lon).
    Raises if the DEM returns NoData at that location.
    """
    with rasterio.open(path) as ds:
        with WarpedVRT(ds, crs="EPSG:4326", resampling=Resampling.bilinear) as vrt:
            val = list(vrt.sample([(lon, lat)]))[0][0]
            if val is None or np.isnan(val):
                raise RuntimeError("DEM NoData")
            return float(val)

# ----------------------------- Rotations -----------------------------
def Rz(a):
    """Rotation matrix around Z-axis by angle a (radians)."""
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]], float)

def Rx(a):
    """Rotation matrix around X-axis by angle a (radians)."""
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0,  0],
                     [0, c, -s],
                     [0, s,  c]], float)

def Ry(a):
    """Rotation matrix around Y-axis by angle a (radians)."""
    c, s = math.cos(a), math.sin(a)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]], float)

def camera_forward_world(
    drone_yaw_deg: float, drone_pitch_deg: float, drone_roll_deg: float,
    gimbal_yaw_deg: float, gimbal_pitch_deg: float, gimbal_roll_deg: float,
    rcg: str, yaw_is_heading: bool = False, debug: bool = False
) -> np.ndarray:
    """
    Compute the camera "forward" direction in world coordinates.

    Steps:
    1. Interpret drone yaw/pitch/roll as world->drone rotation (R_world_drone).
    2. Interpret gimbal yaw/pitch/roll as drone->gimbal rotation (R_drone_gimbal).
    3. Combine: R_wc = R_world_drone @ R_drone_gimbal (world->camera/gimbal).
    4. Take the camera forward vector (z-axis in camera frame, possibly adjusted
       by `rcg` if camera model uses z-forward vs z-down).
    5. Transform that forward vector into world coordinates and normalize.

    `yaw_is_heading`:
        If True, interpret drone_yaw_deg as compass heading (DJI style),
        and convert it to mathematical yaw.
    """
    # Convert DJI-style heading to mathematical yaw if requested
    if yaw_is_heading:
        # For DJI: heading 0° = North. We convert to yaw w.r.t x-axis.
        drone_yaw_deg = 90.0 - drone_yaw_deg
        if debug:
            print(f"[dbg] DJI heading {drone_yaw_deg + 90:.2f}° -> drone math yaw {drone_yaw_deg:.2f}°")

    # Drone orientation in radians
    psi_d, th_d, ph_d = map(math.radians, (drone_yaw_deg, drone_pitch_deg, drone_roll_deg))
    R_world_drone = Rz(psi_d) @ Rx(th_d) @ Ry(ph_d)

    # Gimbal orientation in radians (relative to drone)
    psi_g, th_g, ph_g = map(math.radians, (gimbal_yaw_deg, gimbal_pitch_deg, gimbal_roll_deg))
    R_drone_gimbal = Rz(psi_g) @ Rx(th_g) @ Ry(ph_g)

    # World -> Camera rotation
    R_wc = R_world_drone @ R_drone_gimbal

    # Base camera forward direction: z-axis of camera frame
    z_cam = np.array([0, 0, 1.0], float)

    # If the camera convention is "z-forward", but we want "z-down", adjust here
    if rcg == "zforward_to_zdown":
        z_cam = Rx(math.radians(90.0)) @ z_cam

    # Transform camera forward into world coordinates
    d_world = R_wc @ z_cam
    d_world /= np.linalg.norm(d_world)

    if debug:
        print(f"[dbg] forward ray (world): {d_world}")

    return d_world

# ----------------------------- Intrinsics -----------------------------
def backproject_ray(u: float, v: float, K: np.ndarray) -> np.ndarray:
    """
    Backproject an image point (u, v) using camera intrinsics K
    into a normalized camera-frame ray direction.

    Returns a 3D unit vector in camera coordinates.
    NOTE: In this script, `d_cam` is computed but not used directly
    because the camera forward direction is computed from attitudes.
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    # Normalized coordinates
    x = (u - cx) / fx
    y = (v - cy) / fy
    r = np.array([x, y, 1.0], float)
    return r / np.linalg.norm(r)

# ----------------------------- Intersection -----------------------------
def intersect_ray_with_dem(
    lat0: float, lon0: float, h0_ortho: float,
    d_world: np.ndarray, dem_path: str,
    geoid_path: Optional[str] = None,
    max_iters: int = 5, tol_m: float = 0.05,
    max_range_m: float = 3000.0, step_m: float = 50.0,
    debug: bool = False
):
    """
    Intersect a ray with a DEM surface using:
    1. Coarse marching along the ray (in ENU-like space),
    2. Then Newton refinement along the same ray.

    Inputs:
    - lat0, lon0, h0_ortho : drone position (orthometric height, e.g. MSL).
    - d_world              : ray direction in world/ENU coords (unit vector).
    - dem_path             : DEM file or URL.
    - geoid_path           : optional geoid model for ellipsoidal<->orthometric conversion.
    - max_iters            : maximum Newton iterations.
    - tol_m                : vertical residual tolerance in meters.
    - max_range_m          : maximum distance along the ray.
    - step_m               : step for coarse marching.
    - debug                : print debugging info.

    Returns:
    - (lat, lon, h_ortho): ground intersection point with DEM.
    """
    dem_local = ensure_cached(dem_path)
    geoid_local = ensure_cached(geoid_path) if geoid_path else None

    # Build ENU conversions around the drone position (but using ellipsoidal height)
    # We still start from orthometric height and convert using N later.
    fwd, inv = enu_fwd_inv(lat0, lon0, h0_ortho)

    # Normalize direction
    d_world = d_world / np.linalg.norm(d_world)

    # ----- Initial guess for distance along ray (w) -----
    try:
        # Height of DEM directly under the drone (approx)
        h_dem_start = sample_dem_height(dem_local, lat0, lon0)
    except:
        # If DEM sample fails at drone location, guess something below the drone
        h_dem_start = h0_ortho - 100

    # Geoid undulation at drone location
    N0 = sample_geoid_undulation(lon0, lat0, geoid_local) if geoid_local else 0.0
    h0_ell = h0_ortho + N0

    # Vertical difference between drone ellipsoidal height and DEM orthometric height
    dz = h0_ell - h_dem_start

    # Ray parameter initial guess based on vertical component
    if abs(d_world[2]) > 1e-6:
        w = dz / d_world[2]
        w = abs(w)
    else:
        # If ray is almost horizontal, just pick some distance
        w = 100.0

    if w <= 0:
        w = 100.0

    if debug:
        print(f"[dbg] initial w = {w:.1f} m (dz={dz:.1f})")

    # ----- Coarse marching along the ray to bracket the intersection -----
    w_current = w
    res_prev = float('inf')
    w_prev = w
    lat_prev, lon_prev = lat0, lon0

    found_valid_point = False  # Track if we ever sampled a valid DEM point

    while w_current <= max_range_m:
        # Convert distance w_current along ray into ENU and then to lat/lon/h
        x, y, z = w_current * d_world
        lon, lat, h_ell = inv(x, y, z)

        try:
            # DEM orthometric height
            h_dem = sample_dem_height(dem_local, lat, lon)
            # Geoid undulation at this point
            N = sample_geoid_undulation(lon, lat, geoid_local) if geoid_local else 0.0

            # Residual between ray altitude and DEM+geoid (in ellipsoidal domain)
            res = h_ell - (h_dem + N)

            if debug and int(w_current) % (step_m * 10) == 0:
                print(f"[dbg] w={w_current:6.1f} res={res:+6.2f}")

            # Check sign change or small residual → crossed the surface
            if res_prev * res < 0 or abs(res) < tol_m:
                found_valid_point = True
                break

            # Update previous values and advance
            w_prev, lat_prev, lon_prev, res_prev = w_current, lat, lon, res
            found_valid_point = True
            w_current += step_m

        except Exception as e:
            # If DEM sample fails (e.g. outside coverage), skip this step
            if debug:
                print(f"[dbg] DEM sample failed at w={w_current:.1f}: {e}")
            w_current += step_m
            continue

    # === Safety net: if no valid point found during marching ===
    if not found_valid_point:
        if debug:
            print("[dbg] No valid DEM intersection during marching. Using initial guess.")
        # Use initial guess as fallback and try to sample DEM around it
        x, y, z = w * d_world
        lon, lat, _ = inv(x, y, z)
        try:
            h_ortho = sample_dem_height(dem_local, lat, lon)
            return lat, lon, h_ortho
        except:
            # Hard failure: no intersection found
            raise RuntimeError("No valid DEM intersection found within range.")

    # === Newton refinement along the ray (starting from best of w_prev / w_current) ===
    # Choose the w with smaller residual as starting point
    w = w_prev if abs(res_prev) < abs(res) else w_current

    for _ in range(max_iters):
        x, y, z = w * d_world
        lon, lat, h_ell = inv(x, y, z)
        try:
            h_dem = sample_dem_height(dem_local, lat, lon)
        except:
            if debug:
                print(f"[dbg] DEM sample failed in Newton at w={w:.1f}")
            break

        N = sample_geoid_undulation(lon, lat, geoid_local) if geoid_local else 0.0
        res = h_ell - (h_dem + N)

        # Stop if residual is already small enough
        if abs(res) < tol_m:
            break

        # 1D Newton step in parameter w, using vertical component of ray
        if abs(d_world[2]) > 1e-6:
            dw = -res / d_world[2]
        else:
            # If ray almost horizontal, fall back to a simple step
            dw = -res

        # Clamp step size for stability
        dw = np.clip(dw, -500, 500)
        w += dw

        # Avoid negative parameter (going "behind" the drone)
        if w < 0:
            w = 100.0

    # Final intersection coordinates
    x, y, z = w * d_world
    lon, lat, _ = inv(x, y, z)

    try:
        # Final orthometric height from DEM
        h_ortho = sample_dem_height(dem_local, lat, lon)
    except:
        # Fallback to DEM under the drone if final sample fails
        h_ortho = h_dem_start

    return lat, lon, h_ortho

# ----------------------------- Main -----------------------------
def main():
    """
    Parse CLI arguments, build camera/world geometry,
    intersect the camera ray with the DEM, and print the ground point.
    """
    ap = argparse.ArgumentParser()
    # Drone position (orthometric altitude)
    ap.add_argument("--lat", type=float, required=True, help="Drone latitude (deg)")
    ap.add_argument("--lon", type=float, required=True, help="Drone longitude (deg)")
    ap.add_argument("--alt_ortho", type=float, required=True, help="Drone orthometric altitude (MSL, m)")

    # Drone attitude (OSD)
    ap.add_argument("--drone_yaw", type=float, required=True, help="Drone yaw (deg)")
    ap.add_argument("--drone_pitch", type=float, required=True, help="Drone pitch (deg)")
    ap.add_argument("--drone_roll", type=float, default=0.0, help="Drone roll (deg, default=0)")

    # Gimbal orientation (relative to drone)
    ap.add_argument("--gimbal_yaw", type=float, required=True, help="Gimbal yaw (deg, relative to drone)")
    ap.add_argument("--gimbal_pitch", type=float, required=True, help="Gimbal pitch (deg, relative to drone)")
    ap.add_argument("--gimbal_roll", type=float, default=0.0, help="Gimbal roll (deg, default=0)")

    # Yaw interpretation
    ap.add_argument("--yaw_is_heading", action="store_true",
                    help="Interpret drone_yaw as compass heading (DJI-style) instead of math yaw")

    # Camera model convention
    ap.add_argument("--rcg", type=str, default="identity",
                    choices=["identity", "zforward_to_zdown"],
                    help="Camera axis convention correction")

    # Intrinsic parameters
    ap.add_argument("--fx", type=float, required=True, help="Focal length fx (pixels)")
    ap.add_argument("--fy", type=float, required=True, help="Focal length fy (pixels)")
    ap.add_argument("--cx", type=float, default=None, help="Principal point cx (pixels, default=2592)")
    ap.add_argument("--cy", type=float, default=None, help="Principal point cy (pixels, default=1944)")
    ap.add_argument("--u", type=float, default=None, help="Image x-coordinate (pixels) of target point")
    ap.add_argument("--v", type=float, default=None, help="Image y-coordinate (pixels) of target point")

    # Elevation models
    ap.add_argument("--dem", type=str, required=True, help="DEM file path or URL")
    ap.add_argument("--geoid", type=str, default=None, help="Optional geoid model file path or URL")

    # Ray/solver parameters
    ap.add_argument("--max_range_m", type=float, default=3000.0, help="Max ray range (m)")
    ap.add_argument("--step_m", type=float, default=50.0, help="Step size for coarse marching (m)")
    ap.add_argument("--max_iters", type=int, default=5, help="Max Newton iterations")
    ap.add_argument("--debug", action="store_true", help="Enable debug prints")

    args = ap.parse_args()

    # Default principal point: center of a 5184x3888 sensor (example values)
    cx = args.cx if args.cx is not None else 2592
    cy = args.cy if args.cy is not None else 1944

    # If (u, v) are not provided, assume ray through principal point (optical axis)
    u = args.u if args.u is not None else cx
    v = args.v if args.v is not None else cy

    # Intrinsics matrix
    K = np.array([[args.fx,     0, cx],
                  [    0,  args.fy, cy],
                  [    0,      0,  1]], float)

    # Ray in camera coordinates (currently not used to tilt ray by pixel)
    d_cam = backproject_ray(u, v, K)

    # Ray "forward" in world coordinates from drone + gimbal attitudes
    d_world = camera_forward_world(
        args.drone_yaw, args.drone_pitch, args.drone_roll,
        args.gimbal_yaw, args.gimbal_pitch, args.gimbal_roll,
        args.rcg, args.yaw_is_heading, args.debug
    )

    # Intersect ray with DEM to find ground point
    lat, lon, h = intersect_ray_with_dem(
        args.lat, args.lon, args.alt_ortho,
        d_world, args.dem,
        geoid_path=args.geoid,
        max_iters=args.max_iters,
        max_range_m=args.max_range_m,
        step_m=args.step_m,
        debug=args.debug
    )

    # Print result
    print("\n=== GROUND POINT ===")
    print(f"Latitude : {lat:.8f}")
    print(f"Longitude: {lon:.8f}")
    print(f"DEM h(MSL): {h:.2f} m")

if __name__ == "__main__":
    main()
