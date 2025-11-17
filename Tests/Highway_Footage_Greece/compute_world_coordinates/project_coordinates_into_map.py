import pandas as pd
import folium

# === ΡΥΘΜΙΣΕ ΕΔΩ ΤΟ PATH ΤΟΥ CSV ===
CSV_PATH = r"C:\Users\User\OneDrive - Democritus University of Thrace\Έγγραφα\GitHub\Monitoring_hightways_M350RTK_ML\ground_points_from_excel_snapped.csv"
# Φορτώνουμε το CSV
df = pd.read_csv(CSV_PATH)

# Περιμένουμε στήλες: ground_lat_deg, ground_lon_deg
if not {"ground_lat_deg", "ground_lon_deg"}.issubset(df.columns):
    raise ValueError("Δεν βρήκα τις στήλες 'ground_lat_deg' και 'ground_lon_deg' στο CSV!")

# Κέντρο χάρτη = μέσος όρος lat/lon
center_lat = df["ground_lat_deg"].mean()
center_lon = df["ground_lon_deg"].mean()

# Δημιουργία folium map
m = folium.Map(location=[center_lat, center_lon], zoom_start=15)

# Προσθέτουμε τα σημεία
for _, row in df.iterrows():
    lat = row["ground_lat_deg"]
    lon = row["ground_lon_deg"]
    idx = row.get("row_index", "")
    alt = row.get("ground_h_m", "")

    popup_text = f"row_index: {idx}<br>lat: {lat:.8f}<br>lon: {lon:.8f}<br>h: {alt}"
    folium.CircleMarker(
        location=[lat, lon],
        radius=4,
        popup=folium.Popup(popup_text, max_width=250),
        fill=True,
    ).add_to(m)

# Προαιρετικά: γραμμή που ενώνει τα σημεία με τη σειρά
coords = list(zip(df["ground_lat_deg"], df["ground_lon_deg"]))
folium.PolyLine(coords, weight=2).add_to(m)

# Save σε HTML
OUTPUT_HTML = "ground_points_map.html"
m.save(OUTPUT_HTML)
print(f"[info] Αποθήκευσα χάρτη στο {OUTPUT_HTML}")
