import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from pulp import LpProblem, LpMinimize, LpVariable, lpSum

import plotly.express as px
import folium
from streamlit_folium import st_folium

from io import BytesIO

# ======================================================
# KONFIGURASI HALAMAN
# ======================================================
st.set_page_config(
    page_title="AI Supply Chain & CPI Pangan",
    layout="wide"
)

# ======================================================
# STYLE
# ======================================================
st.markdown("""
<style>
body {background-color:#020617; color:#e5e7eb;}
h1,h2,h3 {color:#38bdf8;}
.card {
    background: linear-gradient(135deg, #020617, #0f172a);
    border: 1px solid #38bdf8;
    border-radius: 18px;
    padding: 20px;
    box-shadow: 0 0 25px rgba(56,189,248,0.35);
}
</style>
""", unsafe_allow_html=True)

st.title("🤖 AI Supply Chain Pangan, CPI & EWS")
st.caption("Prediksi • Optimasi Logistik • Indeks Ketahanan Pangan")

# ======================================================
# UPLOAD FILE (ANTI NoneType)
# ======================================================
uploaded_file = st.file_uploader(
    "📥 Upload Data Excel (.xlsx)",
    type=["xlsx"]
)

if uploaded_file is None:
    st.info("Silakan upload file Excel terlebih dahulu.")
    st.stop()

# ======================================================
# LOAD DATA (df DIJAMIN ADA)
# ======================================================
try:
    df = pd.read_excel(
        uploaded_file,
        sheet_name="Data_Pangan_Wilayah"
    )
except Exception:
    st.error("❌ Gagal membaca file atau sheet 'Data_Pangan_Wilayah' tidak ditemukan.")
    st.stop()

# ======================================================
# VALIDASI STRUKTUR DATA (ANTI CRASH)
# ======================================================
required_cols = [
    "Wilayah",
    "Produksi",
    "Harga_Rata2",
    "Curah_Hujan",
    "Jumlah_Penduduk",
    "Konsumsi_Historis",
    "Latitude",
    "Longitude"
]

missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    st.error(f"❌ Kolom berikut wajib ada: {', '.join(missing_cols)}")
    st.stop()

# ======================================================
# AI – PREDIKSI KONSUMSI
# ======================================================
X = df[[
    "Produksi",
    "Harga_Rata2",
    "Curah_Hujan",
    "Jumlah_Penduduk"
]]
y = df["Konsumsi_Historis"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=6,
    random_state=42
)
model.fit(X_scaled, y)

df["Prediksi_Konsumsi"] = model.predict(X_scaled).round(0)
df["Gap"] = df["Produksi"] - df["Prediksi_Konsumsi"]

# ======================================================
# EARLY WARNING SYSTEM
# ======================================================
def ews(g):
    if g < -200:
        return "MERAH"
    elif g < 0:
        return "KUNING"
    return "HIJAU"

df["EWS"] = df["Gap"].apply(ews)

# ======================================================
# CPI – INDEKS KETAHANAN PANGAN
# ======================================================
cpi_data = df[[
    "Produksi",
    "Harga_Rata2",
    "Prediksi_Konsumsi",
    "Gap",
    "Jumlah_Penduduk"
]].copy()

# variabel negatif dibalik
cpi_data["Harga_Rata2"] *= -1
cpi_data["Jumlah_Penduduk"] *= -1

scaler_cpi = MinMaxScaler()
df["CPI"] = scaler_cpi.fit_transform(cpi_data).mean(axis=1).round(3)

def status_cpi(x):
    if x < 0.4:
        return "RENDAH"
    elif x < 0.7:
        return "SEDANG"
    return "TINGGI"

df["Status_CPI"] = df["CPI"].apply(status_cpi)

# ======================================================
# METRIK UTAMA
# ======================================================
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(
        f"<div class='card'><h3>Total Produksi</h3><h1>{int(df.Produksi.sum())}</h1></div>",
        unsafe_allow_html=True
    )
with c2:
    st.markdown(
        f"<div class='card'><h3>Wilayah Risiko</h3><h1>{(df.EWS=='MERAH').sum()}</h1></div>",
        unsafe_allow_html=True
    )
with c3:
    st.markdown(
        f"<div class='card'><h3>CPI Rata-rata</h3><h1>{df.CPI.mean():.2f}</h1></div>",
        unsafe_allow_html=True
    )
with c4:
    st.markdown(
        f"<div class='card'><h3>CPI Rendah</h3><h1>{(df.Status_CPI=='RENDAH').sum()}</h1></div>",
        unsafe_allow_html=True
    )

st.divider()

# ======================================================
# OPTIMASI LOGISTIK (ANTI EDGE CASE)
# ======================================================
surplus = df[df.Gap > 0]
defisit = df[df.Gap < 0]

if surplus.empty or defisit.empty:
    st.warning("⚠️ Optimasi logistik tidak dijalankan (tidak ada surplus atau defisit).")
    df_logistik = pd.DataFrame()
else:
    lp = LpProblem("Distribusi_Pangan", LpMinimize)
    x = {}

    for i in surplus.index:
        for j in defisit.index:
            x[(i, j)] = LpVariable(f"x_{i}_{j}", lowBound=0)

    lp += lpSum(
        x[(i, j)] * np.sqrt(
            (df.loc[i, "Latitude"] - df.loc[j, "Latitude"]) ** 2 +
            (df.loc[i, "Longitude"] - df.loc[j, "Longitude"]) ** 2
        )
        for i in surplus.index for j in defisit.index
    )

    for i in surplus.index:
        lp += lpSum(x[(i, j)] for j in defisit.index) <= surplus.loc[i, "Gap"]

    for j in defisit.index:
        lp += lpSum(x[(i, j)] for i in surplus.index) >= abs(defisit.loc[j, "Gap"])

    lp.solve()

    df_logistik = pd.DataFrame([
        {
            "Dari": df.loc[i, "Wilayah"],
            "Ke": df.loc[j, "Wilayah"],
            "Jumlah": round(var.value(), 2)
        }
        for (i, j), var in x.items()
        if var.value() and var.value() > 0
    ])

# ======================================================
# PETA
# ======================================================
st.subheader("🗺️ Peta Ketahanan Pangan & EWS")

m = folium.Map(location=[-2.5, 118], zoom_start=5)
warna = {"HIJAU": "green", "KUNING": "orange", "MERAH": "red"}

for _, r in df.iterrows():
    folium.CircleMarker(
        [r.Latitude, r.Longitude],
        radius=9,
        color=warna[r.EWS],
        fill=True,
        popup=f"""
        <b>{r.Wilayah}</b><br>
        CPI: {r.CPI}<br>
        Status CPI: {r.Status_CPI}<br>
        EWS: {r.EWS}
        """
    ).add_to(m)

st_folium(m, width=1200, height=600)

# ======================================================
# VISUALISASI
# ======================================================
fig = px.bar(
    df,
    x="Wilayah",
    y="CPI",
    color="Status_CPI",
    template="plotly_dark",
    title="Indeks Ketahanan Pangan (CPI)"
)
st.plotly_chart(fig, use_container_width=True)

# ======================================================
# EXPORT EXCEL
# ======================================================
st.subheader("💾 Unduh Hasil Analisis")

output = BytesIO()
with pd.ExcelWriter(output, engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="Hasil_Analisis", index=False)
    df_logistik.to_excel(writer, sheet_name="Optimasi_Logistik", index=False)

st.download_button(
    "📥 Download Hasil Analisis (Excel)",
    data=output.getvalue(),
    file_name="hasil_analisis_ai_pangan_cpi.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

st.success("✅ Analisis selesai tanpa error.")
