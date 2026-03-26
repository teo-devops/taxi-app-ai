"""
NYC Taxi AI — Demo App
Ejecutar desde la raíz del proyecto:
    streamlit run src/app.py

Requiere en src/models/:
    model_duration.pkl
    model_fare.pkl
    dist_media.csv

Requiere en dataset/:
    taxi_zone_lookup.csv
"""

import streamlit as st
import pandas as pd
import joblib
import os
import plotly.graph_objects as go
from datetime import datetime

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="NYC Taxi AI",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded",
)

SRC     = os.path.dirname(os.path.abspath(__file__))
ROOT    = os.path.dirname(SRC)
DATASET = os.path.join(ROOT, "dataset")
MODELS  = os.path.join(SRC, "models")

FEATURES = [
    "PULocationID", "DOLocationID", "trip_distance",
    "hour", "day_of_week", "month_num", "is_weekend", "fleet",
]
DIAS = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]

# ─────────────────────────────────────────────
# ESTILOS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@700&family=DM+Sans:wght@300;400;600&display=swap');
html,[class*="css"]{font-family:'DM Sans',sans-serif;}
.stApp{background:#0d0d14;}
section[data-testid="stSidebar"]{background:#13131f;border-right:1px solid #1e1e30;}
.hero{font-family:'Space Mono',monospace;font-size:1.8rem;font-weight:700;color:#f5c518;line-height:1.1;}
.sub{color:#444;font-size:.8rem;text-transform:uppercase;letter-spacing:.1em;margin-top:4px;}
.lbl{color:#444;font-size:.7rem;text-transform:uppercase;letter-spacing:.12em;margin:18px 0 6px;}
.card{background:#13131f;border:1px solid #1e1e30;border-radius:14px;padding:22px;text-align:center;}
.big{font-family:'Space Mono',monospace;font-size:2.4rem;font-weight:700;color:#f5c518;line-height:1;}
.unit{font-size:1rem;color:#f5c518;opacity:.6;}
.label{font-size:.72rem;color:#444;text-transform:uppercase;letter-spacing:.1em;margin-top:6px;}
.row{display:flex;gap:20px;flex-wrap:wrap;background:#13131f;border:1px solid #1e1e30;border-radius:10px;padding:14px 18px;margin-top:6px;}
.stat{display:flex;flex-direction:column;}
.sv{font-family:'Space Mono',monospace;font-size:.9rem;color:#fff;}
.sl{font-size:.68rem;color:#444;text-transform:uppercase;letter-spacing:.08em;}
.route{background:#13131f;border:1px solid #1e1e30;border-left:3px solid #f5c518;border-radius:10px;padding:12px 16px;margin-bottom:10px;}
.rname{color:#fff;font-weight:600;}
.rmeta{color:#444;font-size:.78rem;margin-top:4px;}
.by{display:inline-block;padding:2px 9px;border-radius:20px;font-size:.7rem;font-weight:700;}
.yel{background:#f5c518;color:#000;}
.grn{background:#00b894;color:#000;}
.dist-auto{color:#555;font-size:.72rem;margin-top:4px;font-style:italic;}
div[data-testid="stButton"]>button{background:#f5c518!important;color:#000!important;font-weight:700!important;font-family:'Space Mono',monospace!important;border:none!important;border-radius:10px!important;width:100%!important;padding:13px!important;}
div[data-testid="stButton"]>button:hover{background:#ffd700!important;box-shadow:0 6px 20px rgba(245,197,24,.2)!important;}
hr{border-color:#1e1e30!important;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CARGA DE RECURSOS (cacheados)
# ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    dur_path  = os.path.join(MODELS, "model_duration.pkl")
    fare_path = os.path.join(MODELS, "model_fare.pkl")
    if not os.path.exists(dur_path) or not os.path.exists(fare_path):
        return None, None
    return joblib.load(dur_path), joblib.load(fare_path)

@st.cache_data
def load_zones():
    return pd.read_csv(os.path.join(DATASET, "taxi_zone_lookup.csv"))

@st.cache_data
def load_dist_map():
    """Distancia media histórica por par PU/DO. Opcional."""
    path = os.path.join(MODELS, "dist_media.csv")
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    return {(int(r.PULocationID), int(r.DOLocationID)): float(r.dist_media_millas)
            for _, r in df.iterrows()}

m_dur, m_fare = load_models()
df_z          = load_zones()
dist_map      = load_dist_map()

zone_map  = dict(zip(df_z["LocationID"], df_z["Zone"]))
boro_map  = dict(zip(df_z["LocationID"], df_z["Borough"]))

_sorted   = sorted(zone_map.items(), key=lambda x: str(x[1]))
zone_ids  = [z[0] for z in _sorted]
zone_lbl  = [f"{z[1]}  ·  {boro_map.get(z[0], '')}" for z in _sorted]

models_ok = m_dur is not None and m_fare is not None
has_dist  = len(dist_map) > 0


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div class="hero">NYC<br>Taxi AI</div>'
        '<div class="sub">Predictor de duración y coste</div>',
        unsafe_allow_html=True,
    )
    st.markdown("<hr>", unsafe_allow_html=True)

    if not models_ok:
        st.error("Modelos no encontrados.\nEjecuta 02_predictive.ipynb hasta el final.")

    # ── Ruta ──
    st.markdown('<div class="lbl">Ruta</div>', unsafe_allow_html=True)
    pu_i = st.selectbox(
        "Zona de recogida", range(len(zone_ids)),
        format_func=lambda i: zone_lbl[i],
        index=zone_ids.index(161) if 161 in zone_ids else 0,
    )
    do_i = st.selectbox(
        "Zona de destino", range(len(zone_ids)),
        format_func=lambda i: zone_lbl[i],
        index=zone_ids.index(237) if 237 in zone_ids else 1,
    )

    # pu_id y do_id se definen aquí — antes de cualquier uso
    pu_id = int(zone_ids[pu_i])
    do_id = int(zone_ids[do_i])

    # Distancia automática desde historial, fallback 2.5 mi
    dist_auto = dist_map.get((pu_id, do_id), 2.5)

    # ── Tiempo ──
    st.markdown('<div class="lbl">Tiempo</div>', unsafe_allow_html=True)
    hora  = st.slider("Hora del día", 0, 23, 9, format="%d:00")
    dia   = st.selectbox("Día", DIAS, index=1)
    dia_n = DIAS.index(dia)
    finde = int(dia_n >= 5)

    # ── Flota ──
    st.markdown('<div class="lbl">Flota</div>', unsafe_allow_html=True)
    flota_l = st.radio("Taxi", ["Yellow", "Green"], horizontal=True)
    flota   = 0 if flota_l == "Yellow" else 1

    # Mostrar distancia usada (informativo)
    src_dist = "histórica" if (pu_id, do_id) in dist_map else "estimada"
    st.markdown(
        f'<div class="dist-auto">Distancia {src_dist}: {dist_auto:.2f} mi</div>',
        unsafe_allow_html=True,
    )

    st.markdown("<hr>", unsafe_allow_html=True)
    go_btn = st.button("PREDECIR VIAJE", disabled=not models_ok)


# ─────────────────────────────────────────────
# LAYOUT PRINCIPAL
# ─────────────────────────────────────────────
col_l, col_r = st.columns([1, 1.5], gap="large")

# ══ COLUMNA IZQUIERDA — Predicción ═══════════
with col_l:

    # Resumen de ruta
    pn = zone_map.get(pu_id, str(pu_id))
    dn = zone_map.get(do_id, str(do_id))
    pb = boro_map.get(pu_id, "")
    db = boro_map.get(do_id, "")
    bc = "yel" if flota == 0 else "grn"
    bt = "YELLOW" if flota == 0 else "GREEN"

    st.markdown(f"""
    <div class="route">
        <span class="by {bc}">{bt}</span>
        <span class="rname">&nbsp;{pn} &rarr; {dn}</span>
        <div class="rmeta">
            {pb} &rarr; {db}
            &nbsp;&middot;&nbsp; {hora:02d}:00
            &nbsp;&middot;&nbsp; {dia}
            &nbsp;&middot;&nbsp; {dist_auto:.2f} mi
        </div>
    </div>""", unsafe_allow_html=True)

    # Predicción
    if go_btn:
        if pu_id == do_id:
            st.warning("Origen y destino son la misma zona.")
        else:
            X = pd.DataFrame(
                [[pu_id, do_id, dist_auto, hora, dia_n,
                  datetime.now().month, finde, flota]],
                columns=FEATURES,
            )
            dur  = max(float(m_dur.predict(X)[0]),  1.0)
            fare = max(float(m_fare.predict(X)[0]), 3.0)
            km   = dist_auto * 1.609
            vel  = km / dur * 60

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(
                    f'<div class="card">'
                    f'<div class="big">{dur:.0f}<span class="unit"> min</span></div>'
                    f'<div class="label">Duración estimada</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(
                    f'<div class="card">'
                    f'<div class="big">${fare:.2f}</div>'
                    f'<div class="label">Tarifa estimada</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            st.markdown(f"""
            <div class="row">
                <div class="stat">
                    <span class="sv">{km:.1f} km</span>
                    <span class="sl">Distancia</span>
                </div>
                <div class="stat">
                    <span class="sv">${fare/km:.2f}</span>
                    <span class="sl">$/km</span>
                </div>
                <div class="stat">
                    <span class="sv">${fare/dur:.2f}</span>
                    <span class="sl">$/min</span>
                </div>
                <div class="stat">
                    <span class="sv">{vel:.0f} km/h</span>
                    <span class="sl">Vel. media</span>
                </div>
            </div>""", unsafe_allow_html=True)

            st.session_state["last"] = (dur, fare, km)

    elif "last" not in st.session_state:
        st.markdown(
            '<div style="color:#2a2a3a;text-align:center;padding:40px 0;font-size:.9rem;">'
            'Configura el viaje y pulsa <strong style="color:#444">PREDECIR</strong>'
            '</div>',
            unsafe_allow_html=True,
        )

    # Gráfico estimación por hora
    if models_ok:
        st.markdown(
            '<div class="lbl" style="margin-top:16px">Estimación por hora del día</div>',
            unsafe_allow_html=True,
        )
        mes  = datetime.now().month
        hrs  = list(range(24))
        durs, fares = [], []
        for h in hrs:
            X_h = pd.DataFrame(
                [[pu_id, do_id, dist_auto, h, dia_n, mes, finde, flota]],
                columns=FEATURES,
            )
            durs.append(float(m_dur.predict(X_h)[0]))
            fares.append(float(m_fare.predict(X_h)[0]))

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hrs, y=durs, name="Duración (min)",
            line=dict(color="#f5c518", width=2),
            fill="tozeroy", fillcolor="rgba(245,197,24,0.05)",
            hovertemplate="%{x:02d}:00 → %{y:.1f} min<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=hrs, y=fares, name="Tarifa ($)",
            line=dict(color="#00b894", width=2), yaxis="y2",
            hovertemplate="%{x:02d}:00 → $%{y:.2f}<extra></extra>",
        ))
        fig.add_vline(x=hora, line_dash="dash", line_color="#333", line_width=1)
        fig.update_layout(
            paper_bgcolor="#0d0d14", plot_bgcolor="#13131f",
            font=dict(color="#555", size=10),
            height=220, margin=dict(l=8, r=8, t=8, b=8),
            xaxis=dict(
                gridcolor="#1a1a2a",
                tickvals=list(range(0, 24, 3)),
                ticktext=[f"{h:02d}h" for h in range(0, 24, 3)],
            ),
            yaxis=dict(
                gridcolor="#1a1a2a",
                title=dict(text="min", font=dict(color="#f5c518")),
            ),
            yaxis2=dict(
                overlaying="y", side="right",
                gridcolor="rgba(0,0,0,0)",
                title=dict(text="$", font=dict(color="#00b894")),
            ),
            legend=dict(
                orientation="h", y=1.1,
                bgcolor="rgba(0,0,0,0)", font=dict(size=10),
            ),
        )
        st.plotly_chart(fig, use_container_width=True)


# ══ COLUMNA DERECHA — Mapa ════════════════════
with col_r:
    st.markdown(
        '<div class="lbl">Tarifa estimada por borough desde el origen</div>',
        unsafe_allow_html=True,
    )

    if models_ok:
        mes  = datetime.now().month
        rows = []
        for _, row in df_z.iterrows():
            zid  = int(row["LocationID"])
            boro = str(row["Borough"])
            if zid == pu_id:
                continue
            # Distancia para este par específico
            d = dist_map.get((pu_id, zid), 2.5)
            X_z = pd.DataFrame(
                [[pu_id, zid, d, hora, dia_n, mes, finde, flota]],
                columns=FEATURES,
            )
            rows.append({
                "zone":    row["Zone"],
                "borough": boro,
                "fare":    float(m_fare.predict(X_z)[0]),
                "zone_id": zid,
            })

        df_pred = pd.DataFrame(rows)
        df_boro = (df_pred.groupby("borough")["fare"]
                   .mean().reset_index()
                   .rename(columns={"fare": "fare_mean"}))

        df_cheap = df_pred.nsmallest(5, "fare")[["zone", "fare"]].reset_index(drop=True)
        df_exp   = df_pred.nlargest(5,  "fare")[["zone", "fare"]].reset_index(drop=True)

        # Barras por borough
        fig_b = go.Figure(go.Bar(
            x=df_boro["borough"],
            y=df_boro["fare_mean"].round(2),
            marker=dict(
                color=df_boro["fare_mean"],
                colorscale=[[0, "#1a2a3a"], [0.5, "#f5c518"], [1, "#ff6b35"]],
                showscale=False,
            ),
            text=["$" + str(round(v, 1)) for v in df_boro["fare_mean"]],
            textposition="outside",
            textfont=dict(color="#666", size=10),
            hovertemplate="%{x}<br>Tarifa media: $%{y:.2f}<extra></extra>",
        ))
        fig_b.update_layout(
            paper_bgcolor="#0d0d14", plot_bgcolor="#13131f",
            font=dict(color="#555", size=10),
            height=260, margin=dict(l=8, r=8, t=8, b=8),
            xaxis=dict(gridcolor="rgba(0,0,0,0)"),
            yaxis=dict(gridcolor="#1a1a2a", title=None),
            showlegend=False,
        )
        st.plotly_chart(fig_b, use_container_width=True)

        # Tablas top 5
        t1, t2 = st.columns(2)
        with t1:
            st.markdown('<div class="lbl">5 destinos más baratos</div>', unsafe_allow_html=True)
            st.dataframe(
                df_cheap.rename(columns={"zone": "Zona", "fare": "$ est."})
                        .assign(**{"$ est.": df_cheap["fare"].round(2)}),
                use_container_width=True, hide_index=True,
            )
        with t2:
            st.markdown('<div class="lbl">5 destinos más caros</div>', unsafe_allow_html=True)
            st.dataframe(
                df_exp.rename(columns={"zone": "Zona", "fare": "$ est."})
                      .assign(**{"$ est.": df_exp["fare"].round(2)}),
                use_container_width=True, hide_index=True,
            )
    else:
        st.markdown(
            '<div style="color:#2a2a3a;text-align:center;padding:60px 0;">'
            'Modelos necesarios para mostrar el mapa'
            '</div>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
for col, txt in zip(st.columns(4), [
    "Duración · R²=0.80 · MAE=3.64 min",
    "Tarifa · R²=0.90 · MAE=$2.67",
    "NYC TLC · Marzo + Nov 2025",
    "XGBoost · 7.07M registros limpios",
]):
    with col:
        st.markdown(
            f'<div style="color:#1e1e2a;font-size:.7rem">{txt}</div>',
            unsafe_allow_html=True,
        )