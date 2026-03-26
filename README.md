# 🚕 NYC Taxi AI

> Análisis end-to-end del dataset de taxis de Nueva York (NYC TLC) — Máster Big Data 2025/26

Proyecto completo de ciencia de datos que cubre adquisición, limpieza, análisis exploratorio, modelado predictivo con XGBoost y despliegue de una aplicación demo en Streamlit.

---

## Estructura del Proyecto

```
taxi_app.ai/
│
├── dataset/                            # Datos fuente (no incluidos en el repo)
│   ├── yellow_tripdata_2025-03.parquet
│   ├── yellow_tripdata_2025-11.parquet
│   ├── green_tripdata_2025-03.parquet
│   ├── green_tripdata_2025-11.parquet
│   └── taxi_zone_lookup.csv            # Mapa de zonas NYC (263 zonas)
│
├── Notebooks/
│   ├── 01_dta.ipynb                    # EDA cuantitativo y cualitativo
│   └── 02_predictive.ipynb             # Entrenamiento del modelo XGBoost
│
├── src/
│   ├── models/                         # Generado al ejecutar 02_predictive.ipynb
│   │   ├── model_duration.pkl
│   │   ├── model_fare.pkl
│   │   └── dist_media.csv              # Distancia media histórica por par PU/DO
│   └── app.py                          # App Streamlit (punto de entrada)
│
├── requirements.txt
└── README.md
```

---

## Datos

Los datos provienen del portal oficial de la **NYC Taxi & Limousine Commission**:
[https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)

Se usan dos meses de 2025 y dos flotas:

| Dataset | Registros brutos | Registros limpios | % Eliminado |
|---|---|---|---|
| Yellow Marzo 2025 | 4.145.257 | 3.635.976 | 12,3% |
| Yellow Noviembre 2025 | 4.181.444 | 3.358.117 | 19,7% |
| Green Marzo 2025 | 51.539 | 42.014 | 18,5% |
| Green Noviembre 2025 | 46.912 | 37.260 | 20,6% |
| **TOTAL** | **8.425.152** | **7.073.367** | **16,0%** |

---

## Instalación

```bash
# 1. Clonar el repositorio
git clone <url-del-repo>
cd taxi-app-ai

# 2. Crear entorno virtual
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
.venv\Scripts\activate           # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Descargar los datos (Parquet) en la carpeta dataset/ (no en repo)
# https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
#yellow_tripdata_2025-03.parquet
#yellow_tripdata_2025-11.parquet
#green_tripdata_2025-11.parquet
#green_tripdata_2025-03.parquet
#taxi_zone_lookup.csv 
```

---

## Uso

### Paso 1 — Análisis exploratorio

Abre y ejecuta `Notebooks/01_dta.ipynb`:

- Análisis pre-descriptivo de los 4 datasets
- Limpieza con lógica geográfica (3 iteraciones)
- Heatmap de trayectos por zona PU/DO
- Diagrama Sankey de las 5 zonas más activas
- Análisis de ratios: $/km, min/km, $/min
- Rentabilidad horaria (Earnings Per Hour)
- Comparativa de ganancias brutas por flota y mes

### Paso 2 — Entrenamiento del modelo

Abre y ejecuta `Notebooks/02_predictive.ipynb`:

- Carga y unión de los 4 datasets limpios
- Feature engineering (hora, día, mes, fin de semana, flota)
- Entrenamiento de dos modelos XGBoost independientes
- Evaluación en test set
- **Exportación automática** de modelos y tabla de distancias a `src/models/`

### Paso 3 — Aplicación demo

```bash
streamlit run src/app.py
# Abre en http://localhost:8501
```

---

## Modelo Predictivo

Dos modelos XGBoost entrenados sobre 7 millones de viajes reales:

| Modelo | MAE | RMSE | R² |
|---|---|---|---|
| Duración (min) | 3.64 min | 6.09 min | 0.80 |
| Tarifa ($) | $2.67 | $4.92 | 0.90 |

**Features:** `PULocationID`, `DOLocationID`, `trip_distance`, `hour`, `day_of_week`, `month_num`, `is_weekend`, `fleet`

**Hiperparámetros:** `n_estimators=300`, `max_depth=6`, `learning_rate=0.1`, `subsample=0.8`, `tree_method=hist`

---

## Aplicación Streamlit

La app carga los modelos `.pkl` y expone una interfaz interactiva:

- Selector de zona origen y destino (263 zonas NYC)
- Hora del día, día de la semana y flota (Yellow / Green)
- **Distancia automática** — calculada como media histórica real del par PU/DO desde `dist_media.csv`, sin que el usuario tenga que introducirla
- Predicción de duración y tarifa con métricas derivadas ($/km, $/min, velocidad media)
- Gráfico de estimación por hora del día para la ruta seleccionada
- Mapa de tarifa estimada por borough desde el origen
- Tablas de 5 destinos más baratos y más caros

---

## Limpieza de Datos

Filtros aplicados en `limpiar_con_logica_nyc()`:

| Filtro | Umbral | Motivo |
|---|---|---|
| `trip_distance >` | 0.18 millas | < 300m = error GPS |
| `trip_distance <` | 100 millas | Imposible dentro de NYC |
| `fare_amount >` | $3.0 | Tarifa mínima oficial NYC TLC |
| `PU/DOLocationID <=` | 263 | Solo zonas oficiales NYC |
| `PULocationID !=` | `DOLocationID` | Evita ratios infinitos |
| `duration_min >` | 1 min | Elimina errores de timestamp |
| `duration_min <` | 300 min | Máximo 5 horas por viaje |
| `fare_amount <` | $600 | Elimina tarifas anómalas |

---

## Tecnologías

| Capa | Tecnología |
|---|---|
| Datos | Python, Pandas, Parquet |
| Modelo | XGBoost, scikit-learn, joblib |
| Visualización | Plotly, Matplotlib, Seaborn |
| App | Streamlit |
| Cloud (producción) | Oracle Cloud Infrastructure (OCI) |

---

## Referencias

- [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
- [NYC Taxi Zone Lookup](https://d3qem0ej3lxixm.cloudfront.net/taxi_zones.zip)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Streamlit Documentation](https://docs.streamlit.io/)
