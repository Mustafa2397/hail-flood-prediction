"""
╔══════════════════════════════════════════════════════════════════════════════╗
║       REAL-TIME FLOOD PREDICTION DASHBOARD — HAIL CITY, SAUDI ARABIA        ║
║                          app.py  |  Streamlit + Plotly                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

SETUP (run once):
    pip install streamlit plotly pandas numpy xgboost scikit-learn requests folium streamlit-folium geopy streamlit-autorefresh fpdf2 arabic_reshaper python-bidi

RUN:
    streamlit run app.py
"""

import time
import warnings
import hashlib
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import xgboost as xgb
from sklearn.model_selection import train_test_split
import requests
import folium
from folium.plugins import HeatMap
from folium import IFrame
from geopy.distance import geodesic
from streamlit_folium import st_folium
from streamlit_autorefresh import st_autorefresh
from fpdf import FPDF
import arabic_reshaper
from bidi.algorithm import get_display

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════
#                           PAGE CONFIG  (must be first)
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="نظام إنذار السيول - حائل",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════
#                           CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

CFG = {
    "HISTORICAL_CSV"   : "hail_history.csv",
    "LIVE_CSV"         : "hail_flood_data.csv",
    "RESULTS_CSV"      : "hail_final_results.csv",
    "MODEL_FILE"       : "flood_model.json",
    "REFRESH_SECONDS"  : 30,
    "CHART_HOURS"      : 24,
    "FLOOD_3H_MM"      : 15.0,
    "FLOOD_6H_MM"      : 25.0,
    "FEATURES": [
        "temp", "rain", "humidity", "pressure", "wind_speed",
        "rain_3h", "rain_6h",
        "pressure_drop_1h", "pressure_drop_3h",
        "humidity_x_rain",
        "hour_sin", "hour_cos", "month_sin", "month_cos",
        "temp_rolling_3h", "wind_rolling_3h",
    ],
}

RISK_COLORS = {
    "منخفض جداً" : "#00C853",
    "منخفض"      : "#64DD17",
    "متوسط"      : "#FFD600",
    "مرتفع"     : "#FF6D00",
    "حرج" : "#D50000",
}

def risk_color(pct: float) -> str:
    if pct < 20:  return RISK_COLORS["منخفض جداً"]
    if pct < 40:  return RISK_COLORS["منخفض"]
    if pct < 60:  return RISK_COLORS["متوسط"]
    if pct < 80:  return RISK_COLORS["مرتفع"]
    return RISK_COLORS["حرج"]

def risk_label(pct: float) -> str:
    if pct < 20:  return "منخفض جداً"
    if pct < 40:  return "منخفض"
    if pct < 60:  return "متوسط"
    if pct < 80:  return "مرتفع"
    return "حرج"

# ═══════════════════════════════════════════════════════════════════════════
#           ✅ VULNERABILITY MAP — إحداثيات مدققة لمدينة حائل
# ═══════════════════════════════════════════════════════════════════════════
HAIL_CENTER = (27.5167, 41.6833)

# ✅ إحداثيات مراجعة: كل نقطة ضمن حدود مدينة حائل العمرانية الفعلية
# المسافة القصوى عن المركز لا تتجاوز ~8كم (ضمن النطاق الحضري)
HAIL_VULNERABILITY_POINTS = [
    # ─── أنفاق ومنخفضات حرجة (أعلى معامل ضعف) ───
    {"name": "نفق الملك فهد (تقاطع الشعيب)", "lat": 27.5215, "lon": 41.6930, "v_factor": 1.30},
    {"name": "نفق الأمير سلطان (طريق الملك)", "lat": 27.5183, "lon": 41.6990, "v_factor": 1.25},
    {"name": "تقاطع الطوقي مع الاستقبال", "lat": 27.5190, "lon": 41.6980, "v_factor": 1.22},
    {"name": "تقاطع الستين مع الملك عبدالله", "lat": 27.5235, "lon": 41.6945, "v_factor": 1.18},
    # ─── أودية ومجاري سيول ───
    {"name": "مجرى وادي حائل (الشعيب)", "lat": 27.5340, "lon": 41.6810, "v_factor": 1.20},
    {"name": "وادي القطيفية (المدخل الشرقي)", "lat": 27.5380, "lon": 41.6980, "v_factor": 1.15},
    {"name": "وادي الجثامية (المحطة)", "lat": 27.5320, "lon": 41.7050, "v_factor": 1.10},
    # ─── سد ومنخفضات ───
    {"name": "محيط سد حائل (الشمالي)", "lat": 27.5520, "lon": 41.7120, "v_factor": 1.12},
    {"name": "منخفضات حي السمراء", "lat": 27.4980, "lon": 41.6930, "v_factor": 1.14},
    {"name": "حي النقرة (منخفض عمراني)", "lat": 27.5105, "lon": 41.6845, "v_factor": 1.10},
    # ─── تقاطعات رئيسية ───
    {"name": "شارع الأمير سعد الدين", "lat": 27.5150, "lon": 41.7040, "v_factor": 1.08},
    {"name": "شارع الأمير نايف", "lat": 27.5172, "lon": 41.6890, "v_factor": 1.05},
    # ─── أحياء سكنية ───
    {"name": "حي المشفى (مستشفى الملك خالد)", "lat": 27.5225, "lon": 41.6855, "v_factor": 1.12},
    {"name": "حي الفيصلية", "lat": 27.5120, "lon": 41.6960, "v_factor": 1.00},
    {"name": "حي الزهرة", "lat": 27.5075, "lon": 41.6880, "v_factor": 0.98},
    {"name": "حي المطار (شرق المدينة)", "lat": 27.4920, "lon": 41.7050, "v_factor": 1.03},
    # ─── طرق سريعة ───
    {"name": "شارع الملك خالد الغربي", "lat": 27.5265, "lon": 41.6700, "v_factor": 0.95},
    {"name": "طريق المدينة المنورة", "lat": 27.5380, "lon": 41.7180, "v_factor": 0.92},
    {"name": "تقاطع طريق جبة الشمالي", "lat": 27.5440, "lon": 41.6860, "v_factor": 1.06},
    # ─── منطقة صناعية ───
    {"name": "المنطقة الصناعية (جنوب)", "lat": 27.4860, "lon": 41.7040, "v_factor": 0.88},
]

def generate_dynamic_risks(base_peak_pct, current_temp):
    points_data = []
    for point in HAIL_VULNERABILITY_POINTS:
        # ✅ لا np.random.uniform — النقاط ثابته بدون اهتزاز
        simulated_risk = base_peak_pct * point["v_factor"]
        simulated_risk = max(0.0, min(100.0, simulated_risk))
        
        point_coords = (point["lat"], point["lon"])
        distance_km = geodesic(HAIL_CENTER, point_coords).km
        points_data.append({
            "name": point["name"],
            "lat": point["lat"],
            "lon": point["lon"],
            "risk": simulated_risk,
            "temp": current_temp,
            "distance_km": distance_km
        })
    return points_data

# ═══════════════════════════════════════════════════════════════════════════
#                      PDF REPORT GENERATION
# ═══════════════════════════════════════════════════════════════════════════

FONT_URL = "https://github.com/google/fonts/raw/main/ofl/amiri/Amiri-Regular.ttf"
FONT_PATH = Path("Amiri-Regular.ttf")

def ensure_arabic_font() -> bool:
    """تحميل خط أميري العربي للـ PDF (مرة واحدة فقط)."""
    if FONT_PATH.exists():
        return True
    try:
        urllib.request.urlretrieve(FONT_URL, str(FONT_PATH))
        return True
    except Exception:
        return False

def reshape_ar(text: str) -> str:
    """إعادة تشكيل النص العربي للـ PDF."""
    try:
        reshaped = arabic_reshaper.reshape(text)
        return get_display(reshaped)
    except Exception:
        return text

def generate_pdf_report(peak_pct, map_points, start_date, end_date,
                        cur_temp, cur_rain, cur_humid, cur_press, cur_wind,
                        cur_rain3h, cur_rain6h, max_rain3h, max_rain, max_rain6h):
    """إنشاء تقرير PDF كامل."""
    has_font = ensure_arabic_font()
    
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    font_name = 'Amiri' if has_font else 'Helvetica'
    if has_font:
        pdf.add_font('Amiri', '', str(FONT_PATH), uni=True)
    
    # ═══ العنوان ═══
    pdf.set_font(font_name, '', 18)
    title = reshape_ar("تقرير نظام الإنذار المبكر للسيول - حائل")
    pdf.cell(0, 15, title, 0, 1, 'C')
    
    pdf.set_font(font_name, '', 11)
    if start_date and end_date:
        date_txt = reshape_ar(f"الفترة: {start_date.strftime('%Y/%m/%d')} - {end_date.strftime('%Y/%m/%d')}")
    else:
        date_txt = reshape_ar("الفترة: آخر 24 ساعة")
    pdf.cell(0, 10, date_txt, 0, 1, 'C')
    
    pdf.set_font(font_name, '', 9)
    ts_txt = reshape_ar(f"توقيت التقرير: {datetime.now().strftime('%Y/%m/%d %H:%M')}")
    pdf.cell(0, 8, ts_txt, 0, 1, 'C')
    
    pdf.ln(5)
    
    # ═══ مستوى الخطر ═══
    pdf.set_font(font_name, '', 14)
    risk_lbl = risk_label(peak_pct)
    risk_txt = reshape_ar(f"مستوى الخطر العام: {risk_lbl} ({peak_pct:.1f}%)")
    pdf.cell(0, 12, risk_txt, 0, 1, 'R')
    
    pdf.ln(3)
    
    # ═══ الظروف الحالية ═══
    pdf.set_font(font_name, '', 12)
    pdf.cell(0, 10, reshape_ar("══ الظروف الحالية ══"), 0, 1, 'R')
    
    pdf.set_font(font_name, '', 10)
    conditions = [
        f"درجة الحرارة: {cur_temp:.1f} مئوية",
        f"هطول الأمطار الحالي: {cur_rain:.1f} مم",
        f"الرطوبة: {cur_humid:.1f} %",
        f"الضغط الجوي: {cur_press:.1f} hPa",
        f"سرعة الرياح: {cur_wind:.1f} كم/س",
    ]
    for cond in conditions:
        pdf.cell(0, 8, reshape_ar(cond), 0, 1, 'R')
    
    pdf.ln(3)
    
    # ═══ ذروة الهطول ═══
    pdf.set_font(font_name, '', 12)
    pdf.cell(0, 10, reshape_ar("══ ذروة الهطول خلال الفترة ══"), 0, 1, 'R')
    
    pdf.set_font(font_name, '', 10)
    peak_conds = [
        f"أقصى مطر في ساعة واحدة: {max_rain:.1f} مم",
        f"أقصى مجموع 3 ساعات: {max_rain3h:.1f} مم",
        f"أقصى مجموع 6 ساعات: {max_rain6h:.1f} مم",
    ]
    for cond in peak_conds:
        pdf.cell(0, 8, reshape_ar(cond), 0, 1, 'R')
    
    pdf.ln(5)
    
    # ═══ نقاط الضعف العمرانية ═══
    pdf.set_font(font_name, '', 12)
    pdf.cell(0, 10, reshape_ar("══ تقييم نقاط الضعف العمرانية ══"), 0, 1, 'R')
    
    pdf.set_font(font_name, '', 9)
    for p in sorted(map_points, key=lambda x: x['risk'], reverse=True):
        risk_lbl_p = risk_label(p['risk'])
        line = f"{p['name']} — خطر {risk_lbl_p} ({p['risk']:.1f}%) — مسافة {p['distance_km']:.1f} كم"
        pdf.cell(0, 7, reshape_ar(line), 0, 1, 'R')
    
    pdf.ln(5)
    
    # ═══ الإجراءات الموصى بها ═══
    pdf.set_font(font_name, '', 12)
    pdf.cell(0, 10, reshape_ar("══ الإجراءات الموصى بها ══"), 0, 1, 'R')
    
    pdf.set_font(font_name, '', 10)
    if peak_pct >= 80:
        rec = "تنبيه قصوى: تفعيل صفارات الإنذار، إخلاء الأنفاق فوراً، توجيه فرق الدفاع المدني للمناطق المنخفضة."
    elif peak_pct >= 60:
        rec = "إنذار: استعداد فرق الطوارئ، إغلاق الأنفاق والشوارع المنخفضة بشكل استباقي."
    elif peak_pct >= 40:
        rec = "مراقبة: تكثيف المراقبة في الأودية والمنخفضات الطبوغرافية وتجهيز مضخات المياه."
    else:
        rec = "أمان: لا توجد إجراءات وقائية مستعجلة، استمرار الرصد الروتيني."
    
    pdf.multi_cell(0, 8, reshape_ar(rec), 0, 'R')
    
    # ═══ التوقيع ═══
    pdf.ln(10)
    pdf.set_font(font_name, '', 8)
    footer = reshape_ar(f"تقرير مولد آلياً — نظام الإنذار المبكر للسيول بحائل — {datetime.now().strftime('%Y/%m/%d %H:%M')}")
    pdf.cell(0, 8, footer, 0, 1, 'C')
    
    return bytes(pdf.output())

# ═══════════════════════════════════════════════════════════════════════════
#                           GLOBAL CSS (RTL & Arabic)
# ═══════════════════════════════════════════════════════════════════════════
DARK_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@300;400;600;700&family=JetBrains+Mono:wght@400;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Cairo', sans-serif;
    background: #080C14;
    color: #C8D6E8;
    direction: ltr; 
    text-align: left;
}

.main .block-container {
    direction: rtl !important;
    text-align: right !important;
    padding: 1.2rem 2rem 3rem 2rem; 
    max-width: 1600px; 
}

section[data-testid="stSidebar"] {
    background: #0A1628;
    border-right: 1px solid #1E3A5F; 
    direction: ltr !important; 
    text-align: left !important;
}
section[data-testid="stSidebar"] > div:first-child {
    padding: 2rem 1rem 2rem 1.5rem !important; 
}

section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stCaption,
section[data-testid="stSidebar"] .stRadio > div > div > label,
section[data-testid="stSidebar"] .stMarkdown {
    direction: rtl !important;
    text-align: right !important;
}
section[data-testid="stSidebar"] label {
    color: #C8D6E8 !important;
}

.dash-header {
    background: linear-gradient(135deg, #0D1B2A 0%, #0A1628 60%, #091420 100%);
    border: 1px solid #1E3A5F;
    border-radius: 12px;
    padding: 1.4rem 2rem;
    margin-bottom: 1.4rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 4px 40px rgba(0,120,255,0.08);
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.dash-header::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, #0047AB, #00BFFF, #0047AB);
    animation: scanline 3s linear infinite;
}
@keyframes scanline {
    0%   { background-position: -200% 0; }
    100% { background-position: 200% 0; }
}
.header-content { z-index: 1; }
.header-title {
    font-family: 'Cairo', sans-serif;
    font-size: 2rem; font-weight: 700;
    color: #E8F4FD; letter-spacing: 0.02em;
    margin: 0; line-height: 1.4;
}
.header-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem; color: #5B8DB8;
    margin-top: 0.5rem; letter-spacing: 0.05em;
}
.header-status { text-align: left; z-index: 1; }
.status-dot {
    display: inline-block; width: 8px; height: 8px; border-radius: 50%;
    background: #00C853; margin-left: 6px;
    box-shadow: 0 0 8px #00C853;
    animation: pulse 2s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; box-shadow: 0 0 8px #00C853; }
    50%       { opacity: 0.5; box-shadow: 0 0 16px #00C853; }
}
.status-txt {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem; color: #00C853; letter-spacing: 0.1em;
}
.ts-txt {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem; color: #3A6080; margin-top: 0.25rem;
}

.metric-card {
    background: linear-gradient(145deg, #0D1B2A, #091420);
    border: 1px solid #1A3050;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    height: 100%;
    position: relative; overflow: hidden;
    transition: border-color 0.3s, box-shadow 0.3s;
}
.metric-card:hover {
    border-color: #2A5080;
    box-shadow: 0 0 20px rgba(0,100,200,0.12);
}

.section-title {
    font-family: 'Cairo', sans-serif;
    font-size: 0.95rem; font-weight: 600;
    letter-spacing: 0.05em; color: #2A6090;
    border-right: 3px solid #0047AB;
    padding-right: 0.7rem; margin: 1.4rem 0 0.8rem 0;
}

.alert-critical {
    background: linear-gradient(90deg, rgba(213,0,0,0.12), rgba(213,0,0,0.04));
    border: 1px solid rgba(213,0,0,0.4); border-right: 4px solid #D50000;
    border-radius: 8px; padding: 0.9rem 1.2rem; margin: 0.8rem 0;
    font-family: 'Cairo', sans-serif; font-size: 1rem;
    color: #FF5252; letter-spacing: 0.02em;
    animation: alertpulse 1.5s ease-in-out infinite;
}
@keyframes alertpulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.85; }
}
.alert-ok {
    background: rgba(0,200,83,0.06);
    border: 1px solid rgba(0,200,83,0.25); border-right: 4px solid #00C853;
    border-radius: 8px; padding: 0.9rem 1.2rem; margin: 0.8rem 0;
    font-family: 'Cairo', sans-serif; font-size: 1rem;
    color: #00C853; letter-spacing: 0.02em;
}

.stDataFrame { border-radius: 8px; overflow: hidden; }
.js-plotly-plot .plotly { background: transparent !important; }
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
hr { border: none; border-top: 1px solid #1A3050; margin: 1.2rem 0; }
.eng-warning {
    background-color: rgba(255, 214, 0, 0.08);
    border-right: 4px solid #FFD600;
    border-radius: 6px;
    padding: 12px 15px;
    margin-bottom: 15px;
    font-family: 'Cairo', sans-serif;
    font-size: 0.85rem;
    color: #FFD600;
    direction: rtl;
    text-align: right;
    line-height: 1.7;
}
.eng-warning span {
    display: inline-block;
    direction: ltr;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    margin: 0 2px;
}
</style>
"""

# ═══════════════════════════════════════════════════════════════════════════
#                      DATA LOADING HELPERS
# ═══════════════════════════════════════════════════════════════════════════

HIST_MAP = {
    "timestamp": "timestamp", "temperature": "temp",
    "precipitation": "rain",  "humidity": "humidity",
    "pressure": "pressure",   "wind_speed": "wind_speed",
}
LIVE_MAP = {
    "timestamp": "timestamp",        "temperature_c": "temp",
    "precipitation_mm": "rain",      "humidity_pct": "humidity",
    "pressure_hpa": "pressure",      "wind_speed_kmh": "wind_speed",
}
STD_COLS = ["timestamp", "temp", "rain", "humidity", "pressure", "wind_speed"]

def _file_hash(path: str) -> str:
    p = Path(path)
    if not p.exists(): return "missing"
    stat = p.stat()
    return hashlib.md5(f"{stat.st_size}-{stat.st_mtime}".encode()).hexdigest()

def safe_read_csv(path: str, retries: int = 3, delay: float = 0.4) -> pd.DataFrame:
    for attempt in range(retries):
        try:
            return pd.read_csv(path, low_memory=False)
        except (pd.errors.ParserError, OSError, PermissionError) as e:
            if attempt < retries - 1: time.sleep(delay)
            else: return pd.DataFrame()
    return pd.DataFrame()

def load_source(path: str, col_map: dict, label: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists(): return pd.DataFrame(columns=STD_COLS)
    raw = safe_read_csv(path)
    if raw.empty: return pd.DataFrame(columns=STD_COLS)
    
    valid_map = {k: v for k, v in col_map.items() if k in raw.columns}
    df = raw.rename(columns=valid_map)
    available = [c for c in STD_COLS if c in df.columns]
    df = df[available].copy()
    df["source"] = label
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    
    for col in ["temp", "rain", "humidity", "pressure", "wind_speed"]:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")
    if "rain" in df.columns: df["rain"] = df["rain"].fillna(0.0)
    
    if label == "Live":
        feature_cols = [c for c in ["temp","humidity","pressure","wind_speed"] if c in df.columns]
        if feature_cols: df = df.dropna(subset=feature_cols, how="all")
    for col in ["temp", "humidity", "pressure", "wind_speed"]:
        if col in df.columns: df[col] = df[col].ffill().bfill()
    df = df.drop_duplicates(subset=["timestamp"], keep="last")
    return df

@st.cache_data(ttl=1800) 
def fetch_current_live_data() -> pd.DataFrame:
    """يجيب بيانات الطقس اللحظية مباشرة من Open-Meteo API للـ 24 ساعة الجاية"""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 27.52, "longitude": 41.69,
        "forecast_days": 1,
        "hourly": "temperature_2m,relative_humidity_2m,precipitation,surface_pressure,wind_speed_10m",
        "timezone": "Asia/Riyadh"
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data["hourly"])
        rename_map = {
            "time": "timestamp", "temperature_2m": "temp",
            "relative_humidity_2m": "humidity", "precipitation": "rain",
            "surface_pressure": "pressure", "wind_speed_10m": "wind_speed"
        }
        df = df.rename(columns=rename_map)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["source"] = "Live API"
        return df
    except requests.exceptions.RequestException:
        st.warning("⚠️ تعذر الاتصال بخدمة الأرصاد الجوية اللحظية حالياً. يتم عرض البيانات التاريخية المتاحة.")
        return pd.DataFrame()
    except Exception:
        st.warning("⚠️ حدث خطأ غير متوقع في جلب البيانات اللحظية. يتم عرض البيانات التاريخية.")
        return pd.DataFrame()

# ═══════════════════════════════════════════════════════════════════════════
#                      FEATURE ENGINEERING & MODEL
# ═══════════════════════════════════════════════════════════════════════════

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["rain_3h"]          = df["rain"].rolling(3, min_periods=1).sum()
    df["rain_6h"]          = df["rain"].rolling(6, min_periods=1).sum()
    df["pressure_drop_1h"] = df["pressure"].diff(1).fillna(0)
    df["pressure_drop_3h"] = df["pressure"].diff(3).fillna(0)
    df["humidity_x_rain"]  = df["humidity"] * df["rain"]
    df["temp_rolling_3h"]  = df["temp"].rolling(3, min_periods=1).mean()
    df["wind_rolling_3h"]  = df["wind_speed"].rolling(3, min_periods=1).mean()
    hour  = df["timestamp"].dt.hour
    month = df["timestamp"].dt.month
    df["hour_sin"]  = np.sin(2 * np.pi * hour  / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * hour  / 24)
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    return df

@st.cache_resource(show_spinner="⚙️ جاري تدريب نموذج XGBoost على بيانات حائل التاريخية...")
def get_model(hist_path: str):
    model_path = Path(CFG["MODEL_FILE"])
    if model_path.exists():
        model = xgb.XGBClassifier()
        model.load_model(str(model_path))
        model._feature_names = CFG["FEATURES"]
        return model

    hist = load_source(hist_path, HIST_MAP, "Historical")
    if hist.empty or len(hist) < 200: return None
    hist = engineer_features(hist)
    hist["flood_label"] = ((hist["rain_3h"] >= CFG["FLOOD_3H_MM"]) | (hist["rain_6h"] >= CFG["FLOOD_6H_MM"])).astype(int)
    
    available = [f for f in CFG["FEATURES"] if f in hist.columns]
    X = hist[available].fillna(0)
    y = hist["flood_label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y if y.sum() > 10 else None)
    
    model = xgb.XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=3, gamma=0.1, reg_alpha=0.1, reg_lambda=1.0, scale_pos_weight=20, eval_metric="auc", random_state=42, n_jobs=-1)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    model.save_model(str(model_path))
    model._feature_names = available
    return model

def predict_df(model, df: pd.DataFrame) -> pd.DataFrame:
    features = getattr(model, "_feature_names", CFG["FEATURES"])
    available = [f for f in features if f in df.columns]
    X = df[available].reindex(columns=features, fill_value=0).fillna(0)
    proba = model.predict_proba(X)[:, 1]
    df = df.copy()
    df["flood_pct"] = (proba * 100).round(2)
    return df

# ═══════════════════════════════════════════════════════════════════════════
#                        CHART BUILDERS (Arabic Titles)
# ═══════════════════════════════════════════════════════════════════════════

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(8,12,20,0.6)",
    font=dict(family="Cairo, JetBrains Mono", color="#5B8DB8", size=11),
    margin=dict(l=10, r=10, t=36, b=10), legend=dict(bgcolor="rgba(13,27,42,0.8)", bordercolor="#1A3050", borderwidth=1, font=dict(size=10)),
    xaxis=dict(gridcolor="#0F2035", showgrid=True, zeroline=False, tickfont=dict(size=9), linecolor="#1A3050"),
    yaxis=dict(gridcolor="#0F2035", showgrid=True, zeroline=False, tickfont=dict(size=9), linecolor="#1A3050"),
)

def chart_rain_vs_flood(df_filtered: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df_filtered["timestamp"], y=df_filtered["rain"], name="هطول الأمطار (مم)", marker_color="rgba(0,150,255,0.55)", marker_line_width=0, yaxis="y"))
    fig.add_trace(go.Scatter(x=df_filtered["timestamp"], y=df_filtered["rain_3h"], name="مجموع 3 ساعات", mode="lines", line=dict(color="#00BFFF", width=1.5, dash="dot"), yaxis="y"))
    fill_color = [risk_color(p) for p in df_filtered["flood_pct"]]
    fig.add_trace(go.Scatter(x=df_filtered["timestamp"], y=df_filtered["flood_pct"], name="نسبة مخاطر الفيضان %", mode="lines+markers", line=dict(color="#FF6D00", width=2.5), marker=dict(size=5, color=fill_color, line=dict(width=0)), yaxis="y2"))
    fig.add_hline(y=60, line_dash="dash", line_color="rgba(213,0,0,0.4)", line_width=1, annotation_text="⚠ خطر مرتفع", annotation_font_color="rgba(213,0,0,0.7)", annotation_font_size=9, yref="y2")
    fig.update_layout(**PLOT_LAYOUT, title=dict(text="هطول الأمطار ومخاطر الفيضان", font=dict(family="Cairo", size=14, color="#7AADD4"), x=0), yaxis2=dict(title="%", overlaying="y", side="left", range=[0, 105], tickfont=dict(size=9), gridcolor="rgba(0,0,0,0)", zeroline=False), barmode="overlay", hovermode="x unified")
    return fig

def chart_gauge(flood_pct: float) -> go.Figure:
    color = risk_color(flood_pct); label = risk_label(flood_pct)
    fig = go.Figure(go.Indicator(mode="gauge+number+delta", value=flood_pct, number=dict(suffix="%", font=dict(family="Cairo", size=42, color=color)), delta=dict(reference=20, suffix="%", increasing=dict(color="#D50000"), decreasing=dict(color="#00C853")), gauge=dict(axis=dict(range=[0, 100], tickwidth=1, tickcolor="#1A3050", tickfont=dict(size=9, color="#3A6080"), nticks=6), bar=dict(color=color, thickness=0.22), bgcolor="rgba(13,27,42,0.6)", borderwidth=1, bordercolor="#1A3050", steps=[dict(range=[0,20], color="rgba(0,200,83,0.08)"), dict(range=[20,40], color="rgba(100,221,23,0.08)"), dict(range=[40,60], color="rgba(255,214,0,0.08)"), dict(range=[60,80], color="rgba(255,109,0,0.10)"), dict(range=[80,100], color="rgba(213,0,0,0.14)")], threshold=dict(line=dict(color="#FF5252", width=3), thickness=0.75, value=60)), title=dict(text=f"<b>{label}</b>", font=dict(family="Cairo", size=16, color=color)), domain=dict(x=[0, 1], y=[0, 1])))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=20, r=20, t=30, b=10), font=dict(family="Cairo", color="#5B8DB8"), height=280)
    return fig

def chart_temperature(df_filtered: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_filtered["timestamp"], y=df_filtered["temp"], name="درجة الحرارة °م", mode="lines", line=dict(color="#FF6D00", width=2), fill="tozeroy", fillcolor="rgba(255,109,0,0.06)"))
    fig.add_trace(go.Scatter(x=df_filtered["timestamp"], y=df_filtered["humidity"], name="الرطوبة %", mode="lines", line=dict(color="#00BFFF", width=1.5, dash="dot"), yaxis="y2"))
    fig.update_layout(**PLOT_LAYOUT, title=dict(text="درجة الحرارة والرطوبة", font=dict(family="Cairo", size=14, color="#7AADD4"), x=0), yaxis2=dict(title="%", overlaying="y", side="left", range=[0, 110], tickfont=dict(size=9), gridcolor="rgba(0,0,0,0)", zeroline=False), hovermode="x unified", height=240)
    return fig

def chart_pressure(df_filtered: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_filtered["timestamp"], y=df_filtered["pressure"], name="الضغط الجوي hPa", mode="lines", line=dict(color="#B388FF", width=2), fill="tozeroy", fillcolor="rgba(179,136,255,0.05)"))
    fig.update_layout(**PLOT_LAYOUT, title=dict(text="الضغط الجوي", font=dict(family="Cairo", size=14, color="#7AADD4"), x=0), hovermode="x unified", height=240)
    return fig

def chart_wind(df_filtered: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_filtered["timestamp"], y=df_filtered["wind_speed"], name="سرعة الرياح كم/س", mode="lines", line=dict(color="#40C4FF", width=2), fill="tozeroy", fillcolor="rgba(64,196,255,0.06)"))
    fig.update_layout(**PLOT_LAYOUT, title=dict(text="سرعة الرياح", font=dict(family="Cairo", size=14, color="#7AADD4"), x=0), hovermode="x unified", height=240)
    return fig

# ═══════════════════════════════════════════════════════════════════════════
#                        MAIN DASHBOARD RENDER
# ═══════════════════════════════════════════════════════════════════════════
def render_dashboard(model, hist_df: pd.DataFrame):
    """Render the full dashboard. Called inside the refresh loop."""

    # ── Load & merge data ────────────────────────────────────────────────
    live_df = fetch_current_live_data()
    results_path = Path(CFG["RESULTS_CSV"])
    if results_path.exists():
        try:
            base_df = pd.read_csv(CFG["RESULTS_CSV"], low_memory=False)
            base_df["timestamp"] = pd.to_datetime(base_df["timestamp"], errors="coerce")
            base_df = base_df.dropna(subset=["timestamp"]).sort_values("timestamp")
            if "flood_pct" not in base_df.columns and "flood_probability_pct" in base_df.columns:
                base_df["flood_pct"] = base_df["flood_probability_pct"]
        except Exception:
            base_df = pd.DataFrame()
    else:
        base_df = pd.DataFrame()

    if not live_df.empty:
        live_feat = engineer_features(live_df.copy())
        if model:
            live_pred = predict_df(model, live_feat)
        else:
            live_pred = live_feat.copy()
            live_pred["flood_pct"] = 0.0

        if not base_df.empty:
            combined = pd.concat([base_df, live_pred], ignore_index=True)
            combined = combined.drop_duplicates(subset=["timestamp"], keep="last")
            combined = combined.sort_values("timestamp")
        else:
            combined = live_pred
    else:
        combined = base_df.copy() if not base_df.empty else pd.DataFrame()

    if combined.empty:
        st.error("❌ لا توجد بيانات متاحة. تأكد من وجود ملفات البيانات.")
        return

    min_ts = combined["timestamp"].min()
    max_ts = combined["timestamp"].max()

    # ── Date Filter (Sidebar) ─────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ تحكم وفلترة البيانات")
        st.markdown("---")
        st.info(f"📅 **نطاق البيانات المتاح:**\nمن `{min_ts.strftime('%Y/%m/%d')}`\nإلى `{max_ts.strftime('%Y/%m/%d %H:%M')}`\n\n*ملاحظة: التوقعات اللحظية متاحة لـ 24 ساعة فقط.*")
        
        filter_type = st.radio("اختر نوع الفلترة", ["آخر 24 ساعة", "يوم محدد", "شهر محدد", "فترة مخصصة"], horizontal=False)
        
        start_date = None; end_date = None
        today = datetime.now().date()
        
        if filter_type == "يوم محدد":
            selected_day = st.date_input("اختر اليوم", value=today)
            start_date = pd.to_datetime(selected_day); end_date = start_date + timedelta(days=1)
            
        elif filter_type == "شهر محدد":
            col1, col2 = st.columns(2)
            with col1:
                selected_month = st.selectbox("الشهر", range(1, 13), index=today.month - 1, format_func=lambda x: f"شهر {x}")
            with col2:
                selected_year = st.selectbox("السنة", range(2020, today.year + 1), index=len(range(2020, today.year + 1)) - 1)
            start_date = pd.to_datetime(f"{selected_year}-{selected_month}-01")
            if selected_month == 12:
                end_date = pd.to_datetime(f"{selected_year + 1}-01-01")
            else:
                end_date = pd.to_datetime(f"{selected_year}-{selected_month + 1}-01")
                
        elif filter_type == "فترة مخصصة":
            date_range = st.date_input("من تاريخ - إلى تاريخ", value=(today - timedelta(days=7), today))
            if len(date_range) == 2:
                start_date = pd.to_datetime(date_range[0]); end_date = pd.to_datetime(date_range[1]) + timedelta(days=1)

        st.markdown("---")
        
        if st.button("🔄 تحديث البيانات اللحظية الآن", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
            
        st.markdown(f"""<div style="font-family:'Cairo',sans-serif; font-size:0.62rem; color:#1E3A5F; text-align:center; padding:0.3rem 0 1rem 0;">Flood Early Warning System &nbsp;|&nbsp; Eng. Mustafa Zalam &nbsp;|&nbsp; {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>""", unsafe_allow_html=True)
        
    # ═══ الـ Auto-Refresh ═══
    if filter_type == "آخر 24 ساعة":
        st_autorefresh(interval=300000, key="datarefresh")

    # تطبيق الفلترة
    if start_date and end_date:
        df_filtered = combined[(combined["timestamp"] >= start_date) & (combined["timestamp"] < end_date)].copy()
        if df_filtered.empty:
            st.error(f"⚠️ **لا توجد بيانات للفترة المختارة ({start_date.strftime('%Y/%m/%d')} - {end_date.strftime('%Y/%m/%d')}).**\n\nالنظام لا يملك أرشيفاً لهذه السنوات، والبيانات اللحظية مقتصرة على الـ 24 ساعة القادمة فقط.\n\nيرجى اختيار فترة ضمن النطاق المتاح أعلاه.")
            return
    else:
        cutoff_24h = max_ts - timedelta(hours=CFG["CHART_HOURS"])
        df_filtered = combined[combined["timestamp"] >= cutoff_24h].copy()

    if df_filtered.empty:
        st.error("لا توجد بيانات للفترة المحددة لعرض المؤشرات.")
        return

    # ═══ التنبؤ الديناميكي وأخذ ذروة الخطر ═══
    if "flood_pct" not in df_filtered.columns and model:
        df_filtered = engineer_features(df_filtered.copy())
        df_filtered = predict_df(model, df_filtered)
    elif "flood_pct" not in df_filtered.columns:
        df_filtered["flood_pct"] = 0.0

    peak_flood_pct = float(df_filtered["flood_pct"].max())
    
    # ═══ القيم الحالية (للعرض) ═══
    latest = df_filtered.iloc[-1]
    last_ts = latest["timestamp"]
    cur_temp = float(latest.get("temp", 0) or 0)
    cur_rain = float(latest.get("rain", 0) or 0)
    cur_humid = float(latest.get("humidity", 0) or 0)
    cur_press = float(latest.get("pressure", 0) or 0)
    cur_wind = float(latest.get("wind_speed", 0) or 0)
    cur_rain3h = float(latest.get("rain_3h", 0) or 0)
    cur_rain6h = float(latest.get("rain_6h", 0) or 0)

    # ═══✅ الإصلاح الحرج: Reality Check بأقصى قيم الفترة بالكامل ✐══
    max_rain3h = float(df_filtered["rain_3h"].max()) if "rain_3h" in df_filtered.columns else 0.0
    max_rain   = float(df_filtered["rain"].max())     if "rain" in df_filtered.columns else 0.0
    max_rain6h = float(df_filtered["rain_6h"].max())  if "rain_6h" in df_filtered.columns else 0.0

    # التصحيح المنطقي: لو مفيش أمطار حقيقية في الفترة كلها → نقفل الخطر
    # ✅ لو في أمطار غزيرة → النموذج حر يقرر بدون تقليص
    if max_rain3h <= 0.1 and max_rain <= 0.1:
        peak_flood_pct = min(peak_flood_pct, 15.0)
    elif max_rain3h < 5.0 and max_rain < 2.0:
        peak_flood_pct = min(peak_flood_pct, 40.0)
    # ✅ لو max_rain3h >= 5 أو max_rain >= 2 → لا تقليص!
    # هذا يضمن: يوم 19/11/2022 = أمطار غزيرة = نقاط حمرا على الخريطة

    # استدعاء دالة الخريطة (بعد التصحيح المنطقي)
    current_temp = cur_temp
    map_points = generate_dynamic_risks(peak_flood_pct, current_temp)

    # ═══ إنشاء تقرير PDF ═══
    pdf_bytes = None
    try:
        pdf_bytes = generate_pdf_report(
            peak_pct=peak_flood_pct, map_points=map_points,
            start_date=start_date, end_date=end_date,
            cur_temp=cur_temp, cur_rain=cur_rain, cur_humid=cur_humid,
            cur_press=cur_press, cur_wind=cur_wind,
            cur_rain3h=cur_rain3h, cur_rain6h=cur_rain6h,
            max_rain3h=max_rain3h, max_rain=max_rain, max_rain6h=max_rain6h,
        )
    except Exception:
        pdf_bytes = None

    # ✅ زرار الـ PDF في الـ Sidebar
    if pdf_bytes:
        st.sidebar.download_button(
            label="📄 تحميل تقرير PDF",
            data=pdf_bytes,
            file_name=f"flood_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    # ── Header ───────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="dash-header">
        <div class="header-content">
            <div class="header-title">🌊 نظام الإنذار المبكر للسيول - مدينة حائل</div>
            <div class="header-sub">
                المركز الوطني للأرصاد  ·  منطقة حائل، المملكة العربية السعودية
                &nbsp;|&nbsp; خط العرض 27.52°N  خط الطول 41.69°E  &nbsp;|&nbsp;  محرك XGBoost
            </div>
        </div>
        <div class="header-status">
            <div><span class="status-txt">بث مباشر</span><span class="status-dot"></span></div>
            <div class="ts-txt">آخر تحديث للبيانات: {last_ts.strftime('%Y-%m-%d %H:%M') if hasattr(last_ts,'strftime') else str(last_ts)}</div>
            <div class="ts-txt">وقت التحديث: {datetime.now().strftime('%H:%M:%S')}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Alert banner ─────────────────────────────────────────────────
    if peak_flood_pct >= 60:
        icon = "🚨" if peak_flood_pct >= 80 else "⚠️"
        st.markdown(f'<div class="alert-critical">{icon} تنبيه مخاطر الفيضان — أقصى احتمالية: <b>{peak_flood_pct:.1f}%</b> &nbsp;|&nbsp; خطر {risk_label(peak_flood_pct)} &nbsp;|&nbsp; أقصى أمطار 3س: {max_rain3h:.1f} مم &nbsp;|&nbsp; أقصى أمطار 6س: {max_rain6h:.1f} مم</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="alert-ok">✅ الظروف طبيعية — أقصى مخاطر: {peak_flood_pct:.1f}% ({risk_label(peak_flood_pct)}) &nbsp;|&nbsp; أقصى أمطار 3س: {max_rain3h:.1f} مم</div>', unsafe_allow_html=True)

    # ── Vulnerability Map & Assessment ──────────────────────────
    st.markdown('<div class="section-title">تقييم نقاط الضعف العمرانية (تحليل مكاني واحتمالي)</div>', unsafe_allow_html=True)
    
    if start_date and end_date:
        date_display = f"الفترة من {start_date.strftime('%Y/%m/%d')} إلى {end_date.strftime('%Y/%m/%d')}"
    else:
        date_display = "آخر 24 ساعة"

    st.markdown(f"""
    <div class="eng-warning">
        ⚠️ تنبيه هندسي: بناءً على غياب بيانات الارتفاعات الرقمية (<span>DEM</span>)، تعتمد الخريطة الحرارية على الاستقراء الهيدرولوجي لتاريخ ({date_display}) ومعامل الضعف العمراني لكل نقطة.
    </div>
    """, unsafe_allow_html=True)

    # حساب الإحصائيات
    high_risk_count = sum(1 for p in map_points if p['risk'] > 70)
    medium_risk_count = sum(1 for p in map_points if 30 <= p['risk'] <= 70)
    low_risk_count = sum(1 for p in map_points if p['risk'] < 30)

    col_map, col_stats = st.columns([2, 1])

    with col_stats:
        st.markdown("#### 📊 توزيع المخاطر العمرانية")
        
        m1, m2, m3 = st.columns(3)
        m1.metric("🔴 خطر مرتفع", value=f"{high_risk_count}")
        m2.metric("🟠 خطر متوسط", value=f"{medium_risk_count}")
        m3.metric("🟢 خطر منخفض", value=f"{low_risk_count}")
        
        st.markdown("---")
        
        df_risks = pd.DataFrame({
            "التصنيف": ["مرتفع", "متوسط", "منخفض"],
            "العدد": [high_risk_count, medium_risk_count, low_risk_count]
        })
        fig_pie = px.pie(df_risks, values='العدد', names='التصنيف', 
                         color='التصنيف',
                         color_discrete_map={'مرتفع':'#FF4B4B', 'متوسط':'#FFA500', 'منخفض':'#21C35E'},
                         hole=0.4)
        fig_pie.update_layout(margin=dict(t=20, b=20, l=20, r=20), showlegend=True, paper_bgcolor='rgba(0,0,0,0)', font=dict(family="Cairo, JetBrains Mono", color="#5B8DB8", size=11))
        st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": "hover"})

        if peak_flood_pct >= 80:
            st.error("🚨 **خطر حرج على:**\n- الأنفاق وتحت الجسور\n- المسالك والشعاب التاريخية\n- المخططات النامية\n- التقاطعات الرئيسية")
        elif peak_flood_pct >= 60:
            st.warning("⚠️ **خطر مرتفع على:**\n- مداخل ومخارج الأنفاق\n- الأحياء المجاورة للأودية\n- الشوارع الفرعية")
        elif peak_flood_pct >= 40:
            st.info("💡 **مراقبة:**\n- المنخفضات الطبوغرافية\n- التقاطعات غير المزودة بصرف")
        elif peak_flood_pct >= 20:
            st.info("🌧️ **احتمال:**\n- تكون برك مائية في المنخفضات")
        else:
            st.success("✅ **أمان:** لا توجد إجراءات وقائية مستعجلة، استمرار الرصد الروتيني.")

    with col_map:
        st.markdown("#### 🗺️ الخريطة الحرارية والنقاط الحرجة")
        
        # ✅ مركز الخريطة على وسط مدينة حائل الفعلي
        m = folium.Map(location=[27.5167, 41.6833], zoom_start=13, tiles='CartoDB dark_matter')

        # ─── إضافة طبقة الخريطة الحرارية ───
        heat_data = [[p['lat'], p['lon'], p['risk']] for p in map_points]
        
        # ✅ تعزيز كثافة الخريطة الحرارية عند الخطر المرتفع
        heat_radius = 20 if peak_flood_pct >= 60 else 15
        heat_blur = 25 if peak_flood_pct >= 60 else 20
        
        HeatMap(heat_data, 
                radius=heat_radius, 
                blur=heat_blur, 
                gradient={0.2: 'blue', 0.4: 'lime', 0.6: 'yellow', 0.8: 'orange', 1.0: 'red'},
                min_opacity=0.5).add_to(m)

        # ───✅ إضافة النقاط بهرم بصري واضح ✐───
        for p in map_points:
            risk = p['risk']
            
            popup_html = f"""
            <div style="direction: rtl; font-family: 'Cairo', sans-serif; text-align: center; padding: 10px; min-width: 180px;">
                <h4 style="margin:0 0 8px 0;">{p['name']}</h4>
                <hr style="margin:5px 0;">
                <p style="margin:4px 0; font-size:13px;">🌡️ <b>الحرارة:</b> {p['temp']:.1f} °C</p>
                <p style="margin:4px 0; font-size:13px;">📍 <b>المسافة:</b> {p['distance_km']:.1f} كم</p>
                <p style="margin:4px 0; font-size:15px; font-weight:bold;">⚠️ <b>نسبة الخطر:</b> {p['risk']:.1f}%</p>
            </div>
            """
            popup = folium.Popup(IFrame(html=popup_html, width=220, height=150), max_width=250)            
            if risk > 70:
                # ═══🔴 خطر مرتفع: نقطة كبيرة حمراء بحدود بيضاء بارزة ═══
                folium.CircleMarker(
                    location=[p['lat'], p['lon']],
                    radius=12,
                    popup=popup, 
                    color='white',
                    weight=3,
                    fill=True,
                    fill_color='#D50000',
                    fill_opacity=0.9
                ).add_to(m)
                # إضافة هالة حمراء حول النقطة الحرجة
                folium.CircleMarker(
                    location=[p['lat'], p['lon']],
                    radius=20,
                    color='#D50000',
                    weight=1,
                    fill=True,
                    fill_color='#D50000',
                    fill_opacity=0.15
                ).add_to(m)
                
            elif risk >= 30:
                # ═══🟠 خطر متوسط: نقطة برتقالية متوسطة ═══
                folium.CircleMarker(
                    location=[p['lat'], p['lon']],
                    radius=8,
                    popup=popup, 
                    color='#FF6D00',
                    weight=1.5,
                    fill=True,
                    fill_color='#FF6D00',
                    fill_opacity=0.75
                ).add_to(m)
                
            else:
                # ═══🟢✅ خطر منخفض: نقطة صغيرة جداً خافتة (ما تلفت الانتباه) ═══
                folium.CircleMarker(
                    location=[p['lat'], p['lon']],
                    radius=3,
                    popup=popup, 
                    color='#2A5080',
                    weight=0.5,
                    fill=True,
                    fill_color='#2A5080',
                    fill_opacity=0.35
                ).add_to(m)

        # ✅ returned_objects=[] بتمنع الـ Rerun لما المستخدم يضغط أو يعمل زوم
        st_folium(m, height=500, use_container_width=True, key="hail_flood_map", returned_objects=[])

    # ── Metric cards ────────────────────────────────────────────────────
    st.markdown('<div class="section-title">الظروف الحالية <span style="color:#D50000; font-size:0.7rem; vertical-align:middle;">● مباشر (LIVE)</span></div>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("🌡 الحرارة", f"{cur_temp:.1f} °م")
    c2.metric("🌧 الأمطار الآن", f"{cur_rain:.1f} مم")
    c3.metric("💧 الرطوبة", f"{cur_humid:.1f} %")
    c4.metric("🔵 الضغط الجوي", f"{cur_press:.1f} hPa")
    c5.metric("💨 الرياح", f"{cur_wind:.1f} كم/س")
    c6.metric("⚠️ أقصى خطر", f"{peak_flood_pct:.1f}%", delta=risk_label(peak_flood_pct))

    # صف ذروة الهطول
    st.markdown('<div class="section-title">ذروة الهطول خلال الفترة المحددة</div>', unsafe_allow_html=True)
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("⏱ أقصى مطر/ساعة", f"{max_rain:.1f} مم")
    p2.metric("⏱ أقصى مجموع 3س", f"{max_rain3h:.1f} مم")
    p3.metric("⏱ أقصى مجموع 6س", f"{max_rain6h:.1f} مم")
    p4.metric("🌧 مجموع 3س حالي", f"{cur_rain3h:.1f} مم")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts ──────────────────────────────────────────────────────────
    g_col, r_col = st.columns([1, 2.6])
    with g_col:
        st.markdown('<div class="section-title">أقصى مخاطر الفيضان (الذروة)</div>', unsafe_allow_html=True)
        st.plotly_chart(chart_gauge(peak_flood_pct), use_container_width=True, config={"displayModeBar": "hover"})
        
        pc, wc = st.columns(2)
        pc.metric("🔵 الضغط", f"{cur_press:.1f} hPa")
        wc.metric("💨 الرياح", f"{cur_wind:.1f} كم/س")

        st.markdown('<div class="section-title">📌 الإجراءات الموصى بها</div>', unsafe_allow_html=True)
        if peak_flood_pct >= 80:
            st.error("🚨 **تنبيه قصوى:** تفعيل صفارات الإنذار، إخلاء الأنفاق فوراً، توجيه فرق الدفاع المدني للمناطق المنخفضة.")
        elif peak_flood_pct >= 60:
            st.warning("⚠️ **إنذار:** استعداد فرق الطوارئ، إغلاق الأنفاق والشوارع المنخفضة بشكل استباقي.")
        elif peak_flood_pct >= 40:
            st.info("💡 **مراقبة:** تكثيف المراقبة في الأودية والمنخفضات الطبوغرافية وتجهيز مضخات المياه.")
        else:
            st.success("✅ **أمان:** لا توجد إجراءات وقائية مستعجلة، استمرار الرصد الروتيني.")

    with r_col:
        st.markdown('<div class="section-title">هطول الأمطار مقابل مخاطر الفيضان</div>', unsafe_allow_html=True)
        if not df_filtered.empty and "flood_pct" in df_filtered.columns:
            st.plotly_chart(chart_rain_vs_flood(df_filtered), use_container_width=True, config={"displayModeBar": "hover"})
            
    st.markdown('<div class="section-title">الاتجاهات الجوية</div>', unsafe_allow_html=True)
    tc, pc_col = st.columns(2)
    with tc:
        if not df_filtered.empty: st.plotly_chart(chart_temperature(df_filtered), use_container_width=True, config={"displayModeBar": "hover"})
    with pc_col:
        if not df_filtered.empty: st.plotly_chart(chart_pressure(df_filtered), use_container_width=True, config={"displayModeBar": "hover"})
    if not df_filtered.empty: st.plotly_chart(chart_wind(df_filtered), use_container_width=True, config={"displayModeBar": "hover"})

    # ═══✅ الجدول التاريخي داخل expander + بدون زرار CSV مكرر ═══
    with st.expander("📋 الأحداث التاريخية عالية المخاطر (احتمالية فيضان ≥ 60%)"):
        flood_col = "flood_pct" if "flood_pct" in combined.columns else "flood_probability_pct"
        high_risk = combined[combined.get(flood_col, combined.get("flood_pct", pd.Series(dtype=float))) >= 60].copy()
        if not high_risk.empty:
            display_cols = ["timestamp", "temp", "rain", "rain_3h", "rain_6h", "humidity", "pressure", "wind_speed", flood_col]
            display_cols = [c for c in display_cols if c in high_risk.columns]
            high_risk_show = high_risk[display_cols].tail(50).sort_values("timestamp", ascending=False)
            rename_map = {"timestamp":"التاريخ", "temp":"الحرارة", "rain":"الأمطار", "rain_3h":"مجموع 3س", "rain_6h":"مجموع 6س", "humidity":"الرطوبة", "pressure":"الضغط", "wind_speed":"الرياح", "flood_pct":"نسبة الخطر %", "flood_probability_pct":"نسبة الخطر %"}
            high_risk_show = high_risk_show.rename(columns=rename_map)
            st.dataframe(high_risk_show, use_container_width=True, height=250)
        else:
            st.info("لا توجد أحداث عالية المخاطر في البيانات المحملة.")

    # ── Footer ──────────────────────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    src_label = "مباشر + تاريخي" if not live_df.empty else "تاريخي (Open-Meteo)"
    st.markdown(f"""<div style="font-family:'Cairo',sans-serif; font-size:0.62rem; color:#1E3A5F; text-align:center; padding:0.3rem 0 1rem 0;">نظام إنذار السيول بحائل &nbsp;·&nbsp; مصدر البيانات: {src_label} &nbsp;·&nbsp; النموذج: XGBoost &nbsp;·&nbsp; تحديث تلقائي &nbsp;·&nbsp; {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
#                      STREAMLIT ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def main():
    st.markdown(DARK_CSS, unsafe_allow_html=True)
    model = get_model(CFG["HISTORICAL_CSV"])

    @st.cache_data(ttl=3600, show_spinner=False)
    def load_hist():
        return load_source(CFG["HISTORICAL_CSV"], HIST_MAP, "Historical")
    hist_df = load_hist()

    render_dashboard(model, hist_df)


if __name__ == "__main__":
    main()