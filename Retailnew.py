import streamlit as st
import pandas as pd
import numpy as np
from pytrends.request import TrendReq
import plotly.graph_objects as go
import sqlite3
import hashlib
import secrets
import os
import io
import zipfile
from datetime import datetime
import logging

# Optional PDF (ReportLab)
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.platypus import Paragraph
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

# Prophet may not be installed — handle gracefully
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False

# sklearn fallback for LR forecasting
try:
    from sklearn.linear_model import LinearRegression
    LR_AVAILABLE = True
except Exception:
    LinearRegression = None
    LR_AVAILABLE = False

st.set_page_config(page_title="Retail Trends & Forecasting", layout="wide")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# --- INLINE CSS FOR UI TWEAKS ---
# This method is more robust as it doesn't rely on an external file.
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');

    body {
        font-family: 'Poppins', sans-serif;
        background-color: #f0f2f6;
    }

    .main .block-container {
        background-color: #ffffff;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        padding: 2rem;
        margin-top: 2rem;
    }

    .css-1d3f905, .css-18e3th9 {
        background-color: #2c3e50 !important;
        padding: 1rem;
        border-radius: 12px;
    }

    h1 {
        color: #4a90e2;
        font-weight: 600;
        text-align: center;
        margin-bottom: 20px;
    }

    .css-1y4p8ic, .css-1y4p8ic:hover, .css-1y4p8ic:active {
        color: #f5f5f5;
        background-color: transparent;
        border-radius: 8px;
        padding: 10px;
        transition: all 0.2s ease-in-out;
    }
    
    .stButton button {
        background-color: #4a90e2;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 10px 20px;
        border: none;
        transition: all 0.2s ease-in-out;
    }
    
    .stButton button:hover {
        background-color: #357bd8;
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

# -------------------- DATABASE --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
DB_PATH = os.path.join(BASE_DIR, "users.db")

def get_db_connection():
    return sqlite3.connect(DB_PATH)

def init_db():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password_hash TEXT,
            salt TEXT,
            email TEXT,
            role TEXT DEFAULT 'user'
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            message TEXT,
            timestamp TEXT
        )
        """
    )
    conn.commit()
    conn.close()

init_db()

# -------------------- PASSWORD HASHING --------------------
def hash_password(password: str, salt: str = None):
    if salt is None:
        salt = secrets.token_hex(16)
    pw = (salt + password).encode("utf-8")
    h = hashlib.sha256(pw).hexdigest()
    return salt, h

def verify_password(stored_hash: str, stored_salt: str, provided_password: str) -> bool:
    _, h = hash_password(provided_password, salt=stored_salt)
    return secrets.compare_digest(h, stored_hash)

# -------------------- DB OPERATIONS --------------------
def register_user(username, password, email, role):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        salt, h = hash_password(password)
        cur.execute(
            "INSERT INTO users (username, password_hash, salt, email, role) VALUES (?,?,?,?,?)",
            (username, h, salt, email, role),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def verify_user(username, password):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT password_hash, salt FROM users WHERE username=?", (username,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return False
    stored_hash, stored_salt = row
    return verify_password(stored_hash, stored_salt, password)

def reset_password(username, new_password):
    conn = get_db_connection()
    cur = conn.cursor()
    salt, h = hash_password(new_password)
    cur.execute(
        "UPDATE users SET password_hash=?, salt=? WHERE username=?",
        (h, salt, username),
    )
    conn.commit()
    conn.close()

def get_user_role(username):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT role FROM users WHERE username=?", (username,))
    row = cur.fetchone()
    conn.close()
    return row[0] if row else "user"

def save_feedback(username, message):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("INSERT INTO feedback (username, message, timestamp) VALUES (?,?,?)",
                (username, message, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

def update_user_role(username, role):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("UPDATE users SET role=? WHERE username=?", (role, username))
    conn.commit()
    conn.close()

# -------------------- TRENDS --------------------
@st.cache_data(ttl=3600)
def fetch_trends(keywords, timeframe, geo):
    try:
        pt = TrendReq(hl='en-US', tz=0, timeout=(10, 25))
        pt.build_payload(keywords, cat=0, timeframe=timeframe, geo=geo, gprop='')
        df = pt.interest_over_time()
        if df is None or df.empty:
            return pd.DataFrame()
        if 'isPartial' in df.columns:
            df = df.drop(columns=['isPartial'])
        df.index = pd.to_datetime(df.index)
        return df
    except Exception:
        logger.exception("Pytrends fetch failed")
        return pd.DataFrame()

# -------------------- FORECAST --------------------
def prophet_forecast(series, periods, freq='D'):
    df = series.reset_index()
    df.columns = ['ds', 'y']
    df = df.dropna()
    m = Prophet(daily_seasonality=True, yearly_seasonality=True, weekly_seasonality=True)
    m.fit(df)
    future = m.make_future_dataframe(periods=periods, freq=freq)
    forecast = m.predict(future)
    return forecast, m

def lr_forecast(series, periods, freq='D'):
    s = series.dropna().astype(float)
    if s.empty:
        return pd.DataFrame({'ds': [], 'yhat': []})
    arr = s.reset_index()
    arr['t'] = np.arange(len(arr))
    X = arr[['t']].values
    y = arr.iloc[:, 1].values
    if LR_AVAILABLE and LinearRegression is not None:
        model = LinearRegression()
        model.fit(X, y)
        future_t = np.arange(len(arr), len(arr) + periods).reshape(-1, 1)
        preds = model.predict(future_t)
    else:
        last_val = float(s.iloc[-1])
        preds = np.full(periods, last_val)
    last_date = s.index[-1]
    future_index = pd.date_range(start=last_date, periods=periods + 1, freq=freq)[1:]
    forecast = pd.DataFrame({'ds': future_index, 'yhat': preds})
    return forecast

# -------------------- EXPLANATION --------------------
def generate_detailed_explanation(model_name, forecast_df, keyword, horizon):
    """
    Generates a detailed, human-readable explanation of the forecast.
    """
    if model_name != "Prophet":
        return f"This forecast for **{keyword}** was generated using a simple {model_name} model. The forecast is based on the historical trend and assumes similar patterns will continue into the future."

    parts = [f"This forecast for **{keyword}** was created using the **Prophet** model, which breaks down the trend into several components to provide deeper insights."]

    # 1. Overall Trend
    start_val = forecast_df['yhat'].iloc[0]
    end_val = forecast_df['yhat'].iloc[-1]
    change = (end_val - start_val) / start_val if start_val != 0 else 0
    if change > 0.05:
        trend_desc = f"The overall trend is **increasing** with a projected growth of **{change:.1%}** over the next {horizon} days."
    elif change < -0.05:
        trend_desc = f"The overall trend is **decreasing** with a projected decline of **{-change:.1%}** over the next {horizon} days."
    else:
        trend_desc = "The overall trend is relatively **stable** with minor fluctuations expected."
    parts.append(f"### Overall Trend\n\n- {trend_desc}")

    # 2. Key Seasonalities
    seasonality_desc = ""
    weekly_comp = "weekly" in forecast_df.columns
    yearly_comp = "yearly" in forecast_df.columns

    if weekly_comp and yearly_comp:
        seasonality_desc = "The forecast is influenced by both **weekly and yearly seasonal patterns**."
    elif weekly_comp:
        seasonality_desc = "A clear **weekly seasonality** is identified, with demand likely fluctuating on a day-of-week basis."
    elif yearly_comp:
        seasonality_desc = "A strong **yearly seasonality** is detected, indicating that demand will follow a predictable pattern throughout the year (e.g., higher in summer or winter)."
    else:
        seasonality_desc = "No significant weekly or yearly seasonality was detected in the data."
    parts.append(f"### Seasonality\n\n- {seasonality_desc}")

    # 3. Forecasted Period Summary
    peak_date = forecast_df.loc[forecast_df['yhat'].idxmax(), 'ds'].strftime('%Y-%m-%d')
    peak_value = forecast_df['yhat'].max()
    
    summary_desc = f"Based on these patterns, the forecast predicts that the highest demand will occur around **{peak_date}**, with an estimated value of **{peak_value:.2f}**."
    parts.append(f"### Forecast Summary\n\n- {summary_desc}")
    
    return "\n\n".join(parts)


# -------------------- EXPORT HELPERS --------------------
def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=True)
    return buf.getvalue().encode('utf-8')

def forecasts_to_zip_bytes(forecasts: dict) -> bytes:
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for kw, fdf in forecasts.items():
            csv_bytes = fdf.to_csv(index=False).encode('utf-8')
            zf.writestr(f"{kw}_forecast.csv", csv_bytes)
    mem.seek(0)
    return mem.read()

def make_pdf_report_bytes(title: str, keywords, df_hist: pd.DataFrame, forecasts: dict, horizon: int, explanations: dict) -> bytes:
    """
    Simple PDF report built with reportlab (if available). Returns bytes.
    """
    if not REPORTLAB_AVAILABLE:
        return b""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4
    x, y = 40, height - 50
    styles = getSampleStyleSheet()
    
    def draw_text(c, text, x, y, style):
        p = Paragraph(text, style)
        w, h = p.wrapOn(c, width - 2*x, height)
        if y - h < 50:
            c.showPage()
            y = height - 50
        p.drawOn(c, x, y - h)
        return y - h - 10

    # Main Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(x, y, title)
    y -= 20
    
    # Metadata
    c.setFont("Helvetica", 10)
    y = draw_text(c, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", x, y, styles['Normal'])
    y = draw_text(c, f"Keywords: {', '.join(keywords)}", x, y, styles['Normal'])
    y = draw_text(c, f"Horizon (days): {horizon}", x, y, styles['Normal'])
    y -= 20

    # Basic hist stats
    c.setFont("Helvetica-Bold", 12)
    y = draw_text(c, "Historical summary (per keyword):", x, y, styles['Heading2'])
    c.setFont("Helvetica", 10)
    for kw in keywords:
        if kw in df_hist.columns:
            mean_val = float(df_hist[kw].dropna().mean())
            max_val = float(df_hist[kw].dropna().max())
            y = draw_text(c, f"- {kw}: mean={mean_val:.2f}, max={max_val:.2f}", x, y, styles['Normal'])
            if y < 80:
                c.showPage()
                y = height - 50
    
    # Forecast summary
    y -= 10
    c.setFont("Helvetica-Bold", 12)
    y = draw_text(c, "Forecast summary (last forecast point):", x, y, styles['Heading2'])
    c.setFont("Helvetica", 10)
    for kw, fdf in forecasts.items():
        try:
            last = float(fdf['yhat'].iloc[-1])
            y = draw_text(c, f"- {kw}: forecast_last={last:.2f}", x, y, styles['Normal'])
            if y < 80:
                c.showPage()
                y = height - 50
        except Exception:
            continue
            
    # Detailed Explanations
    y -= 10
    for kw in keywords:
        if kw in explanations:
            c.showPage()
            y = height - 50
            c.setFont("Helvetica-Bold", 14)
            y = draw_text(c, f"Detailed Forecast Explanation: {kw}", x, y, styles['Title'])
            c.setFont("Helvetica", 10)
            y = draw_text(c, explanations[kw], x, y, styles['Normal'])
            y -= 20

    c.save()
    buf.seek(0)
    return buf.read()

# -------------------- APP STATE --------------------
if "logged_in" not in st.session_state: st.session_state.logged_in = False
if "username" not in st.session_state: st.session_state.username = ""
if "role" not in st.session_state: st.session_state.role = "user"
if "page" not in st.session_state: st.session_state.page = "login"
if "theme" not in st.session_state: st.session_state.theme = 'light'

# Forecasting-related state
if "last_df" not in st.session_state: st.session_state.last_df = pd.DataFrame()
if "last_forecasts" not in st.session_state: st.session_state.last_forecasts = {}
if "last_models" not in st.session_state: st.session_state.last_models = {}
if "last_keywords" not in st.session_state: st.session_state.last_keywords = []
if "last_horizon" not in st.session_state: st.session_state.last_horizon = 30
if "forecast_history" not in st.session_state: st.session_state.forecast_history = []
if "last_explanations" not in st.session_state: st.session_state.last_explanations = {}

# -------------------- THEME --------------------
def get_css():
    light_theme = """
    :root {
        --primary-color: #26C6DA;
        --secondary-color: #388E3C;
        --background-color: #ffffff;
        --card-background: #eef2f6;
        --text-color: #333333;
        --light-text-color: #555555;
        --border-color: #e0e0e0;
    }
    .stApp {
        background-color: var(--background-color);
        color: var(--text-color);
    }
    .stApp, .stSidebar, .css-1d3f5r2 {
        background-color: var(--background-color);
        color: var(--text-color);
    }
    .stSelectbox>div>div>div {
        background-color: var(--card-background);
        border: 1px solid var(--border-color);
    }
    h1, h2, h3, h4, h5, h6 {
        color: var(--text-color);
    }
    .stMarkdown p {
        color: var(--light-text-color);
    }
    .stButton>button {
        background-color: var(--primary-color);
        color: white;
    }
    .stTextInput>div>div>input {
        background-color: var(--card-background);
        border: 1px solid var(--border-color);
        color: var(--text-color);
    }
    .stDataFrame {
        background-color: var(--card-background);
    }
    """
    dark_theme = """
    :root {
        --primary-color: #26C6DA;
        --secondary-color: #388E3C;
        --background-color: #1a1a1a;
        --card-background: #2b2b2b;
        --text-color: #f0f0f0;
        --light-text-color: #cccccc;
        --border-color: #444444;
    }
    .stApp {
        background-color: var(--background-color);
        color: var(--text-color);
    }
    .stApp, .stSidebar, .css-1d3f5r2 {
        background-color: var(--background-color);
        color: var(--text-color);
    }
    h1, h2, h3, h4, h5, h6 {
        color: var(--text-color);
    }
    .stMarkdown p {
        color: var(--light-text-color);
    }
    .stButton>button {
        background-color: var(--primary-color);
        color: white;
    }
    .stTextInput>div>div>input {
        background-color: var(--card-background);
        border: 1px solid var(--border-color);
        color: var(--text-color);
    }
    .stDataFrame {
        background-color: var(--card-background);
    }
    .stToggle {
        color: var(--text-color);
    }
    """
    if st.session_state.theme == 'dark':
        return dark_theme
    else:
        return light_theme

st.markdown(f"<style>{get_css()}</style>", unsafe_allow_html=True)
st.sidebar.markdown("---")
st.sidebar.subheader("App Settings")
def toggle_theme():
    st.session_state.theme = 'dark' if st.session_state.theme == 'light' else 'light'

st.sidebar.toggle("Dark Mode", value=st.session_state.theme == 'dark', on_change=toggle_theme)

# -------------------- AUTH UI --------------------
if not st.session_state.logged_in:
    menu = ["Login", "Register", "Forgot Password", "Help"]
    choice = st.sidebar.selectbox("Menu", menu, key="auth_menu")

    if choice == "Login":
        st.subheader("🔐 Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if verify_user(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.role = get_user_role(username)
                st.session_state.page = "home" # Redirect to home page
                st.success(f"Welcome {username}! Role: {st.session_state.role}")
                st.rerun()
            else:
                st.error("Invalid username or password.")

    elif choice == "Register":
        st.subheader("📝 Register")
        with st.form(key="register_form"):
            new_username = st.text_input("Choose Username")
            new_email = st.text_input("Email")
            new_password = st.text_input("Choose Password", type="password")
            role_options = ["user", "admin"]
            new_role = st.selectbox("Select Role", role_options)
            register_button = st.form_submit_button("Register")
        
        if register_button:
            if not new_username or not new_email or not new_password:
                st.warning("Please fill in all the details.")
            else:
                if register_user(new_username, new_password, new_email, new_role):
                    st.success("✅ Account created! Please login.")
                else:
                    st.error("Username or email already exists.")

    elif choice == "Forgot Password":
        st.subheader("🔄 Reset Password (manual)")
        username = st.text_input("Enter your username")
        new_pass = st.text_input("Enter new password", type="password")
        if st.button("Reset"):
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute("SELECT id FROM users WHERE username=?", (username,))
            row = cur.fetchone()
            conn.close()
            if row:
                reset_password(username, new_pass)
                st.success("Password reset successful! Please login with your new password.")
            else:
                st.error("Username not found.")

    elif choice == "Help":
        st.title("ℹ️ Help & User Guide")
        st.markdown("""
        **Quick Start**
        1. Register and login.
        2. Go to **Forecasting**: choose Google Trends or upload CSV.
        3. Choose timeframe / region / horizon and click **Fetch + Forecast**.
        4. Download CSV or PDF summary from the Downloads section.

        **Notes**
        - Prophet may take a few seconds to train. If not installed, LR is used as fallback.
        - Google Trends may return rate-limit errors (429) if queried rapidly.
        - PDF export requires `reportlab` (install with `pip install reportlab`).
        """)

else:
    # -------------------- LOGGED-IN MENU --------------------
    menu = ["Home", "Forecasting", "Forecast Explanation", "Analytics Dashboard",
            "Feedback", "Help"]
    if st.session_state.role == "admin":
        menu.append("Admin Panel")
    menu.append("Logout")
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Logout":
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.role = "user"
        st.session_state.page = "login"
        st.success("✅ You have been logged out.")
        st.rerun()

    elif choice == "Home":
        st.title(f"Welcome, {st.session_state.username}!")
        st.subheader(f"Your role is: {st.session_state.role}")
        st.info("Use the sidebar menu to navigate the application. You can fetch and forecast retail trends, view analytics dashboards, and provide feedback.")
        st.markdown("---")
        st.subheader("App Features")
        st.markdown("""
        - **Forecasting**: Use Google Trends data or upload your own sales data to generate future forecasts.
        - **Analytics Dashboard**: Visualize and analyze historical and forecasted data.
        - **Feedback**: Submit your thoughts and suggestions to improve the app.
        - **Admin Panel**: (Admin only) View and manage user feedback and roles.
        """)

# -------------------- FORECASTING --------------------
    elif choice == "Forecasting":
        st.title("📈 Retail Trends & Forecasting")

        st.sidebar.markdown("### Data source")
        use_trends = st.sidebar.checkbox("Google Trends", value=True)
        use_csv = st.sidebar.checkbox("Manual CSV", value=False)

        df = pd.DataFrame()
        keywords = []
        geo = ""

        if use_csv:
            uploaded_file = st.file_uploader("Upload CSV (date,sales[,product...])", type=["csv"])
            if uploaded_file:
                try:
                    raw = pd.read_csv(uploaded_file, parse_dates=["date"])
                    raw = raw.sort_values("date").set_index("date")
                    num_cols = raw.select_dtypes(include=[np.number]).columns
                    if len(num_cols) == 0:
                        st.error("CSV must include at least one numeric column.")
                    else:
                        sales_col = num_cols[0]
                        df = raw[[sales_col]].rename(columns={sales_col: "sales"})
                        keywords = ["sales"]
                except Exception as e:
                    st.error(f"CSV parsing failed: {e}")

        if use_trends:
            kw_input = st.text_input("Keywords (comma separated)", "smartphone,laptop")
            keywords = [k.strip() for k in kw_input.split(",") if k.strip()]

            region_options = {
                "🌐 Global": "",
                "🇺🇸 United States": "US",
                "🇮🇳 India": "IN",
                "🇬🇧 United Kingdom": "GB",
                "🇨🇦 Canada": "CA",
                "🇦🇺 Australia": "AU",
                "🇩🇪 Germany": "DE",
                "🇫🇷 France": "FR",
                "🇯🇵 Japan": "JP",
                "Other (custom)": "OTHER"
            }
            region_choice = st.selectbox("Select Region", list(region_options.keys()))
            if region_options[region_choice] == "OTHER":
                geo = st.text_input("Enter custom region code (2-letter)", value="")
            else:
                geo = region_options[region_choice]

            start_trend = st.date_input("Start Date", pd.to_datetime("2023-01-01"))
            end_trend = st.date_input("End Date", pd.to_datetime("2023-12-31"))
            timeframe = f"{start_trend.strftime('%Y-%m-%d')} {end_trend.strftime('%Y-%m-%d')}"

            with st.spinner("Fetching Google Trends..."):
                df = fetch_trends(keywords, timeframe, geo)

        horizon = st.slider("Forecast horizon (days)", 7, 90, 30)

        if st.button("Fetch + Forecast"):
            if df.empty or not keywords:
                st.warning("⚠️ No data returned. Provide a valid CSV or adjust your keywords/timeframe.")
            else:
                st.subheader("Historical data (tail)")
                st.dataframe(df.tail(10))

                fig_hist = go.Figure()
                for kw in keywords:
                    if kw in df.columns:
                        fig_hist.add_trace(go.Scatter(x=df.index, y=df[kw], mode='lines', name=f"{kw} Hist"))
                st.plotly_chart(fig_hist, use_container_width=True, key=f"hist_{datetime.now().timestamp()}")

                forecasts = {}
                models = {}
                explanations = {}
                compare_fig = go.Figure()

                for kw in keywords:
                    if kw not in df.columns:
                        st.warning(f"No data series named '{kw}' found in data.")
                        continue
                    series = df[kw].astype(float).dropna()
                    st.subheader(f"Forecast — {kw}")

                    try:
                        if PROPHET_AVAILABLE:
                            fc, model = prophet_forecast(series, periods=horizon)
                            models[kw] = model
                            forecasts[kw] = fc
                            display_df = fc[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(horizon)
                            explanations[kw] = generate_detailed_explanation("Prophet", fc, kw, horizon)
                        else:
                            fc = lr_forecast(series, periods=horizon)
                            forecasts[kw] = fc
                            display_df = fc
                            explanations[kw] = generate_detailed_explanation("Linear Regression", fc, kw, horizon)
                    except Exception as e:
                        logger.exception("Forecast error")
                        st.error(f"Forecast failed for {kw}: {e}")
                        continue

                    st.dataframe(display_df.reset_index(drop=True))

                    compare_fig.add_trace(go.Scatter(x=series.index, y=series.values, mode='lines', name=f"{kw} Hist"))
                    if 'ds' in display_df.columns and 'yhat' in display_df.columns:
                        compare_fig.add_trace(go.Scatter(x=display_df['ds'], y=display_df['yhat'], mode='lines', name=f"{kw} Fcst"))

                compare_fig.update_layout(title="Historical vs Forecast Comparison")
                st.plotly_chart(compare_fig, use_container_width=True, key=f"cmp_{datetime.now().timestamp()}")

                st.subheader("Downloads")
                try:
                    hist_csv = df.to_csv(index=True).encode('utf-8')
                    st.download_button("Download Historical (CSV)", data=hist_csv, file_name="historical.csv", mime="text/csv")
                except Exception as e:
                    st.error(f"Failed to prepare historical CSV: {e}")

                for kw, fdf in forecasts.items():
                    try:
                        csv_bytes = fdf.to_csv(index=False).encode('utf-8')
                        st.download_button(f"Download Forecast ({kw}) CSV", data=csv_bytes, file_name=f"{kw}_forecast.csv", mime="text/csv")
                    except Exception as e:
                        st.error(f"Failed to prepare CSV for {kw}: {e}")

                if forecasts:
                    try:
                        zip_bytes = forecasts_to_zip_bytes(forecasts)
                        st.download_button("Download All Forecasts (ZIP)", data=zip_bytes, file_name="forecasts.zip", mime="application/zip")
                    except Exception as e:
                        st.error(f"Failed to prepare ZIP: {e}")

                if REPORTLAB_AVAILABLE:
                    pdf_bytes = make_pdf_report_bytes("Retail Forecast Report", keywords, df, forecasts, horizon, explanations)
                    st.download_button("Download PDF Summary", data=pdf_bytes, file_name="forecast_summary.pdf", mime="application/pdf")
                else:
                    st.info("PDF report requires `reportlab`. Install via `pip install reportlab`.")

                sig = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_{'_'.join(keywords)}_{horizon}"
                st.session_state.forecast_history.append({
                    "sig": sig,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "df": df.copy(),
                    "forecasts": {kw: fdf.copy() for kw, fdf in forecasts.items()},
                    "keywords": keywords[:],
                    "horizon": int(horizon),
                })
                st.session_state.forecast_history = st.session_state.forecast_history[-8:]

                st.session_state.last_df = df.copy()
                st.session_state.last_forecasts = {kw: fdf.copy() for kw, fdf in forecasts.items()}
                st.session_state.last_models = {kw: models.get(kw) for kw in forecasts.keys()}
                st.session_state.last_keywords = keywords[:]
                st.session_state.last_horizon = int(horizon)
                st.session_state.last_explanations = explanations

# ---------------------- FORECAST EXPLANATION ----------------------
    elif choice == "Forecast Explanation":
        st.title("🔍 Forecast Explanation")
        st.info("Get a detailed breakdown of the trends and patterns influencing your forecast.")

        if not st.session_state.last_explanations:
            st.warning("No forecasts available. Run a forecast in the main screen first.")
        else:
            keywords = st.session_state.last_keywords
            kw_choice = st.selectbox("Select series for explanation", keywords)

            if kw_choice in st.session_state.last_explanations:
                model = st.session_state.last_models.get(kw_choice)
                if model and hasattr(model, 'plot_components'):
                    st.subheader(f"Components for: {kw_choice}")
                    try:
                        fc = st.session_state.last_forecasts[kw_choice]
                        fig = model.plot_components(fc)
                        st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Failed to plot components: {e}")

                st.markdown("---")
                st.subheader(f"Explanation for: {kw_choice}")
                st.markdown(st.session_state.last_explanations[kw_choice])
            else:
                st.warning("No detailed explanation available for this series. It might have been generated without Prophet.")


# ---------------------- Analytics Dashboard ----------------------
    elif choice == "Analytics Dashboard":
        st.title("📊 Analytics Dashboard")

        if st.session_state.last_df.empty or not st.session_state.last_forecasts:
            st.warning("⚠️ No forecast available yet. Please run a forecast first.")
        else:
            tab_latest, tab_history = st.tabs(["Latest Forecast", "Forecast History"])

            with tab_latest:
                keywords = st.session_state.last_keywords
                df = st.session_state.last_df
                forecasts = st.session_state.last_forecasts
                horizon = st.session_state.last_horizon

                kw_choice = st.selectbox("Select series to view", keywords)

                if kw_choice in df.columns:
                    series = df[kw_choice].dropna()
                else:
                    series = df[df["keyword"] == kw_choice]["yhat"].dropna()

                col1, col2, col3 = st.columns(3)
                with col1:
                    total_hist = float(series.sum())
                    st.metric("Total (historical)", f"{total_hist:,.0f}")
                with col2:
                    avg_hist = float(series.mean())
                    st.metric("Average (historical)", f"{avg_hist:,.2f}")
                with col3:
                    st.metric("Forecast horizon", f"{horizon} days")

                fig_forecast = go.Figure()
                if not series.empty:
                    fig_forecast.add_trace(go.Scatter(
                        x=series.index if hasattr(series, "index") else None,
                        y=series.values,
                        name="Historical",
                        mode="lines"
                    ))

                fc = forecasts.get(kw_choice)
                if isinstance(fc, pd.DataFrame) and 'ds' in fc.columns and 'yhat' in fc.columns:
                    fig_forecast.add_trace(go.Scatter(x=fc['ds'], y=fc['yhat'], name="Forecast", mode="lines"))
                    if 'yhat_lower' in fc.columns and 'yhat_upper' in fc.columns:
                        fig_forecast.add_trace(go.Scatter(
                            x=pd.concat([fc['ds'], fc['ds'][::-1]]),
                            y=pd.concat([fc['yhat_upper'], fc['yhat_lower'][::-1]]),
                            fill='toself',
                            name='Confidence Interval',
                            mode='lines',
                            line=dict(width=0),
                            opacity=0.2,
                        ))

                fig_forecast.update_layout(title=f"{kw_choice}: Historical & Forecast",
                                           xaxis_title="Date", yaxis_title="Value")
                st.plotly_chart(fig_forecast, use_container_width=True, key=f"forecast_chart_{kw_choice}")

                st.markdown("---")
                st.subheader("Forecasted Demand Comparison")

                # Create a bar chart for total forecasted value
                total_forecasts = {}
                for kw, fc_df in forecasts.items():
                    total_forecasts[kw] = fc_df['yhat'].sum()

                if total_forecasts:
                    bar_fig = go.Figure(go.Bar(
                        x=list(total_forecasts.keys()),
                        y=list(total_forecasts.values())
                    ))
                    bar_fig.update_layout(
                        title="Total Forecasted Demand for all Keywords",
                        xaxis_title="Keywords",
                        yaxis_title="Total Forecasted Value"
                    )
                    st.plotly_chart(bar_fig, use_container_width=True)


            with tab_history:
                history = list(reversed(st.session_state.forecast_history))
                if not history:
                    st.info("ℹ️ No history yet. Forecasts will be saved here when you run them.")
                else:
                    for i, entry in enumerate(history, 1):
                        st.markdown(f"### Run {i} — {entry['timestamp']}")
                        kws = entry.get('keywords', [])
                        st.caption("Keywords: " + ", ".join(kws))
                        df_h = entry['df']
                        forecasts_h = entry['forecasts']

                        fig_h = go.Figure()
                        for kw in kws:
                            if kw in df_h.columns:
                                fig_h.add_trace(go.Scatter(x=df_h.index, y=df_h[kw], mode='lines', name=f"{kw} Hist"))
                            else:
                                series_h = df_h[df_h["keyword"] == kw]["yhat"]
                                fig_h.add_trace(go.Scatter(x=series_h.index, y=series_h.values, mode='lines', name=f"{kw} Hist"))

                            fc_h = forecasts_h.get(kw)
                            if isinstance(fc_h, pd.DataFrame) and 'ds' in fc_h.columns and 'yhat' in fc_h.columns:
                                fig_h.add_trace(go.Scatter(x=fc_h['ds'], y=fc_h['yhat'], mode='lines', name=f"{kw} Fcst"))

                        fig_h.update_layout(title="Historical & Forecast (snapshot)")
                        st.plotly_chart(fig_h, use_container_width=True, key=f"history_chart_{i}")
                        st.markdown("---")


# -------------------- FEEDBACK --------------------
    elif choice == "Feedback":
        st.subheader("💬 Feedback")
        feedback_msg = st.text_area("Enter your feedback")
        if st.button("Submit Feedback"):
            if feedback_msg.strip():
                save_feedback(st.session_state.username, feedback_msg)
                st.success("✅ Feedback submitted! Thank you.")
            else:
                st.warning("Feedback cannot be empty.")

# -------------------- HELP --------------------
    elif choice == "Help":
        st.title("ℹ️ Help & User Guide")
        st.markdown("""
        **Quick Start**
        1. Register and login.
        2. Go to **Forecasting**: choose Google Trends or upload CSV.
        3. Choose timeframe / region / horizon and click **Fetch + Forecast**.
        4. Download CSV or PDF summary from the Downloads section.

        **Notes**
        - Prophet may take a few seconds to train. If not installed, LR is used as fallback.
        - Google Trends may return rate-limit errors (429) if queried rapidly.
        - PDF export requires `reportlab` (install with `pip install reportlab`).
        """)

# -------------------- ADMIN PANEL --------------------
    elif choice == "Admin Panel":
        if st.session_state.role != "admin":
            st.error("⛔ Access denied. Admins only.")
        else:
            st.title("🛠️ Admin Panel")
            
            # Use columns for better layout
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📋 User Feedback")
                conn = get_db_connection()
                feedback_df = pd.read_sql_query("SELECT * FROM feedback ORDER BY timestamp DESC", conn)
                conn.close()
                st.dataframe(feedback_df)
                
            with col2:
                st.subheader("👥 User Management")
                conn = get_db_connection()
                users_df = pd.read_sql_query("SELECT username, email, role FROM users", conn)
                conn.close()
                st.dataframe(users_df, use_container_width=True)

                st.markdown("---")
                
                # Role management form
                with st.form(key="role_form"):
                    st.subheader("Change User Role")
                    target_user = st.text_input("Enter username to change role")
                    action = st.radio("Select action", ["Promote to Admin", "Demote to User"])
                    submit_button = st.form_submit_button("Submit")

                    if submit_button:
                        if target_user == st.session_state.username:
                            st.warning("You cannot change your own role.")
                        elif not target_user:
                            st.warning("Please enter a username.")
                        else:
                            if action == "Promote to Admin":
                                update_user_role(target_user, 'admin')
                                st.success(f"✅ User '{target_user}' has been promoted to Admin.")
                            elif action == "Demote to User":
                                update_user_role(target_user, 'user')
                                st.success(f"✅ User '{target_user}' has been demoted to User.")
                            st.rerun()
