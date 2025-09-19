

📊 Retail Trends & Forecasting App

📌 Overview

The **Retail Trends & Forecasting App** is a **Streamlit-based interactive dashboard** that allows users to analyze **Google Trends data** or upload **custom retail sales datasets** to forecast demand using **Facebook Prophet** or **Linear Regression** models.
It includes **role-based authentication, feedback management, forecasting history, PDF/CSV export, and an admin panel** for user and feedback management.

 🚀 Features

 🔒 Authentication & Roles
* **User registration & login** with secure password hashing (SHA-256 + salt).
* **Manual password reset** (set your own new password).
* **Role-based access control**:
* **User** → Forecasting, dashboards, feedback.
* **Admin** → Manage roles & view feedback.

🌐 Data Sources
* **Google Trends API (PyTrends)** → Analyze search interest for keywords across different regions.
* **CSV Upload** → Upload retail sales data (`date, sales`) for forecasting.

 📈 Forecasting Models
* **Prophet Forecasting** (preferred, if installed).
* **Linear Regression fallback** (if Prophet unavailable).
* Forecast horizon: **7–90 days**.
* Automatic comparison of **historical vs forecast**.

📊 Analytics & Visualization
* Interactive **Plotly charts** for historical and forecasted data.
* **Analytics Dashboard**:
* Latest forecast with metrics (total sales, average sales, horizon).
* Forecast history with snapshot comparison.
* **Forecast Explanations**: Prophet component breakdown + textual insights.

📥 Data Export
* Export **historical and forecast data** as **CSV/ZIP**.
* Generate **PDF reports** (if `reportlab` installed).

📬 Feedback & Help
* Users can **submit feedback**.
* **Admin panel** to view and manage feedback.
* Built-in **Help section** for guidance.

🖥️ UI & Themes
* **Modern responsive UI** with custom CSS.
* **Dark/Light mode toggle**.
* Sidebar navigation for easy access.


📂 Project Structure

```
Retail-Forecasting-App/
│── app.py                  # Main Streamlit app
│── users.db                # SQLite database (created on first run)
│── requirements.txt        # Dependencies
│── README.md               # Project documentation
│── /data                   # (Optional) Example CSV datasets
```

 ⚙️ Installation

1. **Clone repository**

```bash
git clone https://github.com/your-repo/Retail-Forecasting-App.git
cd Retail-Forecasting-App
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Run app**

```bash
streamlit run app.py
```

📦 Requirements

Main dependencies:

* `streamlit` – interactive UI
* `pandas`, `numpy` – data processing
* `pytrends` – Google Trends API
* `plotly` – interactive charts
* `prophet` – forecasting (optional, else LR fallback)
* `scikit-learn` – linear regression fallback
* `reportlab` – PDF export (optional)
* `sqlite3` – user & feedback database

Install via:

```bash
pip install streamlit pandas numpy pytrends plotly prophet scikit-learn reportlab
```

🔑 Usage Guide
1. **Register/Login** → Create an account or login.
2. **Forecasting** →
 * Enter keywords for Google Trends OR upload CSV.
 * Choose **region, timeframe, forecast horizon**.
 * Click **Fetch + Forecast**.
3. **Visualize** → Explore results in interactive charts.
4. **Download** → Export CSV/ZIP/PDF reports.
5. **Feedback** → Submit app suggestions.
6. **Admin Panel** → Manage users & roles (Admin only).


📊 Example Forecast Flow

1. User enters: `smartphone, laptop`
2. Selects region: `India (IN)`
3. Sets horizon: `30 days`
4. App fetches **Google Trends data** → Fits Prophet → Generates forecast
5. User downloads forecast as **CSV & PDF**

🔮 Future Enhancements

* AI-based advanced forecasting (LSTM, Transformer).
* Email notifications for scheduled forecasts.
* Multi-language support.
* Cloud deployment (Streamlit Cloud, Heroku).

---

👨‍💻 Author

Developed as part of an **internship at CodTech IT Solutions** in the **Data Science domain**.

