from flask import Flask, render_template, request
import pandas as pd
import os

app = Flask(__name__)

# Attendance file path
attendance_file = "attendance.csv"

@app.route('/', methods=['GET'])
def index():
    """Renders the dashboard page with attendance data."""
    selected_date = request.args.get('date', '')  # Get date from query parameter
    attendees = []

    if os.path.exists(attendance_file):
        df = pd.read_csv(attendance_file)

        # Convert "Time" column to datetime for filtering
        df["Time"] = pd.to_datetime(df["Time"], format="%Y-%m-%d %H:%M:%S")

        # Filter by date if selected
        if selected_date:
            df = df[df["Time"].dt.strftime("%Y-%m-%d") == selected_date]

        attendees = df.to_dict(orient="records")  # Convert to list of dicts

    return render_template("dashboard.html", attendees=attendees, selected_date=selected_date)

if __name__ == '__main__':
    app.run(debug=True)

