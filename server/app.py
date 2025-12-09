from flask import Flask, jsonify
from apscheduler.schedulers.background import BackgroundScheduler
import subprocess
import os

app = Flask(__name__)
scheduler = BackgroundScheduler()

SCRAPER_PATH = os.path.join(os.path.dirname(__file__), "..", "pilots", "scraper.py")

def run_scraper():
    print(" Running scraper...")
    subprocess.run(["python", SCRAPER_PATH])

#  Run every 5 minutes (change as needed)
scheduler.add_job(run_scraper, 'interval', minutes=5)
scheduler.start()

@app.route("/")
def home():
    return " Instagram Scraper Server is Running!"

@app.route("/refresh")
def refresh():
    run_scraper()
    return jsonify({"status": "scraper manually triggered"})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
