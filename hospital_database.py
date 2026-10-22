
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "hospital_test.db"

DEFAULT_DEVICES = [
    ("Remote Surgery", 1, 12.0, 5),
    ("Patient Monitoring", 10, 1.0, 4),
    ("Medical Telemetry", 8, 1.0, 3),
    ("Doctor Communication", 4, 2.0, 2),
    ("Regular Internet", 20, 1.0, 1),
]


def create_hospital_database():
    with sqlite3.connect(DB_PATH, timeout=10) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS hospital_devices (
                application TEXT PRIMARY KEY,
                active_devices INTEGER NOT NULL,
                bandwidth_per_device REAL NOT NULL,
                priority_weight INTEGER NOT NULL
            )
        """)

        conn.executemany("""
            INSERT OR IGNORE INTO hospital_devices
            (application, active_devices,
             bandwidth_per_device, priority_weight)
            VALUES (?, ?, ?, ?)
        """, DEFAULT_DEVICES)


def get_hospital_devices():
    create_hospital_database()

    with sqlite3.connect(DB_PATH, timeout=10) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT application, active_devices,
                   bandwidth_per_device, priority_weight
            FROM hospital_devices
            ORDER BY priority_weight DESC
        """).fetchall()

    return [dict(row) for row in rows]


def update_device_count(application, count):
    count = int(count)

    if count < 0:
        raise ValueError("Device count cannot be negative")

    create_hospital_database()

    with sqlite3.connect(DB_PATH, timeout=10) as conn:
        result = conn.execute("""
            UPDATE hospital_devices
            SET active_devices = ?
            WHERE application = ?
        """, (count, application))

        if result.rowcount == 0:
            raise ValueError("Unknown application")


def get_application_demands():
    return {
        row["application"]:
            row["active_devices"] * row["bandwidth_per_device"]
        for row in get_hospital_devices()
    }


def get_total_demand():
    return sum(get_application_demands().values())
