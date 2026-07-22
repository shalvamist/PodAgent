import sqlite3
conn = sqlite3.connect("data/podagent.db")
cur = conn.cursor()
cur.execute("SELECT id, video_id, title, duration FROM podcasts ORDER BY CAST(duration AS REAL) DESC LIMIT 10")
for row in cur.fetchall():
    dur = float(row[3]) if row[3] else 0
    print(f"  {row[0]}: {row[1]} | {dur:.0f}s ({dur/60:.0f}min) | {row[2][:50]}")
conn.close()
