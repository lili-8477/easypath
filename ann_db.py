"""SQLite store for pathologist decisions. Reviews are append-only, so the table is the audit trail;
the current decision for an ROI is its most recent review."""
import sqlite3

from ann_common import DB

SCHEMA = """
CREATE TABLE IF NOT EXISTS manual_rois (
  roi_id TEXT PRIMARY KEY, slide_id TEXT, x INT, y INT, w INT, h INT, created_by TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS reviews (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  roi_id TEXT, slide_id TEXT, reviewer TEXT, ts TEXT,
  decision TEXT,            -- accept | change | not_tumor | artifact | second_review
  tumor TEXT, pattern TEXT, necrosis TEXT, artifact INT, second_review INT, comment TEXT,
  model_version TEXT, prediction TEXT, geometry TEXT);
CREATE INDEX IF NOT EXISTS reviews_roi ON reviews(roi_id);
CREATE TABLE IF NOT EXISTS cluster_reviews (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  cluster INT, reviewer TEXT, ts TEXT,
  label TEXT, pattern TEXT, purity TEXT, batch_artifact INT, comment TEXT, clustering TEXT);
CREATE TABLE IF NOT EXISTS slide_reviews (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  slide_id TEXT, reviewer TEXT, ts TEXT,
  subtype TEXT,             -- monophasic | biphasic | poorly_differentiated | not_ss | cannot_assess
  patterns TEXT,            -- comma-separated: glandular, corded
  comment TEXT, predicted_subtype TEXT, clustering TEXT);
CREATE TABLE IF NOT EXISTS slide_seg_reviews (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  slide_id TEXT, segment INT, reviewer TEXT, ts TEXT, label TEXT, predicted TEXT, clustering TEXT);
CREATE TABLE IF NOT EXISTS merge_states (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  clustering TEXT, scope TEXT,   -- scope: '*' = all slides, else a slide_id
  groups TEXT,                   -- JSON list of class lists, the full merge state of this scope after the change
  action TEXT, classes TEXT, reviewer TEXT, ts TEXT);
CREATE TABLE IF NOT EXISTS seg_edits (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  clustering TEXT, slide_id TEXT, reviewer TEXT, ts TEXT,
  action TEXT,              -- paint | stroke | reshape | undo | redo
  segment INT,              -- class painted; 255 = erased to background
  target INT,               -- undo / redo: the id of the edit concerned
  geometry TEXT);           -- JSON, level-0 slide pixels
CREATE INDEX IF NOT EXISTS seg_edits_slide ON seg_edits(clustering, slide_id);
CREATE TABLE IF NOT EXISTS segment_comments (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  clustering TEXT, slide_id TEXT, segment INT, reviewer TEXT, ts TEXT, comment TEXT);
"""
LATEST = """SELECT r.* FROM reviews r JOIN (SELECT roi_id, MAX(id) AS mid FROM reviews GROUP BY roi_id) l
            ON r.id = l.mid"""


def connect():
    con = sqlite3.connect(DB, timeout=30)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


def latest_reviews(con):
    import pandas as pd
    return pd.read_sql_query(LATEST, con)


def latest_cluster_reviews(con, clustering):
    import pandas as pd
    return pd.read_sql_query("""SELECT r.* FROM cluster_reviews r JOIN (SELECT cluster, MAX(id) AS mid FROM cluster_reviews
                                WHERE clustering = ? GROUP BY cluster) l ON r.id = l.mid""", con, params=(clustering,))


def manual_rois(con):
    import pandas as pd
    return pd.read_sql_query("SELECT * FROM manual_rois", con)


def latest_slide_reviews(con):
    import pandas as pd
    return pd.read_sql_query("""SELECT r.* FROM slide_reviews r JOIN (SELECT slide_id, MAX(id) AS mid FROM slide_reviews
                                GROUP BY slide_id) l ON r.id = l.mid""", con)


def latest_slide_seg_reviews(con, clustering):
    """Per-slide segment labels; they override the class-level cluster label on that slide."""
    import pandas as pd
    return pd.read_sql_query("""SELECT r.* FROM slide_seg_reviews r JOIN (SELECT slide_id, segment, MAX(id) AS mid
                                FROM slide_seg_reviews WHERE clustering = ? GROUP BY slide_id, segment) l
                                ON r.id = l.mid""", con, params=(clustering,))


def latest_merges(con, clustering, scope):
    """Reviewer merges of segment classes in one scope ('*' or a slide id): list of class lists."""
    import json
    r = con.execute("SELECT groups FROM merge_states WHERE clustering = ? AND scope = ? ORDER BY id DESC LIMIT 1",
                    (clustering, scope)).fetchone()
    return json.loads(r["groups"]) if r else []


def seg_edit_log(con, clustering, slide_id):
    """Replay a slide's edit log: (edits in force, oldest first; undone edits that can be redone, oldest first)."""
    active, undone = [], []
    for r in con.execute("SELECT * FROM seg_edits WHERE clustering = ? AND slide_id = ? ORDER BY id", (clustering, slide_id)):
        if r["action"] == "undo":
            undone += [e for e in active if e["id"] == r["target"]]
            active = [e for e in active if e["id"] != r["target"]]
        elif r["action"] == "redo":
            active += [e for e in undone if e["id"] == r["target"]]
            undone = [e for e in undone if e["id"] != r["target"]]
        else:
            active.append(dict(r))
            undone = []
    return active, undone
