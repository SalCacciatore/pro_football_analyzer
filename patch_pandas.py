"""
Fixes the pandas 3.x positional-indexing breakage in app.py.

In pandas 3.0, an integer key passed to Series[...] is always treated as a
LABEL, not a position. The pattern `df.loc[df.index[0]][i]` therefore raises
KeyError: 0 because the resulting Series is indexed by column names.

Run this from the same folder as app.py:

    python patch_pandas.py

It prints how many of each pattern it found before changing anything, and
writes a backup to app.py.bak.
"""

import os
import shutil
import sys

TARGET = "app.py"
BACKUP = "app.py.bak"

REPLACEMENTS = [
    ("host_sr.loc[host_sr.index[0]][i]", "host_sr.iloc[0].iloc[i]"),
    ("visitor_sr.loc[visitor_sr.index[0]][i]", "visitor_sr.iloc[0].iloc[i]"),
    ("offense_table.loc[offense_table.index[0]][i]", "offense_table.iloc[0].iloc[i]"),
]


def main():
    if not os.path.exists(TARGET):
        print("ERROR: %s not found in %s" % (TARGET, os.getcwd()))
        print("cd into your pro_football_analyzer folder and run this again.")
        sys.exit(1)

    with open(TARGET, "r", encoding="utf-8") as fh:
        src = fh.read()

    total = 0
    for old, new in REPLACEMENTS:
        count = src.count(old)
        total += count
        status = "ok " if count else "MISS"
        print("[%s] %dx  %s" % (status, count, old))
        if count:
            src = src.replace(old, new)

    if total == 0:
        print("\nNothing matched. app.py may already be patched, or the")
        print("lines differ from what was expected. No changes written.")
        sys.exit(1)

    shutil.copyfile(TARGET, BACKUP)
    with open(TARGET, "w", encoding="utf-8") as fh:
        fh.write(src)
    print("\nPatched %d occurrence(s). Backup saved to %s" % (total, BACKUP))

    # Pin pandas so Streamlit Cloud stops pulling 3.x
    req = "requirements.txt"
    if os.path.exists(req):
        with open(req, "r", encoding="utf-8") as fh:
            lines = fh.read()
        if "pandas" in lines:
            print("\nNOTE: requirements.txt already mentions pandas.")
            print("Check that it reads 'pandas<3' and edit if not.")
        else:
            with open(req, "a", encoding="utf-8") as fh:
                if not lines.endswith("\n"):
                    fh.write("\n")
                fh.write("pandas<3\n")
            print("Added 'pandas<3' to requirements.txt")
    else:
        print("\nWARNING: no requirements.txt found here.")


if __name__ == "__main__":
    main()
