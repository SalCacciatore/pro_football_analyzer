"""
Season handling for app.py:

  * YEARS keeps 2023 and adds 2026
  * this_season -> 2026
  * last_szn: for weeks 1, 2 and 3 of 2026, use 2025 data.
    Otherwise use the season of the game being reviewed.

Run this from the same folder as app.py:

    python patch_season3.py

Prints an ok/MISS line per pattern and writes a backup to app.py.season3.bak.
A MISS on a single line is fine if you already applied that one -- the
script only fails if nothing at all matched.
"""

import os
import shutil
import sys

TARGET = "app.py"
BACKUP = "app.py.season3.bak"

# The replacement block. game_id looks like 2026_01_DAL_PHI, so the week is
# the second underscore-delimited field.
BASELINE_NEW = (
    "week_int = int(game_id.split(\"_\")[1])\n"
    "    if year_int == 2026 and week_int in (1, 2, 3):\n"
    "        baseline_szn = 2025\n"
    "    else:\n"
    "        baseline_szn = year_int\n"
    "    last_szn = data[data['season']==baseline_szn]"
)

REPLACEMENTS = [
    ("YEARS = [2023,2024,2025]", "YEARS = [2023,2024,2025,2026]"),
    ("YEARS = [2024,2025,2026]", "YEARS = [2023,2024,2025,2026]"),
    ("this_season = 2025", "this_season = 2026"),
    # Catch whichever form of the baseline line is currently in the file.
    ("last_szn = data[data['season']==2025]", BASELINE_NEW),
    ("last_szn = data[data['season']==year_int - 1]", BASELINE_NEW),
    ("last_szn = data[data['season']==year_int]", BASELINE_NEW),
]


def main():
    if not os.path.exists(TARGET):
        print("ERROR: %s not found in %s" % (TARGET, os.getcwd()))
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
        print("\nNothing matched. No changes written.")
        sys.exit(1)

    shutil.copyfile(TARGET, BACKUP)
    with open(TARGET, "w", encoding="utf-8") as fh:
        fh.write(src)
    print("\nPatched %d occurrence(s). Backup saved to %s" % (total, BACKUP))
    print("\nSanity check -- season lines now live:")
    for line in src.splitlines():
        if ("YEARS =" in line or "this_season =" in line
                or "last_szn =" in line or "baseline_szn" in line
                or "week_int" in line):
            print("   " + line.rstrip())


if __name__ == "__main__":
    main()
