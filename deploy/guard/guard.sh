#!/bin/bash
# auto-trader repo guard
#
# The working tree has been silently wiped three times (May 18, Jul 17, Jul 21
# 2026) — tracked files deleted while untracked data survived, consistent with
# an editor checkpoint restore. This script restores any deleted tracked files
# from git HEAD and raises a macOS notification.
#
# It is INSTALLED OUTSIDE THE REPO (so it survives wipes) at:
#   ~/Library/Application Support/autotrader/guard.sh
# and run every 15 minutes by ~/Library/LaunchAgents/com.autotrader.guard.plist
# This copy in deploy/guard/ is the source of truth; reinstall with:
#   cp deploy/guard/guard.sh "$HOME/Library/Application Support/autotrader/guard.sh"
#   cp deploy/launchd/com.autotrader.guard.plist ~/Library/LaunchAgents/
#   launchctl load ~/Library/LaunchAgents/com.autotrader.guard.plist

REPO="/Users/tim/Development/auto-trader"
GIT=/usr/bin/git

cd "$REPO" || exit 0

deleted=$($GIT ls-files --deleted)
[ -z "$deleted" ] && exit 0

n=$(printf '%s\n' "$deleted" | wc -l | tr -d ' ')
$GIT ls-files --deleted -z | xargs -0 $GIT restore -- 2>/dev/null

mkdir -p "$REPO/logs"
echo "$(date -u +%FT%TZ) guard restored $n deleted tracked files" >> "$REPO/logs/guard.log"
/usr/bin/osascript -e "display notification \"Restored $n deleted repo files (checkpoint wipe)\" with title \"auto-trader guard\"" 2>/dev/null

exit 0
