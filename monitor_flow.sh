#!/bin/bash
# Monitor node flow for a specific session

SESSION_ID=$1

if [ -z "$SESSION_ID" ]; then
    echo "Usage: ./monitor_flow.sh <session_id>"
    echo ""
    echo "Recent sessions:"
    sqlite3 -header -column data/telemetry.db "
        SELECT session_id, SUBSTR(user_text, 1, 40) as query
        FROM requests 
        ORDER BY timestamp DESC 
        LIMIT 5;
    "
    exit 1
fi

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║           NODE FLOW TRACE: $SESSION_ID"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Node timeline
sqlite3 -header -box data/telemetry.db "
SELECT 
  ROW_NUMBER() OVER (ORDER BY timestamp) as step,
  SUBSTR(timestamp, 12, 12) as time,
  node_name,
  event_type,
  CASE 
    WHEN event_type LIKE '%success%' OR event_type LIKE '%generated%' THEN '✅'
    WHEN event_type LIKE '%failure%' OR event_type LIKE '%error%' THEN '❌'
    WHEN event_type LIKE '%entry%' OR event_type LIKE '%attempt%' THEN '▶️'
    ELSE '📝'
  END as status
FROM logs 
WHERE session_id = '$SESSION_ID' 
ORDER BY timestamp;
"

echo ""
echo "Request Summary:"
echo "───────────────────────────────────────────────────────────────"

sqlite3 -json data/telemetry.db "
SELECT 
  intent,
  ROUND(confidence, 3) as confidence,
  duration_ms,
  user_text as query
FROM requests 
WHERE session_id = '$SESSION_ID';
" | python3 -m json.tool

echo ""

