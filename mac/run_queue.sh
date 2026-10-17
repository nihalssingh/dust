#!/bin/bash
# Run jobs from a queue file one at a time, so runs never share the GPU.
# Each line is "name<TAB>command"; the first line is removed just before it starts, so lines can be appended
# (or deleted) while the queue runs. Logs go to LOG_DIR/<name>.log and a summary to LOG_DIR/queue.log.
# Usage: caffeinate -ims bash mac/run_queue.sh QUEUE_FILE LOG_DIR
queue=$1
logs=$2
mkdir -p "$logs"
while true; do
    line=$(head -n 1 "$queue")
    [ -z "$line" ] && break
    tail -n +2 "$queue" > "$queue.tmp" && mv "$queue.tmp" "$queue"
    name=${line%%$'\t'*}
    cmd=${line#*$'\t'}
    echo "$(date '+%F %T') START $name" >> "$logs/queue.log"
    start=$(date +%s)
    bash -c "$cmd" > "$logs/$name.log" 2>&1
    status=$?
    echo "$(date '+%F %T') END $name status=$status minutes=$(( ($(date +%s) - start) / 60 ))" >> "$logs/queue.log"
done
echo "$(date '+%F %T') QUEUE EMPTY" >> "$logs/queue.log"
