#!/bin/bash
# Start the review app on a free port, hit the run and segment endpoints for every run, then stop it.
module use $HOME/MyModules; module load miniconda3/latest; source activate wsi_env
cd "$(dirname "$0")/../app"
PORT=8799
uvicorn server:app --host 127.0.0.1 --port $PORT --workers 1 > /tmp/smoke_$$.log 2>&1 &
PID=$!
trap "kill $PID" EXIT
sleep 12
T=${EASYPATH_TOKEN:-$(cat ${EASYPATH_TOKEN_FILE:-$HOME/.hss_annot_token})}
H="Authorization: Bearer $T"
get() { curl -s -H "$H" -b "token=$T" "http://127.0.0.1:$PORT/$1?token=$T"; }
post() { curl -s -H "$H" -b "token=$T" -H 'Content-Type: application/json' -d "$2" "http://127.0.0.1:$PORT/$1?token=$T"; }
get api/clusterings | python -c "import json,sys; d=json.load(sys.stdin); print('active', d['active']); [print(' ', r['dir'], r['k'], r['adaptive'], r['k_slide_median']) for r in d['runs']]"
for RUN in $(get api/clusterings | python -c "import json,sys; print(' '.join(r['dir'] for r in json.load(sys.stdin)['runs']))"); do
  post api/clusterings/active "{\"dir\":\"$RUN\"}"; echo
  get api/clusters | python -c "import json,sys; d=json.load(sys.stdin); print('$RUN clusters', len(d['clusters']), d['clustering']['id'])"
  n=0; for SID in $(get api/slides | python -c "import json,sys; print(' '.join(s['slide_id'] for s in json.load(sys.stdin)))"); do
    get api/slides/$SID/segments | python -c "
import json,sys
try:
    d=json.load(sys.stdin)
except Exception as e:
    print('$SID', 'ERROR'); sys.exit()
m=[s for s in d['segments'] if len(s['members'])>1]
print('$SID', len(d['segments']), 'segments', d['predicted']['call'], '| merged:', ' '.join(f\"S{s['segment']}={'+'.join(map(str,s['members']))}\" for s in m))" ; n=$((n+1))
  done
  echo "$RUN: $n slides checked"
  curl -s -o /dev/null -w "segmap %{http_code}\n" -H "$H" -b "token=$T" "http://127.0.0.1:$PORT/api/slides/$SID/clustermap/all.png?token=$T"
done
grep -iE "error|traceback" /tmp/smoke_$$.log | head
