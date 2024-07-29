bash scripts/$TASK.sh --seed -1 "$@"
bash scripts/wd.sh "$@"; bash scripts/width.sh "$@"; bash scripts/label_corruption.sh "$@"; bash scripts/lrbs.sh "$@"
