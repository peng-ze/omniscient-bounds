bash scripts/$TASK.sh --seed -1 "$@"
bash scripts/lrbs.sh $ARGS_FOR_REPEATED "$@"
bash scripts/label_corruption.sh $ARGS_FOR_REPEATED  "$@"; 
bash scripts/wd.sh $ARGS_FOR_REPEATED "$@"; bash scripts/width.sh $ARGS_FOR_REPEATED "$@"; 