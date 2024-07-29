
bash scripts/$TASK.sh --seed -1 "$@"
bash scripts/wd.sh "$ARGS_FOR_REPEATED" "$@"; bash scripts/width.sh "$ARGS_FOR_REPEATED" "$@"; bash scripts/label_corruption.sh "$ARGS_FOR_REPEATED"  "$@"; bash scripts/lrbs.sh "$ARGS_FOR_REPEATED" "$@"

# $ARGS_FOR_REPEATED is arguments for repeated experiments, which controls the costs of them, for example, the frequency of bound estimation.