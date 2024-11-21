for depth_factor in {0.5,0.75,1.5,2,3}; do
    bash scripts/$TASK.sh --depth $depth_factor $ARGS_FOR_REPEATED --seed -1 "$@"
done
