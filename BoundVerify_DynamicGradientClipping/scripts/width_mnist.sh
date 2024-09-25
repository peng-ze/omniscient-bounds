for width in {64,128,256,512,1024}; do
    bash scripts/$TASK.sh  --width $width --seed -1 $ARGS_FOR_REPEATED "$@"
done
