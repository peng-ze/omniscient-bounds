for width in {64,128,256,512,1024}; do
    python main.py --arch fc1 --epochs 500 --batch-size 60 --learning-rate 0.01 --dataset mnist --data-path data --width $width --label-corrupt-prob 0 --early_stop False --fixinit False --k 6 --test-freq 20 "$@"
done
