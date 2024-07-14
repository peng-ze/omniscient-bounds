for batch_size in {120,240,480,600}; do
    python main.py --arch fc1 --epochs 500 --batch-size $batch_size --learning-rate 0.01 --dataset mnist --data-path data --width 512 --label-corrupt-prob 0 --early_stop False --fixinit False --seed 1 --k 6 --test-freq 20 "$@"
done
