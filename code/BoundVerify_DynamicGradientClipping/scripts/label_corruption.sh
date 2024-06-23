for corruption in {0.25,0.50,0.75}; do
    python main.py --arch fc1 --epochs 500 --batch-size 60 --learning-rate 0.01 --dataset mnist --data-path data --width 512 --label-corrupt-prob $corruption --early_stop False --fixinit False --seed 1 --k 6 --test-freq 20
done
