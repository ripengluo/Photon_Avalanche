python plot_n2_snapshot.py r50-8p0-EM1p0/power_09_17018.1/   --animate     --step-interval 1  --fps 200     --video-duration-min 320 --state-visual-mode n2-n4-n5plus #--show-n3
ffmpeg -y -f concat -safe 0 -i r50-8p0-EM1p0/power_09_17018.1/17k_xy_n2n4n5plus_concat_list.txt -c copy r50-8p0-EM1p0/power_09_17018.1/17k_xy_n2n4n5plus.mp4
