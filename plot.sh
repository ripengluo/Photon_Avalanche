python plot_n2_snapshot.py r50-8p0-baseline-50M/power_08_15504.8/   --animate  --step-interval 1  --fps 300  --video-duration-min 180 --state-visual-mode n2-n4-n5plus #--show-n3
ffmpeg -y -f concat -safe 0 -i r50-8p0-baseline-50M/power_08_15504.8/15k_xy_n2n4n5plus_concat_list.txt -c copy r50-8p0-baseline-50M/power_08_15504.8/15k_xy_n2n4n5plus.mp4
