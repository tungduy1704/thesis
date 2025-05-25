set terminal pngcairo enhanced size 1920,1080
set output 'p_vs_kx thu gon.png'

set xlabel 'k_x (2π/a)'
set ylabel '|p|'
set key top left font ",20" spacing 1.5 width 2

set grid
set xrange [0.566:0.766]

plot \
    'nntb_momentum.txt'     using 1:7 with lines linestyle 1 title 'NN tight-binding', \
    'kp_momentum.txt'     using ($1 + 2.0/3):7 with lines linestyle 2 title 'k·p', \
    'p_cv_map294.txt'         using 1:7 with lines linestyle 3 title 'DFT 294', \
    'thirdnn_momentum.txt'    using 1:7 with lines linestyle 4 title '3NN tight-binding', \
    #'p_cv_map.csv'            using 1:7 with lines linestyle 5 title 'DFT 144'

set style line 1 lt 1 lc rgb '#1f77b4' lw 2  # blue
set style line 2 lt 1 lc rgb '#ff7f0e' lw 2  # orange
set style line 3 lt 1 lc rgb '#2ca02c' lw 2  # green
set style line 4 lt 2 lc rgb '#d62728' lw 2  # red, dashline
set style line 5 lt 2 lc rgb '#9467bd' lw 2  # purple, dashline
