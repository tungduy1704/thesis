set terminal pngcairo enhanced size 1920,1080 font "Arial,24"
set output 'bandstructure.png'

set ylabel "Energy (eV)"
set grid
set key bottom center
set key at 0.2, 0.8
set xtics ("-M" -1, "-K" -2.0/3, "Γ" 0, "K" 2.0/3, "M" 1)

set arrow from -1, graph 0 to -1, graph 1 nohead lt 1 lw 2
set arrow from -2.0/3, graph 0 to -2.0/3, graph 1 nohead lt 1 lw 2
set arrow from 0, graph 0 to 0, graph 1 nohead lt 1 lw 2
set arrow from 2.0/3, graph 0 to 2.0/3, graph 1 nohead lt 1 lw 2
set arrow from 1, graph 0 to 1, graph 1 nohead lt 1 lw 2

set datafile separator whitespace

plot \
  "bandstr1 cho kp.txt" using ($1 + 2.0/3):2 with lines lc rgb "red" title "k.p", \
  "bandstr1 cho kp.txt" using ($1 + 2.0/3):3 with lines lc rgb "red" notitle, \
  "bandstr1_tb_shifted.txt" using 1:2 with lines lc rgb "blue" title "NN-TB", \
  "bandstr1_tb_shifted.txt" using 1:3 with lines lc rgb "blue" notitle, \
  "bandstr1_tb_shifted.txt" using 1:4 with lines lc rgb "blue" notitle, \
  "bandstr1_3nn_shifted.txt" using 1:2 with lines lc rgb "green" title "3rd NN", \
  "bandstr1_3nn_shifted.txt" using 1:3 with lines lc rgb "green" notitle, \
  "bandstr1_3nn_shifted.txt" using 1:4 with lines lc rgb "green" notitle, \
  "eigenvalues_map294_shifted.txt" using 1:9 with lines lc rgb "black" title "DFT", \
  "eigenvalues_map294_shifted.txt" using 1:10 with lines lc rgb "black" notitle, \
  "eigenvalues_map294_shifted.txt" using 1:11 with lines lc rgb "black" notitle, \


# set output 'kp_comparison.png'

# set multiplot layout 1,2 title ""

# # ===== Subplot (a): Valence band =====
# set title "(a)"
# set xlabel "k_x"
# set ylabel "Energy (eV)"
# set xtics ("Γ" -0.1, "K" 0, "M" 0.1)
# set key top right
# plot \
#   "conduction_valence_all.txt" using 1:2 with lines lt 2 lw 2 dashtype (3,3) lc rgb "blue" title "k.p(1)", \
#   "" using 1:3 with lines lt 1 lw 2 lc rgb "red" title "k.p(2)", \
#   "" using 1:4 with lines lt 1 lw 2 lc rgb "black" title "k.p(3)"

# # ===== Subplot (b): Conduction band =====
# set title "(b)"
# set xlabel "k_x"
# set ylabel "Energy (eV)"
# set xtics ("Γ" -0.1, "K" 0, "M" 0.1)
# set key top right
# plot \
#   "conduction_valence_all.txt" using 1:5 with lines lt 2 lw 2 dashtype (3,3) lc rgb "blue" title "k.p(1)", \
#   "" using 1:6 with lines lt 1 lw 2 lc rgb "red" title "k.p(2)", \
#   "" using 1:7 with lines lt 1 lw 2 lc rgb "black" title "k.p(3)"

# unset multiplot
