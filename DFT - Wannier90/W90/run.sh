mpirun -np 8 pw.x -in mos2.scf > scf.out
mpirun -np 8 pw.x -in mos2.nscf > nscf.out
wannier90.x -pp mos2
mpirun -np 8 pw2wannier90.x -in mos2.pw2wan > pw2wan.out
wannier90.x mos2

# If you want to get the density of state, do the postw90
postw90.x mos2


