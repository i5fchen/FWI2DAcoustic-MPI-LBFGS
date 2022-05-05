void gaussian_win(size_t n, float delta, float *win);

void lcc_diw_msf(float **obs, float **syn, float *sy_xy, float offset_range[], float eps, int nt, int nr, int nl, int nw, float alpha_win, float sim_barrier, float lag_barrier, float *msf, float **res);
