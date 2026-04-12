// -------------------- QFn -----------------------------------

typedef struct {
    SmoothParametricFn *approx_arch;
    SAFeatureMap *sa_feamap;
    Vec theta;
    Vec fea_SA;
} QFn;

void make_QFn(QFn *qfn, SAFeatureMap *sa_feamap,
              SmoothParametricFn *approx_arch) {
    make_vec(&qfn->fea_SA, 0, 0);
    qfn->sa_feamap = sa_feamap;
    qfn->approx_arch = approx_arch;
    sa_feamap->VT_ACCESS prealloc_out_vec(sa_feamap, &qfn->fea_SA);
    approx_arch->VT_ACCESS alloc_init_theta(approx_arch, &qfn->theta);
}

extern inline
void free_QFn(QFn *qfn) {
    free_vec(&qfn->fea_SA);
    free_vec(&qfn->theta);
}

real Qtheta(QFn *qfn, Elem *S, Elem *A) {
    qfn->sa_feamap->VT_ACCESS
        map_writo_vec(qfn->sa_feamap, S, A, &qfn->fea_SA);
    return qfn->approx_arch->VT_ACCESS
        f(qfn, &qfn->theta, &qfn->fea_SA);
}

void dQtheta_dtheta_writo_vec(QFn *qfn, Elem *S, Elem *A, Vec *dest) {
    qfn->sa_feamap->VT_ACCESS
        map_writo_vec(qfn->sa_feamap, S, A, &qfn->fea_SA);
    qfn->approx_arch->VT_ACCESS
        deriv_writo_vec(qfn, &qfn->theta, &qfn->fea_SA, dest);
}

void dQtheta_dtheta_addto_dvec(QFn *qfn, Elem *S, Elem *A, Vec *dest) {
    qfn->sa_feamap->VT_ACCESS
        map_writo_vec(qfn->sa_feamap, S, A, &qfn->fea_SA);
    qfn->approx_arch->VT_ACCESS
        deriv_addto_dvec(qfn, &qfn->theta, &qfn->fea_SA, dest);
}

void scaled_dQtheta_dtheta_addto_dvec(QFn *qfn, Elem *S, Elem *A, real a,
                                      Vec *dest) {
    qfn->sa_feamap->VT_ACCESS
        map_writo_vec(qfn->sa_feamap, S, A, &qfn->fea_SA);
    qfn->approx_arch->VT_ACCESS
        scaled_deriv_addto_dvec(qfn, &qfn->theta, &qfn->fea_SA, a, dest);
}
