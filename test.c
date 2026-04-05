// -------------------- TESTS ----------------------------------

RngState test_rngs = {.x={1,2,3,4}};

void vec_mat_tests() {
    print_sizeof(Vec);
    print_sizeof(Mat);
    print_sizeof(SparseEntry);
    Vec v1, v2, v3, u1, u2;
    Mat A, B, C;
    make_vec(&v1, 4, 0);
    SET_DV(v1, 22);
    scale_vec(&v1, 2);
    make_vec(&u1, 4, 2);
    push_entry_svec(&u1, 1, 11);
    scale_vec(&u1, -5);
    assert(v1.sparse_cap == 0);
    assert(u1.sparse_cap == 2);
    // v1 = (44,  44, 44, 44)
    // u1 = ( 0, -55,  0,  0)
    // should have big enough mantissa to compare to promoted ints
    assert(inner_vec(&v1, &u1) == -55 * 44);
    assert(inner_vec(&u1, &v1) == -55 * 44);
    assert(inner_vec(&v1, &v1) == 4*(44*44));
    assert(inner_vec(&u1, &u1) == 55*55);

    SET_DV(v1, 2)
    SET_ENTRIES_SV(u1, 5)
    assert(inner_vec(&v1, &u1) == 10);

    make_vec(&v2, 4, 0);
    make_vec(&u2, 4, 3);
    SET_DV(v2, -2)
    push_entry_svec(&u1, 3, 6);
    push_entry_svec(&u2, 1, 2);
    push_entry_svec(&u2, 2, 30);
    push_entry_svec(&u2, 3, 4);
    assert(inner_vec(&u1, &u2) == 5*2 + 6*4);
    assert(inner_vec(&u2, &u1) == 5*2 + 6*4);

    zero_vec(&v1);
    vec_addto_dvec(&u2, &v1);
    assert(!memcmp((real[]){0,2,30,4}, v1.data.dense, 4*sizeof(real)));
    scaled_vec_addto_dvec(&v1, 2, &v2);
    print_vec(&v2);
    assert(req_arr((real[]){-2,2,58,6}, v2.data.dense, 4, 1e-7));
    assert(inner_vec(&v1, &v2) == 2*2 + 30*58 + 4*6);

    make_mat(&A, 4, 4, 0);
    ZERO_DM(A)
    set_dmat_diag(&A, 1);
    assert(!memcmp((real[]){
        1,0,0,0,
        0,1,0,0,
        0,0,1,0,
        0,0,0,1
    }, A.data.dense, 16*sizeof(real)));
    scaled_self_outer_vec_addto_dmat(&v1, -1, &A);
    print_mat(&A);
    assert(req_arr((real[]){
        1, 0,  0,   0,
        0,-3, -60, -8,
        0,-60,-899,-120,
        0,-8, -120,-15
    }, A.data.dense, 16, 1e-7));

    ZERO_DM(A)
    set_dmat_diag(&A, 1);
    ZERO_SV(u1);
    push_entry_svec(&u1, 1, 3);
    push_entry_svec(&u1, 3, -2);
    scaled_self_outer_vec_addto_dmat(&u1, 2, &A);
    assert(!memcmp((real[]){
        1, 0, 0, 0,
        0, 19,0,-12,
        0, 0, 1, 0,
        0,-12,0, 9
    }, A.data.dense, 16*sizeof(real)));

    // outer tests
    make_vec(&v3, 16, 0);
    v2.dim = 3;
    u2.dim = 3;
    ZERO_SV(u1)
    ZERO_SV(u2)
    v1.data.dense[0] = 4;
    v1.data.dense[1] = 1;
    v1.data.dense[2] = 3;
    v1.data.dense[3] = 2;
    v2.data.dense[0] = 10;
    v2.data.dense[1] = 25;
    v2.data.dense[2] = 20;
    push_entry_svec(&u1, 0, 5);
    push_entry_svec(&u1, 2, 11);
    push_entry_svec(&u2, 1, 7);
    // v1 = 4,  1,  3,  2
    // v2 = 10, 25, 20
    // u1 = 5,  0,  11, 0
    // u2 = 0,  7,  0
    // dv dv^T to dmat
    A.nrows = 4; A.ncols = 3;
    outer_vec_writo_mat(&v1, &v2, &A);
    print_mat(&A);
    assert(req_arr((real[]){
        40,100,80,
        10,25, 20,
        30,75, 60,
        20,50, 40
    }, A.data.dense, 12, 1e-7));
    A.nrows = 3; A.ncols = 4;
    outer_vec_writo_mat(&v2, &v1, &A);
    assert(!memcmp((real[]){
        40, 10,30,20,
        100,25,75,50,
        80, 20,60,40
    }, A.data.dense, 12*sizeof(real)));
    // dv sv^T to dmat
    A.nrows = 4; A.ncols = 3;
    outer_vec_writo_mat(&v1, &u2, &A);
    assert(!memcmp((real[]){
        0,28,0,
        0,7, 0,
        0,21,0,
        0,14,0
    }, A.data.dense, 12*sizeof(real)));
    // sv dv^T to dmat
    A.nrows = 3; A.ncols = 3;
    outer_vec_writo_mat(&u2, &v2, &A);
    assert(!memcmp((real[]){
        0, 0,  0,
        70,175,140,
        0, 0,  0
    }, A.data.dense, 9*sizeof(real)));
    v3.dim = 9;
    flat_outer_vec_writo_vec(&u2, &v2, &v3);
    assert(!memcmp(A.data.dense, v3.data.dense, 9*sizeof(real)));
    // sv sv^T to dmat
    A.nrows = 4; A.ncols = 3;
    outer_vec_writo_mat(&u1, &u2, &A);
    assert(!memcmp((real[]){
        0,35, 0,
        0,0,  0,
        0,77, 0,
        0,0,  0
    }, A.data.dense, 12*sizeof(real)));
    assert(!dmat_is_symm(&A));
    v3.dim = 12;
    flat_outer_vec_writo_vec(&u1, &u2, &v3);
    assert(!memcmp(A.data.dense, v3.data.dense, 12*sizeof(real)));


    // outer smat dest tests
    // no need to test dv dv^T to smat
    make_mat(&B, 4, 4, 16);
    v3.sparse_cap = 8;

    // dv sv^T to dmat
    B.nrows = 4; B.ncols = 3;
    outer_vec_writo_mat(&v1, &u2, &B);
    assert(!memcmp((SparseEntry[]){
        {.idx=1, .val=28},
        {.idx=4, .val=7},
        {.idx=7, .val=21},
        {.idx=10, .val=14}
    }, B.data.sparse, 4*sizeof(SparseEntry)));
    v3.dim = 12; ZERO_SV(v3);
    flat_outer_vec_writo_vec(&v1, &u2, &v3);
    assert(!memcmp(B.data.sparse, v3.data.sparse, 4*sizeof(SparseEntry)));

    // sv dv^T to dmat
    B.nrows = 3; B.ncols = 3;
    ZERO_SM(B)
    outer_vec_writo_mat(&u2, &v2, &B);
    assert(!memcmp((SparseEntry[]){
        {.idx=3, .val=70},
        {.idx=4, .val=175},
        {.idx=5, .val=140}
    }, B.data.sparse, 3*sizeof(SparseEntry)));
    v3.dim = 9; ZERO_SV(v3);
    flat_outer_vec_writo_vec(&u2, &v2, &v3);
    assert(!memcmp(B.data.sparse, v3.data.sparse, 3*sizeof(SparseEntry)));

    // sv sv^T to dmat
    B.nrows = 4; B.ncols = 3;
    ZERO_SM(B)
    outer_vec_writo_mat(&u1, &u2, &B);
    assert(!memcmp((SparseEntry[]){
        {.idx=1, .val=35},
        {.idx=7, .val=77}
    }, B.data.sparse, 2*sizeof(SparseEntry)));
    v3.dim = 12; ZERO_SV(v3);
    flat_outer_vec_writo_vec(&u1, &u2, &v3);
    assert(!memcmp(B.data.sparse, v3.data.sparse, 2*sizeof(SparseEntry)));


    A.nrows = 4; A.ncols = 4;
    B.nrows = 4; B.ncols = 4;
    diag_mat_dmat(&A, 2);
    assert(dmat_is_symm(&A));
    ZERO_SM(B)
    push_entry_smat(&B, 0, 0, 16);
    push_entry_smat(&B, 0, 2, 10);
    push_entry_smat(&B, 2, 1, 20);
    push_entry_smat(&B, 3, 2, 4);
    push_entry_smat(&B, 3, 3, 1);
    mat_addto_dmat(&B, &A);
    assert(!memcmp((real[]){
        18,0, 10,0,
        0, 2, 0, 0,
        0, 20,2, 0,
        0, 0, 4, 3
    }, A.data.dense, 16*sizeof(real)));

    make_mat(&C, 4, 4, 0);
    memcpy(C.data.dense, (real[]){
        1,2,3,4,
        4,3,2,1,
        0,1,2,-1,
        3,3,3,3
    }, 16*sizeof(real));
    mat_addto_dmat(&C, &A);
    assert(!memcmp((real[]){
        19,2, 13,4,
        4, 5, 2, 1,
        0, 21,4,-1,
        3, 3, 7, 6
    }, A.data.dense, 16*sizeof(real)));
    A.nrows = 2; C.nrows = 2;
    mat_addto_dmat(&C, &A);
    assert(!memcmp((real[]){
        20,4, 16,8,
        8, 8, 4, 2
    }, A.data.dense, 8*sizeof(real)));

    C.nrows = 4; v2.dim = 4;
    dmat_mul_vec_writo_dvec(&C, &v1, &v2);
    assert(!memcmp((real[]){
        23, 27, 5, 30
    }, v2.data.dense, 4*sizeof(real)));

    C.nrows = 3;
    v2.dim = 3;
    dmat_mul_vec_writo_dvec(&C, &v1, &v2);
    assert(!memcmp((real[]){
        23, 27, 5
    }, v2.data.dense, 3*sizeof(real)));

    dmat_mul_vec_writo_dvec(&C, &u1, &v2);
    assert(!memcmp((real[]){
        38, 42, 22
    }, v2.data.dense, 3*sizeof(real)));

    C.nrows = 4;
    v2.dim = 4;
    dmat_mul_vec_writo_dvec(&C, &u1, &v2);
    assert(!memcmp((real[]){
        38, 42, 22, 48
    }, v2.data.dense, 4*sizeof(real)));

    // copy tests
    v1.dim = 4;
    ZERO_DV(v1);
    copy_vec_to_vec(&v2, &v1);
    assert(!memcmp((real[]){
        38, 42, 22, 48
    }, v1.data.dense, 4*sizeof(real)));

    Vec v4; make_vec(&v4, 0, 0);
    // copy to unallocated dense
    copy_vec_to_vec(&v1, &v4);
    assert(v4.dim == 4);
    assert(VEC_IS_DENSE(v4));
    assert(!memcmp(v2.data.dense, v4.data.dense, 4*sizeof(real)));
    ZERO_DV(v4)
    // copy to allocated dense
    copy_vec_to_vec(&v1, &v4);
    assert(v4.dim == 4);
    assert(VEC_IS_DENSE(v4));
    assert(!memcmp(v2.data.dense, v4.data.dense, 4*sizeof(real)));

    Vec u3; make_vec(&u3, 0, 0);
    // copy to unallocated sparse
    copy_vec_to_vec(&u1, &u3);
    print_vec(&u1);
    print_vec(&u3);
    assert(u3.dim == 4);
    assert(VEC_IS_SPARSE(u3));
    assert(u3.sparse_nentries == 2);
    assert(u3.sparse_cap == 2);
    assert(u3.data.sparse[0].idx == 0);
    assert(u3.data.sparse[0].val == 5);
    assert(u3.data.sparse[1].idx == 2);
    assert(u3.data.sparse[1].val == 11);
    ZERO_SV(u3)
    // copy to allocated sparse
    copy_vec_to_vec(&u1, &u3);
    assert(u3.dim == 4);
    assert(VEC_IS_SPARSE(u3));
    assert(u3.sparse_nentries == 2);
    assert(u3.sparse_cap == 2);
    assert(u3.data.sparse[0].idx == 0);
    assert(u3.data.sparse[0].val == 5);
    assert(u3.data.sparse[1].idx == 2);
    assert(u3.data.sparse[1].val == 11);

    free_vec(&v1);
    free_vec(&v2);
    free_vec(&v3);
    free_vec(&v4);
    free_vec(&u1);
    free_vec(&u2);
    free_vec(&u3);
    free_mat(&A);
    free_mat(&B);
    free_mat(&C);
#ifndef NDEBUG
    puts( "---------------- All vec/mat tests passed.");
#else
    fputs("---------------- No vec/mat tests were ran.\n", stderr);
#endif
}

void smoothfn_qfn_tests() {
    print_sizeof(Linear);
    Vec lin_param; make_vec(&lin_param, 0, 0);
    Vec lin_deriv; make_vec(&lin_deriv, 0, 0);
    Vec lin_feavec; make_vec(&lin_feavec, 2, 0);
    lin_feavec.data.dense[0] = 82.f;
    lin_feavec.data.dense[1] = 77.f;

    Linear lin; make_Linear(&lin, 2);
    lin.super.VT_ACCESS alloc_init_theta(&lin, &lin_param);
    lin.super.VT_ACCESS prealloc_deriv_vec(&lin, &lin_deriv);
    assert(lin_param.dim == 2);

    lin_param.data.dense[0] = 21.f;
    lin_param.data.dense[1] = -17.f;
    assert(lin.super.VT_ACCESS f(&lin, &lin_param, &lin_feavec) == 21*82 - 17*77);

    lin.super.VT_ACCESS deriv_writo_vec(&lin, &lin_param, &lin_feavec, &lin_deriv);
    assert(lin_deriv.dim == 2);
    assert(lin_deriv.data.dense[0] == 82);
    assert(lin_deriv.data.dense[1] == 77);
    free_Linear(&lin);

    // qfn test. need to set up a ton of stuff.
    // spaces
    Space sspace, aspace;
    make_space(&sspace, 2);
    make_space(&aspace, 2);
    set_space_range_factor(&sspace, 0, (Range){.a=-2, .b=2});
    set_space_range_factor(&sspace, 1, (Range){.a= 0, .b=1});
    set_space_discrete_factor(&aspace, 0, (Discrete){.n=2});
    set_space_discrete_factor(&aspace, 1, (Discrete){.n=2});
    // elem
    Elem A, S;
    make_elem(&A, &aspace);
    make_elem(&S, &sspace);
    S.x[0].r = 1.5f;
    S.x[1].r = 0.3f;
    A.x[0].i = 1;
    A.x[1].i = 0;
    // sa feamaps
    uint fborders[] = {2, 2};
    FourierBasis sfmap; make_FourierBasis(&sfmap, &sspace, fborders);
    OneHot afmap; make_OneHot(&afmap, &aspace, 0);
    FlatOuterProduct safmap;
    make_FlatOuterProduct(&safmap, &sfmap, &afmap,
                          FLATOUTERPROD_FL_FLIPORDER);
    // lin approx
    make_Linear(&lin, (*(SAFeatureMap*)&safmap).outdim);
    // finally
    print_sizeof(QFn);
    QFn q; make_QFn(&q, (void*)&safmap, (void*)&lin);
    // fea_A = (cos(pi(1.5*)))
    assert(q.theta.dim == 2 * 2 * 3 * 3);
    assert(Qtheta(&q, &S, &A) == 0.f); // since weight vec initd as all 0s

    // change things so its not just 0
    q.theta.data.dense[0] = 2;
    q.theta.data.dense[1] = 4;
    q.theta.data.dense[10] = 7;
    q.theta.data.dense[15] = -3;
    q.theta.data.dense[20] = 4;
    q.theta.data.dense[25] = 5;
    q.theta.data.dense[35] = -6;

    // only 18-26 matters, due to the onehot
    assert(req(Qtheta(&q, &S, &A),
         4*cosf(M_PI*(2*3.5f/4.f)) + 5*cosf(M_PI*(3.5f/4.f + 2*0.3f)), 1e-7));

    print_vec(&q.fea_SA);
    free_vec(&lin_param);
    free_vec(&lin_deriv);
    free_vec(&lin_feavec);
    free_elem(&A);
    free_elem(&S);
    free_space(&sspace);
    free_space(&aspace);
    free_OneHot(&afmap);
    free_FourierBasis(&sfmap);
    free_FlatOuterProduct(&safmap);
    free_Linear(&lin);
    free_QFn(&q);

#ifndef NDEBUG
    puts( "---------------- All smoothfn/qfn tests passed.");
#else
    fputs("---------------- No smoothfn/qfn tests were ran.\n", stderr);
#endif
}

void space_feamaps_safeamap_tests() {
    print_sizeof(Discrete);
    print_sizeof(Range);
    print_sizeof(SimpleSet);
    print_sizeof(SimpleElem);
    print_sizeof(Space);
    print_sizeof(Elem);

    Space a, b, x, y;
    Elem ea, eb, ex, ey;

    make_space(&a, 1);
    set_space_discrete_factor(&a, 0, (Discrete){.n = 10});
    assert(a.nfactors == 1);
    assert(a.factors[0].type == 'd');
    assert(a.factors[0].as.discrete.n == 10);
    assert(get_finite_space_cardinality(&a) == 10);

    make_space(&b, 4);
    set_space_discrete_factor(&b, 0, (Discrete){.n = 100});
    set_space_discrete_factor(&b, 1, (Discrete){.n = 200});
    set_space_discrete_factor(&b, 2, (Discrete){.n = 50});
    set_space_discrete_factor(&b, 3, (Discrete){.n = 80});
    assert(b.nfactors == 4);
    assert(b.factors[0].type == 'd');
    assert(b.factors[1].type == 'd');
    assert(b.factors[2].type == 'd');
    assert(b.factors[3].type == 'd');
    assert(b.factors[0].as.discrete.n == 100);
    assert(b.factors[1].as.discrete.n == 200);
    assert(b.factors[2].as.discrete.n == 50);
    assert(b.factors[3].as.discrete.n == 80);
    assert(get_finite_space_cardinality(&b) == 100*200*50*80);

    make_elem(&ea, &a);
    // assert(ea.space == &a);
    ea.x[0].i = 5;

    make_elem(&eb, &b);
    // assert(eb.space == &b);
    eb.x[0].i = 13;
    eb.x[1].i = 37;
    eb.x[2].i = 69;
    eb.x[3].i = 79;


    print_sizeof(OneHot);

    OneHot oha, ohb;

    make_OneHot(&oha, &a, 0);
    make_OneHot(&ohb, &b, 0);
    assert(oha.super.outdim == 10);
    assert(ohb.super.outdim == 100*200*50*80);

    Vec va, vb;
    make_vec(&va, 0, 0);
    make_vec(&vb, 0, 0);
    oha.super.VT_ACCESS prealloc_out_vec(&oha, &va);
    ohb.super.VT_ACCESS prealloc_out_vec(&ohb, &vb);
    oha.super.VT_ACCESS map_writo_vec(&oha, &ea, &va);
    ohb.super.VT_ACCESS map_writo_vec(&ohb, &eb, &vb);
    // duplicate call, test ~push flag
    ohb.super.VT_ACCESS map_writo_vec(&ohb, &eb, &vb);
    assert(va.dim == 10);
    assert(vb.dim == 100*200*50*80);
    assert(VEC_IS_SPARSE(va) && va.sparse_nentries == 1);
    assert(VEC_IS_SPARSE(vb) && vb.sparse_nentries == 1);
    assert(!memcmp(va.data.sparse, (SparseEntry[]) {
        {.idx = 5, .val = 1}
    }, sizeof(SparseEntry)));
    assert(!memcmp(vb.data.sparse, (SparseEntry[]) {
        {.idx = 13*200*50*80 + 37*50*80 + 69*80 + 79, .val = 1}
    }, sizeof(SparseEntry)));

    make_space(&x, 1);
    make_space(&y, 3);
    set_space_range_factor(&x, 0, (Range){.a = -1.0f, .b = 1.0f});
    set_space_range_factor(&y, 0, (Range){.a = 0.0f, .b = 10.0f});
    set_space_discrete_factor(&y, 1, (Discrete){.n=12});
    set_space_range_factor(&y, 2, (Range){.a = 50.0f, .b = 100.0f});
    assert(x.nfactors == 1);
    assert(y.nfactors == 3);
    print_space(&x);
    // some weird behavior with gcc -Og, can't do !memcmp
    assert(x.factors[0].as.range.a == -1);
    assert(x.factors[0].as.range.b == 1);
    assert(x.factors[0].type == 'r');

    print_space(&y);
    assert(y.factors[0].type == 'r');
    assert(y.factors[1].type == 'd');
    assert(y.factors[2].type == 'r');
    assert(y.factors[0].as.range.a == 0.0f);
    assert(y.factors[0].as.range.b == 10.0f);
    assert(y.factors[1].as.discrete.n == 12);
    assert(y.factors[2].as.range.a == 50.0f);
    assert(y.factors[2].as.range.b == 100.0f);

    print_space(&a);
    print_space(&b);
    print_space(&x);
    print_space(&y);


    print_sizeof(FourierBasis);

    uint orderx = 10;
    FourierBasis fourierx; make_FourierBasis(&fourierx, &x, &orderx);
    make_elem(&ex, &x);
    ex.x[0].r = 0.82f;
    Vec vx; make_vec(&vx, 0, 0);
    (*(FeatureMap*)&fourierx).VT_ACCESS prealloc_out_vec(&fourierx, &vx);
    (*(FeatureMap*)&fourierx).VT_ACCESS map_writo_vec(&fourierx, &ex, &vx);
    assert(vx.dim == 10+1);
    assert(vx.data.dense[0] == 1);
    // yo wtf this passes with -Ofast
    assert(vx.data.dense[3] == cosf(M_PI*3*((0.82f - -1.f) / 2.f)));
    assert(vx.data.dense[5] == cosf(M_PI*5*((0.82f - -1.f) / 2.f)));

    // y = [0f, 10f] x [0f, 1f] x [50f, 100f]
    set_space_range_factor(&y, 1, (Range){.a = 0.0f, .b = 1.0f});
    print_space(&y); // need this or else assert fails on -O1 lol
    uint ordersy[] = {3, 3, 5};
    FourierBasis fouriery; make_FourierBasis(&fouriery, &y, ordersy);
    make_elem(&ey, &y);
    ey.x[0].r = 2;
    ey.x[1].r = 0.5f;
    ey.x[2].r = 55.0f;
    Vec vy; make_vec(&vy, 0, 0);
    fouriery.super.VT_ACCESS prealloc_out_vec(&fouriery, &vy);
    fouriery.super.VT_ACCESS map_writo_vec(&fouriery, &ey, &vy);
    //print_vec(&vy);
    assert(vy.dim == (3+1)*(3+1)*(5+1));
    assert(vy.data.dense[0] == 1);
    assert(req(vy.data.dense[1 + 0*(3+1) + 0*(3+1)*(3+1)],
        cosf(M_PI*2.f/10.f),1e-6));
    assert(req(vy.data.dense[0 + 1*(3+1) + 0*(3+1)*(3+1)],
        cosf(M_PI*0.5), 1e-6));
    assert(req(vy.data.dense[0 + 0*(3+1) + 1*(3+1)*(3+1)],
        cosf(M_PI*(55.f-50.f)/(100.f - 50.f)), 1e-6));
    assert(req(vy.data.dense[0 + 0*(3+1) + 3*(3+1)*(3+1)],
        cosf(M_PI*3*(55.f-50.f)/(100.f - 50.f)), 1e-6));
    // printf("xdotc should be %.6g\n", (2*2.f/10.f + 0.5f +
    // 5*(55.f-50.f)/50.f));
    // printf("cos(...) should be %.6g but got %.6g\n",
    //     cosf(M_PI*(2*2.f/10.f + 0.5f + 5*(55.f-50.f)/50.f)),
    //     vy.data.dense[2 + 1*(3+1) + 5*(3+1)*(3+1)]
    // );
    assert(req(vy.data.dense[2 + 1*(3+1) + 5*(3+1)*(3+1)], cosf(M_PI*(
        2*2.f/10.f + 0.5f + 5*(55.f-50.f)/50.f
    )), 1e-6));
    assert(req(vy.data.dense[0 + 2*(3+1) + 2*(3+1)*(3+1)], cosf(M_PI*(
        0*2.f/10.f + 2*0.5f + 2*(55.f-50.f)/50.f
    )), 1e-6));
    assert(req(vy.data.dense[3 + 3*(3+1) + 2*(3+1)*(3+1)], cosf(M_PI*(
        3*2.f/10.f + 3*0.5f + 2*(55.f-50.f)/50.f
    )), 1e-6));
    assert(req(vy.data.dense[3 + 3*(3+1) + 5*(3+1)*(3+1)], cosf(M_PI*(
        3*2.f/10.f + 3*0.5f + 5*(55.f-50.f)/50.f
    )), 1e-6));

    // really annoying to exhaustively test flat outer product feamap
    // ea = 5                \in [0..10i)
    // eb = (13,37,69,79)    \in [0..100i) x [0..200i) x [0..50i) x [0..80i)
    // ex = .82f             \in [-1f, 1f]
    // ey = (2.f, .5f, 55.f) \in [0f, 10f] x [0f, 1f] x [50f, 100f]
    // va = [5]=1
    // vb = [13*200*50*80+37*50*80+69*80+79]=1
    print_sizeof(SAFeatureMap);
    print_sizeof(FlatOuterProduct);
    FlatOuterProduct outermap;
    Vec vsa;

    // onehot(ea) onehot(eb)^T
    make_vec(&vsa, 0, 0);
    make_FlatOuterProduct(&outermap, &oha, &ohb, 0);
    ((SAFeatureMap*)&outermap)->VT_ACCESS prealloc_out_vec(&outermap, &vsa);
    ((SAFeatureMap*)&outermap)->VT_ACCESS map_writo_vec(&outermap, &ea, &eb,
                                                        &vsa);
    ((SAFeatureMap*)&outermap)->VT_ACCESS map_writo_vec(&outermap,&ea, &eb,
                                                        &vsa);
    assert(vsa.dim == 10 * 100 * 200 * 50 * 80);
    assert(VEC_IS_SPARSE(vsa));
    assert(vsa.sparse_nentries == 1);
    assert(!memcmp(vsa.data.sparse, &(SparseEntry){
        // idx = ~410M well within uint range
        .val=1, .idx= 5*(100*200*50*80) + (13*200*50*80+37*50*80+69*80+79),
    }, sizeof(SparseEntry)));
    print_vec(&vsa);
    free_FlatOuterProduct(&outermap);
    free_vec(&vsa);

    // onehot(ea) fourier(ey)^T
    make_vec(&vsa, 0, 0);
    make_FlatOuterProduct(&outermap, &oha, &fouriery, 0);
    outermap.super.super.VT_ACCESS prealloc_out_vec(&outermap, &vsa);
    outermap.super.super.VT_ACCESS map_writo_vec(&outermap, &ea, &ey, &vsa);
    assert(vsa.dim == 10 * 4*4*6);
    assert(VEC_IS_SPARSE(vsa)); // should still be sparse: sparsity 10 > threshold 8
    assert(vsa.sparse_nentries == 4*4*6);
    for (uint i = 0; i < 4*4*6; i++) {
        assert(vsa.data.sparse[i].idx == 5*4*4*6 + i);
        assert(req(vsa.data.sparse[i].val, vy.data.dense[i], 1e-7));
    }
    free_FlatOuterProduct(&outermap);
    free_vec(&vsa);

    // onehot(ea) fourier(ex)^T with "flip order" flag
    // actually that flag was kinda unnecessary since both S and A types are
    // Elem so i can just pass it in in reversed order. am i stupid? idk
    make_vec(&vsa, 0, 0);
    make_FlatOuterProduct(&outermap, &fourierx, &oha,
                          FLATOUTERPROD_FL_FLIPORDER);
    outermap.super.super.VT_ACCESS prealloc_out_vec(&outermap, &vsa);
    outermap.super.super.VT_ACCESS map_writo_vec(&outermap, &ex, &ea, &vsa);
    assert(vsa.dim == 10 * 11);
    assert(VEC_IS_SPARSE(vsa)); // should still be sparse: sparsity 10 > threshold 8
    assert(vsa.sparse_nentries == 11);
    for (uint i = 0; i < 11; i++) {
        assert(vsa.data.sparse[i].idx == 5*11 + i);
        assert(req(vsa.data.sparse[i].val, vx.data.dense[i], 1e-9));
    }
    free_FlatOuterProduct(&outermap);
    free_vec(&vsa);

    free_space(&a);
    free_space(&b);
    free_space(&x);
    free_space(&y);
    free_elem(&ea);
    free_elem(&eb);
    free_elem(&ex);
    free_elem(&ey);
    free_vec(&va);
    free_vec(&vb);
    free_vec(&vx);
    free_vec(&vy);
    free_FourierBasis(&fouriery);
    free_FourierBasis(&fourierx);
    free_OneHot(&oha);
    free_OneHot(&ohb);
#ifndef NDEBUG
    puts( "---------------- All space/feamap/safeamap tests passed.");
#else
    fputs("---------------- No space/feamap tests were ran.\n", stderr);
#endif
}

void exp_tests() {
    MountainCar mcar;
    make_MountainCar(&mcar, 0);
    Elem S, A;
    make_elem(&S, &mcar.super.SVT_ACCESS state_space);
    make_elem(&A, &mcar.super.SVT_ACCESS fixed_action_space);

    mcar.super.VT_ACCESS start_state(&mcar, &test_rngs, &S);
    print_expr(S.x[0].r, "%.3f");
    assert(-0.6f <= S.x[0].r && S.x[0].r <= -0.4f);
    assert(S.x[1].r == 0.f);

    S.x[0].r = M_PI/6.f + 1.f;
    assert(mcar.super.VT_ACCESS is_terminal(&mcar, &S, 1));

    free_elem(&S);
    free_elem(&A);
    free_MountainCar(&mcar);
}

void all_tests() {
    vec_mat_tests();
    space_feamaps_safeamap_tests();
    smoothfn_qfn_tests();
    exp_tests();
}
