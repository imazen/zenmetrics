//! Per-thread-accumulator cov_accum kernels (auto-generated body).
//! Each thread maintains a 9x9 (or 10x10) local f32 register file,
//! accumulates outer products over its grid-strided pixel range, and
//! atomic-adds to global cu ONCE at the end — reducing global atomic
//! traffic by factor of (nexp / total_threads), typically ~250x.

use cubecl::prelude::*;

#[cube(launch_unchecked)]
pub fn cov_accum_no_parent_kernel(
    lp: &Array<f32>,
    cu: &mut Array<Atomic<f32>>,
    h: u32,
    w: u32,
) {
    let tid = ABSOLUTE_POS;
    let stride = CUBE_COUNT * (CUBE_DIM_X as usize);
    let nblv = h - 2;
    let nblh = w - 2;
    let nexp = (nblv * nblh) as usize;
    let w_us = w as usize;

    let mut a00 = 0.0_f32; let mut a01 = 0.0_f32; let mut a02 = 0.0_f32; let mut a03 = 0.0_f32; let mut a04 = 0.0_f32; let mut a05 = 0.0_f32; let mut a06 = 0.0_f32; let mut a07 = 0.0_f32; let mut a08 = 0.0_f32;
    let mut a10 = 0.0_f32; let mut a11 = 0.0_f32; let mut a12 = 0.0_f32; let mut a13 = 0.0_f32; let mut a14 = 0.0_f32; let mut a15 = 0.0_f32; let mut a16 = 0.0_f32; let mut a17 = 0.0_f32; let mut a18 = 0.0_f32;
    let mut a20 = 0.0_f32; let mut a21 = 0.0_f32; let mut a22 = 0.0_f32; let mut a23 = 0.0_f32; let mut a24 = 0.0_f32; let mut a25 = 0.0_f32; let mut a26 = 0.0_f32; let mut a27 = 0.0_f32; let mut a28 = 0.0_f32;
    let mut a30 = 0.0_f32; let mut a31 = 0.0_f32; let mut a32 = 0.0_f32; let mut a33 = 0.0_f32; let mut a34 = 0.0_f32; let mut a35 = 0.0_f32; let mut a36 = 0.0_f32; let mut a37 = 0.0_f32; let mut a38 = 0.0_f32;
    let mut a40 = 0.0_f32; let mut a41 = 0.0_f32; let mut a42 = 0.0_f32; let mut a43 = 0.0_f32; let mut a44 = 0.0_f32; let mut a45 = 0.0_f32; let mut a46 = 0.0_f32; let mut a47 = 0.0_f32; let mut a48 = 0.0_f32;
    let mut a50 = 0.0_f32; let mut a51 = 0.0_f32; let mut a52 = 0.0_f32; let mut a53 = 0.0_f32; let mut a54 = 0.0_f32; let mut a55 = 0.0_f32; let mut a56 = 0.0_f32; let mut a57 = 0.0_f32; let mut a58 = 0.0_f32;
    let mut a60 = 0.0_f32; let mut a61 = 0.0_f32; let mut a62 = 0.0_f32; let mut a63 = 0.0_f32; let mut a64 = 0.0_f32; let mut a65 = 0.0_f32; let mut a66 = 0.0_f32; let mut a67 = 0.0_f32; let mut a68 = 0.0_f32;
    let mut a70 = 0.0_f32; let mut a71 = 0.0_f32; let mut a72 = 0.0_f32; let mut a73 = 0.0_f32; let mut a74 = 0.0_f32; let mut a75 = 0.0_f32; let mut a76 = 0.0_f32; let mut a77 = 0.0_f32; let mut a78 = 0.0_f32;
    let mut a80 = 0.0_f32; let mut a81 = 0.0_f32; let mut a82 = 0.0_f32; let mut a83 = 0.0_f32; let mut a84 = 0.0_f32; let mut a85 = 0.0_f32; let mut a86 = 0.0_f32; let mut a87 = 0.0_f32; let mut a88 = 0.0_f32;

    let mut p = tid;
    while p < nexp {
        let py = (p as u32) / nblh;
        let px = (p as u32) - py * nblh;
        let py_us = py as usize;
        let px_us = px as usize;
        let v0 = lp[(py_us + 2) * w_us + (px_us + 2)];
        let v1 = lp[(py_us + 2) * w_us + (px_us + 1)];
        let v2 = lp[(py_us + 2) * w_us + px_us];
        let v3 = lp[(py_us + 1) * w_us + (px_us + 2)];
        let v4 = lp[(py_us + 1) * w_us + (px_us + 1)];
        let v5 = lp[(py_us + 1) * w_us + px_us];
        let v6 = lp[py_us * w_us + (px_us + 2)];
        let v7 = lp[py_us * w_us + (px_us + 1)];
        let v8 = lp[py_us * w_us + px_us];
        a00 += v0 * v0; a01 += v0 * v1; a02 += v0 * v2; a03 += v0 * v3; a04 += v0 * v4; a05 += v0 * v5; a06 += v0 * v6; a07 += v0 * v7; a08 += v0 * v8;
        a10 += v1 * v0; a11 += v1 * v1; a12 += v1 * v2; a13 += v1 * v3; a14 += v1 * v4; a15 += v1 * v5; a16 += v1 * v6; a17 += v1 * v7; a18 += v1 * v8;
        a20 += v2 * v0; a21 += v2 * v1; a22 += v2 * v2; a23 += v2 * v3; a24 += v2 * v4; a25 += v2 * v5; a26 += v2 * v6; a27 += v2 * v7; a28 += v2 * v8;
        a30 += v3 * v0; a31 += v3 * v1; a32 += v3 * v2; a33 += v3 * v3; a34 += v3 * v4; a35 += v3 * v5; a36 += v3 * v6; a37 += v3 * v7; a38 += v3 * v8;
        a40 += v4 * v0; a41 += v4 * v1; a42 += v4 * v2; a43 += v4 * v3; a44 += v4 * v4; a45 += v4 * v5; a46 += v4 * v6; a47 += v4 * v7; a48 += v4 * v8;
        a50 += v5 * v0; a51 += v5 * v1; a52 += v5 * v2; a53 += v5 * v3; a54 += v5 * v4; a55 += v5 * v5; a56 += v5 * v6; a57 += v5 * v7; a58 += v5 * v8;
        a60 += v6 * v0; a61 += v6 * v1; a62 += v6 * v2; a63 += v6 * v3; a64 += v6 * v4; a65 += v6 * v5; a66 += v6 * v6; a67 += v6 * v7; a68 += v6 * v8;
        a70 += v7 * v0; a71 += v7 * v1; a72 += v7 * v2; a73 += v7 * v3; a74 += v7 * v4; a75 += v7 * v5; a76 += v7 * v6; a77 += v7 * v7; a78 += v7 * v8;
        a80 += v8 * v0; a81 += v8 * v1; a82 += v8 * v2; a83 += v8 * v3; a84 += v8 * v4; a85 += v8 * v5; a86 += v8 * v6; a87 += v8 * v7; a88 += v8 * v8;
        p += stride;
    }

    cu[0].fetch_add(a00);
    cu[1].fetch_add(a01);
    cu[2].fetch_add(a02);
    cu[3].fetch_add(a03);
    cu[4].fetch_add(a04);
    cu[5].fetch_add(a05);
    cu[6].fetch_add(a06);
    cu[7].fetch_add(a07);
    cu[8].fetch_add(a08);
    cu[9].fetch_add(a10);
    cu[10].fetch_add(a11);
    cu[11].fetch_add(a12);
    cu[12].fetch_add(a13);
    cu[13].fetch_add(a14);
    cu[14].fetch_add(a15);
    cu[15].fetch_add(a16);
    cu[16].fetch_add(a17);
    cu[17].fetch_add(a18);
    cu[18].fetch_add(a20);
    cu[19].fetch_add(a21);
    cu[20].fetch_add(a22);
    cu[21].fetch_add(a23);
    cu[22].fetch_add(a24);
    cu[23].fetch_add(a25);
    cu[24].fetch_add(a26);
    cu[25].fetch_add(a27);
    cu[26].fetch_add(a28);
    cu[27].fetch_add(a30);
    cu[28].fetch_add(a31);
    cu[29].fetch_add(a32);
    cu[30].fetch_add(a33);
    cu[31].fetch_add(a34);
    cu[32].fetch_add(a35);
    cu[33].fetch_add(a36);
    cu[34].fetch_add(a37);
    cu[35].fetch_add(a38);
    cu[36].fetch_add(a40);
    cu[37].fetch_add(a41);
    cu[38].fetch_add(a42);
    cu[39].fetch_add(a43);
    cu[40].fetch_add(a44);
    cu[41].fetch_add(a45);
    cu[42].fetch_add(a46);
    cu[43].fetch_add(a47);
    cu[44].fetch_add(a48);
    cu[45].fetch_add(a50);
    cu[46].fetch_add(a51);
    cu[47].fetch_add(a52);
    cu[48].fetch_add(a53);
    cu[49].fetch_add(a54);
    cu[50].fetch_add(a55);
    cu[51].fetch_add(a56);
    cu[52].fetch_add(a57);
    cu[53].fetch_add(a58);
    cu[54].fetch_add(a60);
    cu[55].fetch_add(a61);
    cu[56].fetch_add(a62);
    cu[57].fetch_add(a63);
    cu[58].fetch_add(a64);
    cu[59].fetch_add(a65);
    cu[60].fetch_add(a66);
    cu[61].fetch_add(a67);
    cu[62].fetch_add(a68);
    cu[63].fetch_add(a70);
    cu[64].fetch_add(a71);
    cu[65].fetch_add(a72);
    cu[66].fetch_add(a73);
    cu[67].fetch_add(a74);
    cu[68].fetch_add(a75);
    cu[69].fetch_add(a76);
    cu[70].fetch_add(a77);
    cu[71].fetch_add(a78);
    cu[72].fetch_add(a80);
    cu[73].fetch_add(a81);
    cu[74].fetch_add(a82);
    cu[75].fetch_add(a83);
    cu[76].fetch_add(a84);
    cu[77].fetch_add(a85);
    cu[78].fetch_add(a86);
    cu[79].fetch_add(a87);
    cu[80].fetch_add(a88);
}

#[cube(launch_unchecked)]
pub fn cov_accum_with_parent_kernel(
    lp: &Array<f32>,
    parent: &Array<f32>,
    cu: &mut Array<Atomic<f32>>,
    h: u32,
    w: u32,
) {
    let tid = ABSOLUTE_POS;
    let stride = CUBE_COUNT * (CUBE_DIM_X as usize);
    let nblv = h - 2;
    let nblh = w - 2;
    let nexp = (nblv * nblh) as usize;
    let w_us = w as usize;

    let mut a00 = 0.0_f32; let mut a01 = 0.0_f32; let mut a02 = 0.0_f32; let mut a03 = 0.0_f32; let mut a04 = 0.0_f32; let mut a05 = 0.0_f32; let mut a06 = 0.0_f32; let mut a07 = 0.0_f32; let mut a08 = 0.0_f32; let mut a09 = 0.0_f32;
    let mut a10 = 0.0_f32; let mut a11 = 0.0_f32; let mut a12 = 0.0_f32; let mut a13 = 0.0_f32; let mut a14 = 0.0_f32; let mut a15 = 0.0_f32; let mut a16 = 0.0_f32; let mut a17 = 0.0_f32; let mut a18 = 0.0_f32; let mut a19 = 0.0_f32;
    let mut a20 = 0.0_f32; let mut a21 = 0.0_f32; let mut a22 = 0.0_f32; let mut a23 = 0.0_f32; let mut a24 = 0.0_f32; let mut a25 = 0.0_f32; let mut a26 = 0.0_f32; let mut a27 = 0.0_f32; let mut a28 = 0.0_f32; let mut a29 = 0.0_f32;
    let mut a30 = 0.0_f32; let mut a31 = 0.0_f32; let mut a32 = 0.0_f32; let mut a33 = 0.0_f32; let mut a34 = 0.0_f32; let mut a35 = 0.0_f32; let mut a36 = 0.0_f32; let mut a37 = 0.0_f32; let mut a38 = 0.0_f32; let mut a39 = 0.0_f32;
    let mut a40 = 0.0_f32; let mut a41 = 0.0_f32; let mut a42 = 0.0_f32; let mut a43 = 0.0_f32; let mut a44 = 0.0_f32; let mut a45 = 0.0_f32; let mut a46 = 0.0_f32; let mut a47 = 0.0_f32; let mut a48 = 0.0_f32; let mut a49 = 0.0_f32;
    let mut a50 = 0.0_f32; let mut a51 = 0.0_f32; let mut a52 = 0.0_f32; let mut a53 = 0.0_f32; let mut a54 = 0.0_f32; let mut a55 = 0.0_f32; let mut a56 = 0.0_f32; let mut a57 = 0.0_f32; let mut a58 = 0.0_f32; let mut a59 = 0.0_f32;
    let mut a60 = 0.0_f32; let mut a61 = 0.0_f32; let mut a62 = 0.0_f32; let mut a63 = 0.0_f32; let mut a64 = 0.0_f32; let mut a65 = 0.0_f32; let mut a66 = 0.0_f32; let mut a67 = 0.0_f32; let mut a68 = 0.0_f32; let mut a69 = 0.0_f32;
    let mut a70 = 0.0_f32; let mut a71 = 0.0_f32; let mut a72 = 0.0_f32; let mut a73 = 0.0_f32; let mut a74 = 0.0_f32; let mut a75 = 0.0_f32; let mut a76 = 0.0_f32; let mut a77 = 0.0_f32; let mut a78 = 0.0_f32; let mut a79 = 0.0_f32;
    let mut a80 = 0.0_f32; let mut a81 = 0.0_f32; let mut a82 = 0.0_f32; let mut a83 = 0.0_f32; let mut a84 = 0.0_f32; let mut a85 = 0.0_f32; let mut a86 = 0.0_f32; let mut a87 = 0.0_f32; let mut a88 = 0.0_f32; let mut a89 = 0.0_f32;
    let mut a90 = 0.0_f32; let mut a91 = 0.0_f32; let mut a92 = 0.0_f32; let mut a93 = 0.0_f32; let mut a94 = 0.0_f32; let mut a95 = 0.0_f32; let mut a96 = 0.0_f32; let mut a97 = 0.0_f32; let mut a98 = 0.0_f32; let mut a99 = 0.0_f32;

    let mut p = tid;
    while p < nexp {
        let py = (p as u32) / nblh;
        let px = (p as u32) - py * nblh;
        let py_us = py as usize;
        let px_us = px as usize;
        let v0 = lp[(py_us + 2) * w_us + (px_us + 2)];
        let v1 = lp[(py_us + 2) * w_us + (px_us + 1)];
        let v2 = lp[(py_us + 2) * w_us + px_us];
        let v3 = lp[(py_us + 1) * w_us + (px_us + 2)];
        let v4 = lp[(py_us + 1) * w_us + (px_us + 1)];
        let v5 = lp[(py_us + 1) * w_us + px_us];
        let v6 = lp[py_us * w_us + (px_us + 2)];
        let v7 = lp[py_us * w_us + (px_us + 1)];
        let v8 = lp[py_us * w_us + px_us];
        let v9 = parent[(py_us + 1) * w_us + (px_us + 1)];
        a00 += v0 * v0; a01 += v0 * v1; a02 += v0 * v2; a03 += v0 * v3; a04 += v0 * v4; a05 += v0 * v5; a06 += v0 * v6; a07 += v0 * v7; a08 += v0 * v8; a09 += v0 * v9;
        a10 += v1 * v0; a11 += v1 * v1; a12 += v1 * v2; a13 += v1 * v3; a14 += v1 * v4; a15 += v1 * v5; a16 += v1 * v6; a17 += v1 * v7; a18 += v1 * v8; a19 += v1 * v9;
        a20 += v2 * v0; a21 += v2 * v1; a22 += v2 * v2; a23 += v2 * v3; a24 += v2 * v4; a25 += v2 * v5; a26 += v2 * v6; a27 += v2 * v7; a28 += v2 * v8; a29 += v2 * v9;
        a30 += v3 * v0; a31 += v3 * v1; a32 += v3 * v2; a33 += v3 * v3; a34 += v3 * v4; a35 += v3 * v5; a36 += v3 * v6; a37 += v3 * v7; a38 += v3 * v8; a39 += v3 * v9;
        a40 += v4 * v0; a41 += v4 * v1; a42 += v4 * v2; a43 += v4 * v3; a44 += v4 * v4; a45 += v4 * v5; a46 += v4 * v6; a47 += v4 * v7; a48 += v4 * v8; a49 += v4 * v9;
        a50 += v5 * v0; a51 += v5 * v1; a52 += v5 * v2; a53 += v5 * v3; a54 += v5 * v4; a55 += v5 * v5; a56 += v5 * v6; a57 += v5 * v7; a58 += v5 * v8; a59 += v5 * v9;
        a60 += v6 * v0; a61 += v6 * v1; a62 += v6 * v2; a63 += v6 * v3; a64 += v6 * v4; a65 += v6 * v5; a66 += v6 * v6; a67 += v6 * v7; a68 += v6 * v8; a69 += v6 * v9;
        a70 += v7 * v0; a71 += v7 * v1; a72 += v7 * v2; a73 += v7 * v3; a74 += v7 * v4; a75 += v7 * v5; a76 += v7 * v6; a77 += v7 * v7; a78 += v7 * v8; a79 += v7 * v9;
        a80 += v8 * v0; a81 += v8 * v1; a82 += v8 * v2; a83 += v8 * v3; a84 += v8 * v4; a85 += v8 * v5; a86 += v8 * v6; a87 += v8 * v7; a88 += v8 * v8; a89 += v8 * v9;
        a90 += v9 * v0; a91 += v9 * v1; a92 += v9 * v2; a93 += v9 * v3; a94 += v9 * v4; a95 += v9 * v5; a96 += v9 * v6; a97 += v9 * v7; a98 += v9 * v8; a99 += v9 * v9;
        p += stride;
    }

    cu[0].fetch_add(a00);
    cu[1].fetch_add(a01);
    cu[2].fetch_add(a02);
    cu[3].fetch_add(a03);
    cu[4].fetch_add(a04);
    cu[5].fetch_add(a05);
    cu[6].fetch_add(a06);
    cu[7].fetch_add(a07);
    cu[8].fetch_add(a08);
    cu[9].fetch_add(a09);
    cu[10].fetch_add(a10);
    cu[11].fetch_add(a11);
    cu[12].fetch_add(a12);
    cu[13].fetch_add(a13);
    cu[14].fetch_add(a14);
    cu[15].fetch_add(a15);
    cu[16].fetch_add(a16);
    cu[17].fetch_add(a17);
    cu[18].fetch_add(a18);
    cu[19].fetch_add(a19);
    cu[20].fetch_add(a20);
    cu[21].fetch_add(a21);
    cu[22].fetch_add(a22);
    cu[23].fetch_add(a23);
    cu[24].fetch_add(a24);
    cu[25].fetch_add(a25);
    cu[26].fetch_add(a26);
    cu[27].fetch_add(a27);
    cu[28].fetch_add(a28);
    cu[29].fetch_add(a29);
    cu[30].fetch_add(a30);
    cu[31].fetch_add(a31);
    cu[32].fetch_add(a32);
    cu[33].fetch_add(a33);
    cu[34].fetch_add(a34);
    cu[35].fetch_add(a35);
    cu[36].fetch_add(a36);
    cu[37].fetch_add(a37);
    cu[38].fetch_add(a38);
    cu[39].fetch_add(a39);
    cu[40].fetch_add(a40);
    cu[41].fetch_add(a41);
    cu[42].fetch_add(a42);
    cu[43].fetch_add(a43);
    cu[44].fetch_add(a44);
    cu[45].fetch_add(a45);
    cu[46].fetch_add(a46);
    cu[47].fetch_add(a47);
    cu[48].fetch_add(a48);
    cu[49].fetch_add(a49);
    cu[50].fetch_add(a50);
    cu[51].fetch_add(a51);
    cu[52].fetch_add(a52);
    cu[53].fetch_add(a53);
    cu[54].fetch_add(a54);
    cu[55].fetch_add(a55);
    cu[56].fetch_add(a56);
    cu[57].fetch_add(a57);
    cu[58].fetch_add(a58);
    cu[59].fetch_add(a59);
    cu[60].fetch_add(a60);
    cu[61].fetch_add(a61);
    cu[62].fetch_add(a62);
    cu[63].fetch_add(a63);
    cu[64].fetch_add(a64);
    cu[65].fetch_add(a65);
    cu[66].fetch_add(a66);
    cu[67].fetch_add(a67);
    cu[68].fetch_add(a68);
    cu[69].fetch_add(a69);
    cu[70].fetch_add(a70);
    cu[71].fetch_add(a71);
    cu[72].fetch_add(a72);
    cu[73].fetch_add(a73);
    cu[74].fetch_add(a74);
    cu[75].fetch_add(a75);
    cu[76].fetch_add(a76);
    cu[77].fetch_add(a77);
    cu[78].fetch_add(a78);
    cu[79].fetch_add(a79);
    cu[80].fetch_add(a80);
    cu[81].fetch_add(a81);
    cu[82].fetch_add(a82);
    cu[83].fetch_add(a83);
    cu[84].fetch_add(a84);
    cu[85].fetch_add(a85);
    cu[86].fetch_add(a86);
    cu[87].fetch_add(a87);
    cu[88].fetch_add(a88);
    cu[89].fetch_add(a89);
    cu[90].fetch_add(a90);
    cu[91].fetch_add(a91);
    cu[92].fetch_add(a92);
    cu[93].fetch_add(a93);
    cu[94].fetch_add(a94);
    cu[95].fetch_add(a95);
    cu[96].fetch_add(a96);
    cu[97].fetch_add(a97);
    cu[98].fetch_add(a98);
    cu[99].fetch_add(a99);
}

