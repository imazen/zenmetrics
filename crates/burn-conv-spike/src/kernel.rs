//! Copy of `cvvdp_gpu::kernels::pyramid::downscale_kernel`. Verbatim
//! port — see `/home/lilith/work/zen/zenmetrics--cvvdp-new/crates/cvvdp-gpu/src/kernels/pyramid.rs:567`
//! for the source. Pulled into this spike so we can time it without
//! compiling the entire cvvdp-gpu crate.

use cubecl::prelude::*;

#[cube(launch)]
pub fn downscale_kernel(
    src: &Array<f32>,
    dst: &mut Array<f32>,
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
) {
    let idx = ABSOLUTE_POS;
    let total = (dst_w * dst_h) as usize;
    if idx >= total {
        terminate!();
    }
    let dw = dst_w as usize;
    let dy = idx / dw;
    let dx = idx - dy * dw;

    let cy = 2 * (dy as i32);
    let cx = 2 * (dx as i32);
    let sw = src_w as usize;
    let sh = src_h as usize;
    let sh_i = src_h as i32;
    let sw_i = src_w as i32;

    let k0 = f32::new(0.05);
    let k1 = f32::new(0.25);
    let k2 = f32::new(0.40);
    let k3 = f32::new(0.25);
    let k4 = f32::new(0.05);

    let r0_i = if cy - 2 < 0 { -(cy - 2) - 1 } else { cy - 2 };
    let r0 = r0_i as usize;
    let r1_i = if cy - 1 < 0 { -(cy - 1) - 1 } else { cy - 1 };
    let r1 = r1_i as usize;
    let r2 = cy as usize;
    let r3_i = if cy + 1 >= sh_i {
        2 * sh_i - (cy + 1) - 1
    } else {
        cy + 1
    };
    let r3 = r3_i as usize;
    let r4_i = if cy + 2 >= sh_i {
        2 * sh_i - (cy + 2) - 1
    } else {
        cy + 2
    };
    let r4 = r4_i as usize;

    let sx0_i = if cx - 2 < 0 { -(cx - 2) - 1 } else { cx - 2 };
    let sx0 = sx0_i as usize;
    let sx1_i = if cx - 1 < 0 { -(cx - 1) - 1 } else { cx - 1 };
    let sx1 = sx1_i as usize;
    let sx2 = cx as usize;
    let sx3_i = if cx + 1 >= sw_i {
        2 * sw_i - (cx + 1) - 1
    } else {
        cx + 1
    };
    let sx3 = sx3_i as usize;
    let sx4_i = if cx + 2 >= sw_i {
        2 * sw_i - (cx + 2) - 1
    } else {
        cx + 2
    };
    let sx4 = sx4_i as usize;

    let col0 = k0 * src[r0 * sw + sx0]
        + k1 * src[r1 * sw + sx0]
        + k2 * src[r2 * sw + sx0]
        + k3 * src[r3 * sw + sx0]
        + k4 * src[r4 * sw + sx0];
    let col1 = k0 * src[r0 * sw + sx1]
        + k1 * src[r1 * sw + sx1]
        + k2 * src[r2 * sw + sx1]
        + k3 * src[r3 * sw + sx1]
        + k4 * src[r4 * sw + sx1];
    let col2 = k0 * src[r0 * sw + sx2]
        + k1 * src[r1 * sw + sx2]
        + k2 * src[r2 * sw + sx2]
        + k3 * src[r3 * sw + sx2]
        + k4 * src[r4 * sw + sx2];
    let col3 = k0 * src[r0 * sw + sx3]
        + k1 * src[r1 * sw + sx3]
        + k2 * src[r2 * sw + sx3]
        + k3 * src[r3 * sw + sx3]
        + k4 * src[r4 * sw + sx3];
    let col4 = k0 * src[r0 * sw + sx4]
        + k1 * src[r1 * sw + sx4]
        + k2 * src[r2 * sw + sx4]
        + k3 * src[r3 * sw + sx4]
        + k4 * src[r4 * sw + sx4];

    let mut total_v = k0 * col0 + k1 * col1 + k2 * col2 + k3 * col3 + k4 * col4;

    if dx == dw - 1 && sw >= 2 {
        let vs_last = k0 * src[r0 * sw + sw - 1]
            + k1 * src[r1 * sw + sw - 1]
            + k2 * src[r2 * sw + sw - 1]
            + k3 * src[r3 * sw + sw - 1]
            + k4 * src[r4 * sw + sw - 1];
        let vs_last2 = k0 * src[r0 * sw + sw - 2]
            + k1 * src[r1 * sw + sw - 2]
            + k2 * src[r2 * sw + sw - 2]
            + k3 * src[r3 * sw + sw - 2]
            + k4 * src[r4 * sw + sw - 2];

        let sw_odd = sw % 2 == 1;
        let sh_odd = sh % 2 == 1;
        if sw_odd && !sh_odd {
            total_v += f32::new(-0.05) * vs_last2 + f32::new(-0.20) * vs_last;
        } else if !sw_odd && sh_odd {
            total_v += f32::new(0.05) * vs_last2 + f32::new(0.20) * vs_last;
        }
    }

    dst[idx] = total_v;
}
