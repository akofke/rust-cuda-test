#![cfg_attr(target_os="cuda", no_std)]
#![cfg_attr(target_os="cuda", feature(abi_ptx, proc_macro_hygiene, stdsimd, core_intrinsics, asm))]

use num_complex::Complex64;

mod device;


pub fn mandelbrot_value(c: Complex64, max_iter: u32) -> u32 {
    let mut z = Complex64::new(0.0, 0.0);
    for i in 0..max_iter {
        z = z * z + c;
        if z.norm_sqr() > 4.0 {
            return i;
        }
    }
    max_iter
}

pub fn mandelbrot_value_fast(cr: f64, ci: f64, max_iter: u32) -> u32 {
    use core::intrinsics::*;
    let mut zr = 0.0;
    let mut zi = 0.0;
    let mut zrsqr = 0.0;
    let mut zisqr = 0.0;
    let mut i = 0i32;
    unsafe {
        while i < (max_iter as i32) {
//            zi = 2.0 * zr * zi + ci;
            zi = fadd_fast(fmul_fast(fmul_fast(2.0, zr), zi), ci);
//            zr = zrsqr - zisqr + cr;
            zr = fadd_fast(fsub_fast(zrsqr, zisqr), cr);
//            zrsqr = zr * zr;
            zrsqr = fmul_fast(zr, zr);
//            zisqr = zi * zi;
            zisqr = fmul_fast(zi, zi);
            if fadd_fast(zrsqr, zisqr) > 4.0 {
                break;
            }
            i = unchecked_add(i, 1);
        }
    }
//    (0..max_iter)
//        .fold((0.0, 0.0, 0.0, 0.0), |(zr, zi, zrsqr, zisqr), _i| {
//            unsafe {
////            zi = 2.0 * zr * zi + ci;
//                let zi = fadd_fast(fmul_fast(fmul_fast(2.0, zr), zi), ci);
////            zr = zrsqr - zisqr + cr;
//                let zr = fadd_fast(fsub_fast(zrsqr, zisqr), cr);
////            zrsqr = zr * zr;
//                let zrsqr = fmul_fast(zr, zr);
////            zisqr = zi * zi;
//                let zisqr = fmul_fast(zi, zi);
//                (zr, zi, zrsqr, zisqr)
//            }
//        })
//        .take_while()

    i as u32
}