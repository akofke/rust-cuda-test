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