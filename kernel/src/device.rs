use ptx_support::prelude::*;
use core::arch::nvptx::*;
use num_complex::Complex64;
use palette::{LinSrgb, Mix, Srgb, Pixel};

pub struct Context {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl Context {
    pub fn block_dim() -> Self {
        Self {
            x: unsafe { core::arch::nvptx::_block_dim_x() } as u32,
            y: unsafe { core::arch::nvptx::_block_dim_y() } as u32,
            z: unsafe { core::arch::nvptx::_block_dim_z() } as u32,
        }
    }

    pub fn block_idx() -> Self {
        Self {
            x: unsafe { core::arch::nvptx::_block_idx_x() } as u32,
            y: unsafe { core::arch::nvptx::_block_idx_y() } as u32,
            z: unsafe { core::arch::nvptx::_block_idx_z() } as u32,
        }
    }

    pub fn thread_idx() -> Self {
        Self {
            x: unsafe { core::arch::nvptx::_thread_idx_x() } as u32,
            y: unsafe { core::arch::nvptx::_thread_idx_y() } as u32,
            z: unsafe { core::arch::nvptx::_thread_idx_z() } as u32,
        }
    }
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn mandelbrot_kernel(
    w: u32,
    h: u32,
    min_re: f64,
    max_im: f64,
    re_step: f64,
    im_step: f64,
    max_iter: u32,
    out: *mut u32,
) {
    let x = Context::block_idx().x * Context::block_dim().x + Context::thread_idx().x;
    let y = Context::block_idx().y * Context::block_dim().y + Context::thread_idx().y;

    if x >= w || y >= h {
        return
    }

    let c_re = min_re + (x as f64) * re_step;
    let c_im = max_im - (y as f64) * im_step;

//    let c = Complex64::new(c_re, c_im);
//    let value = crate::mandelbrot_value(c, max_iter);
    let value = crate::mandelbrot_value_fast(c_re, c_im, max_iter);

    let idx = y * w + x;

    out.add(idx as usize).write(value);
}

fn pack_color(c: [u8; 3]) -> u32 {
    let [r, g, b] = c;
    ((r as u32) << 16) | ((g as u32) << 8) | b as u32
}

fn unpack_color(c: u32) -> [u8; 3] {
    [(c >> 16) as u8, (c >> 8) as u8, c as u8]
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn mandelbrot_kernel_colors(
    w: u32,
    h: u32,
    min_re: f64,
    max_im: f64,
    re_step: f64,
    im_step: f64,
    max_iter: u32,
    colors: *const u32,
    out: *mut u32,
) {
    let x = Context::block_idx().x * Context::block_dim().x + Context::thread_idx().x;
    let y = Context::block_idx().y * Context::block_dim().y + Context::thread_idx().y;

    if x >= w || y >= h {
        return
    }

    let c_re = min_re + (x as f64) * re_step;
    let c_im = max_im - (y as f64) * im_step;

    let value = crate::mandelbrot_value_fast(c_re, c_im, max_iter);

    let color = if value == max_iter {
        0
    } else {
        *colors.add(value as usize)
    };

    let idx = y * w + x;

    out.add(idx as usize).write(color);
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn julia_set(
    w: u32,
    h: u32,
    min_re: f64,
    max_im: f64,
    re_step: f64,
    im_step: f64,
    k_re: f64,
    k_im: f64,
    max_iter: u32,
    colors: *const u32,
    out: *mut u32,
) {
    let x = Context::block_idx().x * Context::block_dim().x + Context::thread_idx().x;
    let y = Context::block_idx().y * Context::block_dim().y + Context::thread_idx().y;

    if x >= w || y >= h {
        return
    }

    let c_re = min_re + (x as f64) * re_step;
    let c_im = max_im - (y as f64) * im_step;

    let value = crate::julia_value(c_re, c_im, k_re, k_im, max_iter);

    let color = if value == max_iter {
        0
    } else {
        *colors.add(value as usize)
    };

    let idx = y * w + x;

    out.add(idx as usize).write(color);
}
