use rustacuda::stream::Stream;
use rustacuda::function::Function;
use rustacuda::memory::{DeviceBuffer, CopyDestination};
use num_complex::Complex64;
use std::ffi::CString;
use rustacuda::prelude::*;
use rustacuda::launch;
use rustacuda::memory::{DeviceBox, DeviceSlice, DevicePointer};
use palette::{Srgb, Mix, Pixel, LinSrgb};
use anyhow::{Context as AnyhowContext};
use minifb::{Window, WindowOptions, Scale, MouseMode, MouseButton};
use std::time::Duration;
use palette::rgb::Rgb;
use palette::encoding::Linear;

fn pack_color(c: [u8; 3]) -> u32 {
    let [r, g, b] = c;
    ((r as u32) << 16) | ((g as u32) << 8) | b as u32
}

fn unpack_color(c: u32) -> [u8; 3] {
    [(c >> 16) as u8, (c >> 8) as u8, c as u8]
}

const WIDTH: usize = 512;
const HEIGHT: usize = 512;

const MIN_RE: f64 = -2.0;
const MAX_RE: f64 = 0.7;
const MIN_IM: f64 = -1.2;
const MAX_IM: f64 = MIN_IM + (MAX_RE - MIN_RE) * (HEIGHT as f64 / WIDTH as f64);

pub fn main() -> anyhow::Result<()> {
    println!(env!("KERNEL_PTX_PATH"));
    let ptx_file = CString::new(include_str!(env!("KERNEL_PTX_PATH")))?;
    // println!("{}", ptx_file.clone().into_string()?);

    rustacuda::init(CudaFlags::empty())?;

    let device = Device::get_device(0)?;
//         println!("{}", device.name()?);

    let context = Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;

    let module = Module::load_from_string(&ptx_file)?;

    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    let function = module.get_function(&CString::new("mandelbrot_kernel_colors")?)?;
    let julia_function = module.get_function(&CString::new("julia_set")?)?;



    let mut window = Window::new(
        "Mandelbrot",
        WIDTH,
        HEIGHT,
        WindowOptions {
            resize: false,
            scale: Scale::X2,
            ..Default::default()
        }
    )?;

    let mut julia_window = Window::new(
        "Julia Set",
        WIDTH,
        HEIGHT,
        WindowOptions {
            resize: false,
            scale: Scale::X2,
            ..Default::default()
        }
    )?;

    window.limit_update_rate(Some(Duration::from_millis(32)));
    julia_window.limit_update_rate(Some(Duration::from_millis(32)));


    let device_buffer = unsafe { DeviceBuffer::zeroed(WIDTH * HEIGHT) }?;
    let device_buffer_julia = unsafe { DeviceBuffer::zeroed(WIDTH * HEIGHT) }?;
    let mut kernel = MandelbrotKernel {
        stream: &stream,
        function: &function,
        julia_function: &julia_function,
        device_buffer,
        device_buffer_julia,
        host_buffer: vec![0; WIDTH * HEIGHT],
        host_buffer_julia: vec![0; WIDTH * HEIGHT],
    };

    let mut min_re = MIN_RE;
    let mut max_re = MAX_RE;
    let mut min_im = MIN_IM;
    let mut max_im = MAX_IM;
    let max_iter = 1024;

    let colors = make_colors(max_iter);
    let mut device_colors = DeviceBuffer::from_slice(&colors)?;

    window.update_with_buffer(&kernel.host_buffer, WIDTH, HEIGHT)?;
    let mut mouse_coords = (0.0, 0.0);
    let mut need_update = true;
    while window.is_open() {
        let mouse_pos = window.get_mouse_pos(MouseMode::Discard);
        let unscaled_mouse_pos = window.get_unscaled_mouse_pos(MouseMode::Discard);
        let mouse_down = window.get_mouse_down(MouseButton::Left);

        let (w, h) = window.get_size();
        let scale_fac = 0.9;

        if let Some((x, y)) = unscaled_mouse_pos {
            let frac_x = x as f64 / w as f64;
            let frac_y = 1.0 - (y as f64 / h as f64);
            let point_re = min_re + (max_re - min_re) * frac_x;
            let point_im = min_im + (max_im - min_im) * frac_y;
            mouse_coords = (point_re, point_im);

            if mouse_down {
                min_re += (1.0 - scale_fac) * (point_re - min_re);
                max_re -= (1.0 - scale_fac) * (max_re - point_re);
                min_im += (1.0 - scale_fac) * (point_im - min_im);
                max_im -= (1.0 - scale_fac) * (max_im - point_im);
            }
            need_update = true;
        }


        if need_update {
            kernel.compute_image(
                (w as u32, h as u32),
                (Complex64::new(min_re, min_im), Complex64::new(max_re, max_im)),
                mouse_coords,
                &mut device_colors,
                max_iter,
            )?;

            window.update_with_buffer(&kernel.host_buffer, w, h)?;
            julia_window.update_with_buffer(&kernel.host_buffer_julia, w, h)?;
            need_update = false;
        } else {
            window.update();
        }
    }

    Ok(())
}

unsafe fn compute_mandelbrot(image_dims: (u32, u32), bounds: (Complex64, Complex64), module: &Module, stream: &Stream) -> anyhow::Result<()> {
    let (w, h) = image_dims;
    let (min, max) = bounds;
    let min_re = min.re;
    let max_re = max.re;
    let min_im = min.im;
    let max_im = max.im;
//        let max_im = min_im + (max_re - min_re) * (h as f64 / w as f64);
    dbg!(max_im);

    let re_step = (max_re - min_re) / (w as f64 - 1.0);
    let im_step = (max_im - min_im) / (h as f64 - 1.0);

    let max_iter: u32 = 255;

    let mut out = DeviceBuffer::<u32>::zeroed(w as usize * h as usize)?;

    let block_size = (32, 32);
    let grid_w = (w - 1) / block_size.0 + 1;
    let grid_h = (h - 1) / block_size.1 + 1;

    let start = std::time::Instant::now();
    unsafe {
        launch!(module.mandelbrot_kernel<<<(grid_w, grid_h), block_size, 0, stream>>>(
                w,
                h,
                min_re,
                max_im,
                re_step,
                im_step,
                max_iter,
                out.as_device_ptr()
            )).context("failed to launch kernel")?
    }

    stream.synchronize()?;
    let end = start.elapsed();
    println!("Kernel completed in {} ms", end.as_millis());
    let start = std::time::Instant::now();
    let mut out_host = vec![0; w as usize * h as usize];
    out.copy_to(&mut out_host)?;
    println!("Got array back to host, len {} in {} ms", out_host.len(), start.elapsed().as_millis());
    let start = std::time::Instant::now();
    let mut colors: Vec<u8> = Vec::with_capacity(out_host.len() * 3);

    for val in out_host {
        let orangeish = Srgb::new(1.0, 0.6, 0.0).into_linear();
        let blueish = Srgb::new(0.0, 0.2, 1.0).into_linear();
        let fac = val as f64 / max_iter as f64;
        let mixed = blueish.mix(&orangeish, fac);
        let pixel: [u8; 3] = Srgb::from_linear(mixed).into_format().into_raw();
        colors.extend_from_slice(&pixel);
    }
    println!("Computed colors in {} ms", start.elapsed().as_millis());

    let file = std::fs::File::create("mandelbrot.png")?;
    let writer = image::png::PNGEncoder::new(file);
    writer.encode(
        &colors,
        w,
        h,
        image::ColorType::Rgb8,
    )?;

    Ok(())

}

pub fn map_colors(buf: &mut [u32], max_iter: u32) {
    for color in buf.iter_mut() {
        if *color == max_iter {
            *color = 0; // black
        } else {
            let orangeish = Srgb::new(1.0, 0.6, 0.0).into_linear();
            let blueish = Srgb::new(0.0, 0.2, 1.0).into_linear();
            let fac = *color as f64 / max_iter as f64;
            let mixed = blueish.mix(&orangeish, fac);
            let pixel: [u8; 3] = Srgb::from_linear(mixed).into_format().into_raw();
            *color = pack_color(pixel);
        }
    }
}

fn make_colors(max_iter: u32) -> Vec<u32> {
    let orangeish = Srgb::new(1.0, 0.6, 0.0).into_linear();
    let blueish = Srgb::new(0.0, 0.2, 1.0).into_linear();
    (0..max_iter)
        .map(|n| {
            let fac = n as f64 / max_iter as f64;
            let mixed = blueish.mix(&orangeish, fac);
            let color: [u8; 3] = Srgb::from_linear(mixed).into_format().into_raw();
            pack_color(color)
        })
        .collect()
}

struct MandelbrotKernel<'a> {
    stream: &'a Stream,
    function: &'a Function<'a>,
    julia_function: &'a Function<'a>,
    device_buffer: DeviceBuffer<u32>,
    device_buffer_julia: DeviceBuffer<u32>,
    host_buffer: Vec<u32>,
    host_buffer_julia: Vec<u32>,
}

impl<'a> MandelbrotKernel<'a> {
    fn resize(&mut self, image_dims: (u32, u32)) -> anyhow::Result<()> {
        if image_dims.0 as usize * image_dims.1 as usize != self.device_buffer.len() {
            let new_buf = unsafe { DeviceBuffer::zeroed(image_dims.0 as usize * image_dims.1 as usize) }?;
            self.device_buffer = new_buf;
            self.device_buffer_julia = unsafe { DeviceBuffer::zeroed(image_dims.0 as usize * image_dims.1 as usize) }?;
            self.host_buffer = vec![0; image_dims.0 as usize * image_dims.1 as usize];
            self.host_buffer_julia = vec![0; image_dims.0 as usize * image_dims.1 as usize];
        }
        Ok(())
    }

    fn compute_image(&mut self,
                     image_dims: (u32, u32),
                     bounds: (Complex64, Complex64),
                     julia_coords: (f64, f64),
                     colors: &mut DeviceBuffer<u32>,
                     max_iter: u32
    ) -> anyhow::Result<()> {
        self.resize(image_dims)?;
        let (w, h) = image_dims;
        let (min, max) = bounds;
        let min_re = min.re;
        let max_re = max.re;
        let min_im = min.im;
        let max_im = max.im;
//        let max_im = min_im + (max_re - min_re) * (h as f64 / w as f64);

        let re_step = (max_re - min_re) / (w as f64 - 1.0);
        let im_step = (max_im - min_im) / (h as f64 - 1.0);

        let min_re_julia = MIN_RE;
        let max_im_julia = MAX_IM;
        let re_step_julia = (MAX_RE - MIN_RE) / (WIDTH as f64 - 1.0);
        let im_step_julia = (MAX_IM - MIN_IM) / (HEIGHT as f64 - 1.0);

        let block_size = (32, 32);
        let grid_w = (w - 1) / block_size.0 + 1;
        let grid_h = (h - 1) / block_size.1 + 1;

        let start = std::time::Instant::now();

        let stream = self.stream;
        let function = self.function;
        let julia_function = self.julia_function;

        unsafe {
            launch!(function<<<(grid_w, grid_h), block_size, 0, stream>>>(
                w,
                h,
                min_re,
                max_im,
                re_step,
                im_step,
                max_iter,
                colors.as_device_ptr(),
                self.device_buffer.as_device_ptr()
            )).context("failed to launch kernel")?
        }

        unsafe {
            launch!(julia_function<<<(grid_w, grid_h), block_size, 0, stream>>>(
                w,
                h,
                min_re_julia,
                max_im_julia,
                re_step_julia,
                im_step_julia,
                julia_coords.0,
                julia_coords.1,
                max_iter,
                colors.as_device_ptr(),
                self.device_buffer_julia.as_device_ptr()
            )).context("failed to launch kernel")?
        }

        stream.synchronize()?;
        let end = start.elapsed();
        println!("Kernel completed in {} ms", end.as_millis());
//        let start = std::time::Instant::now();
        self.device_buffer.copy_to(&mut self.host_buffer)?;
        self.device_buffer_julia.copy_to(&mut self.host_buffer_julia)?;
//        println!("Got array back to host, len {} in {} ms", self.host_buffer.len(), start.elapsed().as_millis());

        Ok(())
    }
}

