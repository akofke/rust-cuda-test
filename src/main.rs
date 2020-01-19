

fn main() -> Result<(), Box<dyn std::error::Error>> {
    unsafe { host::main() }
}

mod host {
    use std::ffi::CString;

    use rustacuda::prelude::*;
    use rustacuda::launch;
    use rustacuda::memory::{DeviceBox, DeviceSlice, DevicePointer};

    use palette::{Srgb, Mix, Pixel};
    use anyhow::{Context as AnyhowContext};

    pub unsafe fn main() -> Result<(), Box<dyn std::error::Error>> {
        println!(env!("KERNEL_PTX_PATH"));
        let ptx_file = CString::new(include_str!(env!("KERNEL_PTX_PATH")))?;
        // println!("{}", ptx_file.clone().into_string()?);

        rustacuda::init(CudaFlags::empty())?;

        let device = Device::get_device(0)?;
//         println!("{}", device.name()?);

        let context = Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;

        let module = Module::load_from_string(&ptx_file)?;

        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        compute_mandelbrot((1024, 1024), &module, &stream)?;

        // let len = 10;
        // let data = vec![1i32; len];
        // let mut x = DeviceBuffer::from_slice(&data)?;
        // let mut out = DeviceBuffer::uninitialized(len)?;

        // unsafe {
        //     launch!(module.test_kernel<<<1u32, len as u32, 0, stream>>>(x.as_device_ptr(), out.as_device_ptr(), len))?;
        // }

        // stream.synchronize()?;
        // let mut out_host = vec![0; len];
        // out.copy_to(&mut out_host)?;
        // println!("{:?}", out_host);

        Ok(())
    }

    unsafe fn compute_mandelbrot(image_dims: (u32, u32), module: &Module, stream: &Stream) -> anyhow::Result<()> {
        let (w, h) = image_dims;
        let min_re = -2.0;
        let max_re = 0.7;
        let min_im = -1.2;
        let max_im = min_im + (max_re - min_re) * (h as f64 / w as f64);
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
}

