#![cfg_attr(target_os="cuda", no_std)]
#![cfg_attr(target_os="cuda", feature(abi_ptx, proc_macro_hygiene, stdsimd, core_intrinsics, asm))]


mod kernel;

#[cfg(not(target_os = "cuda"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    unsafe { host::main() }
}

#[cfg(not(target_os = "cuda"))]
mod host {
    use std::ffi::CString;

    use rustacuda::prelude::*;
    use rustacuda::launch;
    use rustacuda::memory::{DeviceBox, DeviceSlice, DevicePointer};

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

        let len = 10;
        let data = vec![1i32; len];
        let mut x = DeviceBuffer::from_slice(&data)?;
        let mut out = DeviceBuffer::uninitialized(len)?;

        unsafe {
            launch!(module.test_kernel<<<1u32, len as u32, 0, stream>>>(x.as_device_ptr(), out.as_device_ptr(), len))?;
        }

        stream.synchronize()?;
        let mut out_host = vec![0; len];
        out.copy_to(&mut out_host)?;
        println!("{:?}", out_host);

        Ok(())
    }
}

