use ptx_support::prelude;
use core::arch::nvptx::*;

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
pub unsafe extern "ptx-kernel" fn test_kernel(x: *const i32, out: *mut i32, len: usize) {

//    cuda_printf!(
//        "Hello from block(%lu,%lu,%lu) and thread(%lu,%lu,%lu)\n",
//        Context::block().index().x,
//        Context::block().index().y,
//        Context::block().index().z,
//        Context::thread().index().x,
//        Context::thread().index().y,
//        Context::thread().index().z,
//    );
   let x = core::slice::from_raw_parts(x, len);
   let mut out = core::slice::from_raw_parts_mut(out, len);
   let global_idx: usize = Context::block_idx().x as usize * Context::block_dim().x as usize + Context::thread_idx().x as usize;

   out[global_idx] = x[global_idx] * global_idx as i32;
    // let xval = *x.offset(global_idx as isize);
    // let outval = __sad(xval, global_idx as i32, 15);
    // *out.offset(global_idx as isize) = outval;
}
