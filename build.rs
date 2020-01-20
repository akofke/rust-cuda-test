use ptx_builder::error::Result;
use ptx_builder::prelude::*;

fn main() -> Result<()> {
    println!("cargo:rustc-env=LIBRARY_PATH=/usr/local/cuda-10.1/lib64"); // for cuda-sys
    let builder = Builder::new("kernel")?;
    std::env::set_var("CARGO_BUILD_PIPELINING", "false");
    CargoAdapter::with_env_var("KERNEL_PTX_PATH").build(builder)
}