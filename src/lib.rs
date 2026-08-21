//! A WebAssembly binary parser and fuel-metered stack interpreter, in safe
//! Rust with zero dependencies. See the crate's README for the full pitch
//! and the supported instruction subset.

#![forbid(unsafe_code)]

pub mod leb;
pub mod parse;
pub mod types;

pub use types::{FuncType, Val, ValType};
