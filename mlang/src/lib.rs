pub mod numarray;
pub mod runtime;
mod to_lua;

#[cfg(test)]
mod tests;

use pest_derive::Parser;

#[derive(Parser)]
#[grammar = "mlang.pest"]
pub struct MLangParser;

pub use alpha_algo::Context;
pub use numarray::{BoolArray, NumArray};
pub use runtime::{Line, MRuntime, MValue};
pub use to_lua::to_lua;
