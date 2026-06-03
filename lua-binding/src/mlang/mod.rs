pub mod runtime;
mod to_lua;

use pest_derive::Parser;

#[derive(Parser)]
#[grammar = "mlang/mlang.pest"]
pub struct MLangParser;

pub use to_lua::to_lua;
pub use runtime::{MValue, MRuntime};
