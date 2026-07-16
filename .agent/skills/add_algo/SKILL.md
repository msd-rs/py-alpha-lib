---
name: add_algo
description: Add new algorithm to the library.
---


# Add New Algorithm

1. algorithm should be added to `algo` module in `alpha-algo` crate
2. algorithm function should has prefix `ta_`
3. numeric type should be generic as `NumT: Float + Send + Sync`
4. conditional type should be `bool` 
5. first argument should be `ctx: &Context`, handle `ctx.flags`
6. second argument should be `r: &mut [NumT]` or `r: &mut [bool]` by algorithm type, output buffer
7. implement template can be referenced from `ta_ma` from `algo/ma.rs`
  1. check input/output buffer length matches
  2. fast return if no calculation needed
  3. parallel calculation with `ctx.groups`
8. add document for the new algorithm
9. add test for the new algorithm

