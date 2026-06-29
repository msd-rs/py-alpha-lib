pub fn register_ta_funcs(lua: &Lua) -> LuaResult<()> {
    lua.globals().set(
      "ALPHA",
      lua.create_function(|lua, (input, benchmark, periods): (NumArray, NumArray, usize)| {
        let mut r = vec![0.0; input.len()];
        ta_alpha::<f64>(&ctx(lua), &mut r, &input, &benchmark, periods)?;
        Ok(NumArray::from(r))
      })?,
    )?;
    lua.globals().set(
      "BACKFILL",
      lua.create_function(|lua, (input,): (NumArray,)| {
        let mut r = vec![0.0; input.len()];
        ta_backfill::<f64>(&ctx(lua), &mut r, &input)?;
        Ok(NumArray::from(r))
      })?,
    )?;
    lua.globals().set(
      "BARSLAST",
      lua.create_function(|lua, (input,): (BoolArray,)| {
        let mut r = vec![0.0; input.len()];
        ta_barslast::<f64>(&ctx(lua), &mut r, &input)?;
        Ok(NumArray::from(r))
      })?,
    )?;
    lua.globals().set(
      "BARSSINCE",
      lua.create_function(|lua, (input,): (BoolArray,)| {
        let mut r = vec![0.0; input.len()];
        ta_barssince::<f64>(&ctx(lua), &mut r, &input)?;
        Ok(NumArray::from(r))
      })?,
    )?;
    lua.globals().set(
      "BETA",
      lua.create_function(|lua, (input, benchmark, periods): (NumArray, NumArray, usize)| {
        let mut r = vec![0.0; input.len()];
        ta_beta::<f64>(&ctx(lua), &mut r, &input, &benchmark, periods)?;
        Ok(NumArray::from(r))
      })?,
    )?;
    lua.globals().set(
      "BINS",
      lua.create_function(|lua, (input, bins): (NumArray, usize)| {
        let mut r = vec![0.0; input.len()];
        ta_bins::<f64>(&ctx(lua), &mut r, &input, bins)?;
        Ok(NumArray::from(r))
      })?,
    )?;
    lua.globals().set(
      "BW_SPLIT",
      lua.create_function(|lua, (price, dividend, transfer_shares, right_shares, right_price): (NumArray, NumArray, NumArray, NumArray, NumArray)| {
        let mut r = vec![0.0; price.len()];
        ta_bw_split::<f64>(&ctx(lua), &mut r, &price, &dividend, &transfer_shares, &right_shares, &right_price)?;
        Ok(NumArray::from(r))
      })?,
    )?;
    lua.globals().set(
      "CC_RANK",
      lua.create_function(|lua, (input,): (NumArray,)| {
        let mut r = vec![0.0; input.len()];
        ta_cc_rank::<f64>(&ctx(lua), &mut r, &input)?;
        Ok(NumArray::from(r))
      })?,
    )?;
    lua.globals().set(
      "CC_ZSCORE",
      lua.create_function(|lua, (input,): (NumArray,)| {
        let mut r = vec![0.0; input.len()];
        ta_cc_zscore::<f64>(&ctx(lua), &mut r, &input)?;
        Ok(NumArray::from(r))
      })?,
    )?;
    lua.globals().set(
      "CORR",
      lua.create_function(|lua, (input, periods): (NumArray, usize)| {
        let mut r = vec![0.0; input.len()];
        ta_corr::<f64>(&ctx(lua), &mut r, &input, periods)?;
        Ok(NumArray::from(r))
      })?,
    )?;
    lua.globals().set(
      "CORR2",
      lua.create_function(|lua, (x, y, periods): (NumArray, NumArray, usize)| {
        let mut r = vec![0.0; x.len()];
        ta_corr2::<f64>(&ctx(lua), &mut r, &x, &y, periods)?;
        Ok(NumArray::from(r))
      })?,
    )?;
    lua.globals().set(
      "COUNT",
      lua.create_function(|lua, (input, periods): (BoolArray, usize)| {
        let mut r = vec![0.0; input.len()];
        ta_count::<f64>(&ctx(lua), &mut r, &input, periods)?;
        Ok(NumArray::from(r))
      })?,
    )?;
    lua.globals().set(
      "COUNT_NANS",
      lua.create_function(|lua, (input, periods): (NumArray, usize)| {
        let mut r = vec![0.0; input.len()];
        ta_count_nans::<f64>(&ctx(lua), &mut r, &input, periods)?;
        Ok(NumArray::from(r))
      })?,
    )?;
    lua.globals().set(
      "COV",
      lua.create_function(|lua, (x, y, periods): (NumArray, NumArray, usize)| {
        let mut r = vec![0.0; x.len()];
        ta_cov::<f64>(&ctx(lua), &mut r, &x, &y, periods)?;
        Ok(NumArray::from(r))
      })?,
    )?;
    lua.globals().set(
      "CROSS",
      lua.create_function(|lua, (a, b): (NumArray, NumArray)| {
        let mut r = vec![false; a.len()];
        ta_cross::<f64>(&ctx(lua), &mut r, &a, &b)?;
        Ok(BoolArray::from(r))
      })?,
    )?;
    lua.globals().set(
      "DMA",
      lua.create_function(|lua, (input, weight): (NumArray, f64)| {
        let mut r = vec![0.0; input.len()];
        ta_dma::<f64>(&ctx(lua), &mut r, &input, weight)?;
        Ok(NumArray::from(r))
      })?,
    )?;
    lua.globals().set(
      "EMA",
      lua.create_function(|lua, (input, periods): (NumArray, usize)| {
        let mut r = vec![0.0; input.len()];
        ta_ema::<f64>(&ctx(lua), &mut r, &input, periods)?;
        Ok(NumArray::from(r))
      })?,
    )?;
    lua.globals().set(
      "ENTROPY",
      lua.create_function(|lua, (input, periods, bins): (NumArray, usize, usize)| {
        let mut r = vec![0.0; input.len()];
        ta_entropy::<f64>(&ctx(lua), &mut r, &input, periods, bins)?;
        Ok(NumArray::from(r))
      })?,
    )?;
    lua.globals().set(
      "FRET",
      lua.create_function(|lua, (open, close, is_calc, delay, periods): (NumArray, NumArray, NumArray, usize, usize)| {
        let mut r = vec![0.0; open.len()];
        ta_fret::<f64>(&ctx(lua), &mut r, &open, &close, &is_calc, delay, periods)?;
        Ok(NumArray::from(r))
      })?,
    )?;
    lua.globals().set(
      "FW_SPLIT",
      lua.create_function(|lua, (price, dividend, transfer_shares, right_shares, right_price): (NumArray, NumArray, NumArray, NumArray, NumArray)| {
        let mut r = vec![0.0; price.len()];
        ta_fw_split::<f64>(&ctx(lua), &mut r, &price, &dividend, &transfer_shares, &right_shares, &right_price)?;
        Ok(NumArray::from(r))
      })?,
    )?;
    lua.globals().set(
      "GROUP_RANK",
      lua.create_function(|lua, (category, input): (NumArray, NumArray)| {
        let mut r = vec![0.0; category.len()];
        ta_group_rank::<f64>(&ctx(lua), &mut r, &category, &input)?;
        Ok(NumArray::from(r))
      })?,
    )?;
    lua.globals().set(
      "GROUP_ZSCORE",
      lua.create_function(|lua, (category, input): (NumArray, NumArray)| {
        let mut r = vec![0.0; category.len()];
        ta_group_zscore::<f64>(&ctx(lua), &mut r, &category, &input)?;
        Ok(NumArray::from(r))
      })?,
    )?;
    lua.globals().set(
      "HHV",
      lua.create_function(|lua, (input, periods): (NumArray, usize)| {
        let mut r = vec![0.0; input.len()];
        ta_hhv::<f64>(&ctx(lua), &mut r, &input, periods)?;
        Ok(NumArray::from(r))
      })?,
    )?;
    lua.globals().set(
      "HHVBARS",
      lua.create_function(|lua, (input, periods): (NumArray, usize)| {
        let mut r = vec![0.0; input.len()];
        ta_hhvbars::<f64>(&ctx(lua), &mut r, &input, periods)?;
        Ok(NumArray::from(r))
      })?,
    )?;
    lua.globals().set(
      "INTERCEPT",
      lua.create_function(|lua, (input, periods): (NumArray, usize)| {
        let mut r = vec![0.0; input.len()];
        ta_intercept::<f64>(&ctx(lua), &mut r, &input, periods)?;
        Ok(NumArray::from(r))
      })?,
    )?;
    lua.globals().set(
      "KURTOSIS",
      lua.create_function(|lua, (input, periods): (NumArray, usize)| {
        let mut r = vec![0.0; input.len()];
        ta_kurtosis::<f64>(&ctx(lua), &mut r, &input, periods)?;
        Ok(NumArray::from(r))
      })?,
    )?;
    lua.globals().set(
      "LLV",
      lua.create_function(|lua, (input, periods): (NumArray, usize)| {
        let mut r = vec![0.0; input.len()];
        ta_llv::<f64>(&ctx(lua), &mut r, &input, periods)?;
        Ok(NumArray::from(r))
      })?,
    )?;
    lua.globals().set(
      "LLVBARS",
      lua.create_function(|lua, (input, periods): (NumArray, usize)| {
        let mut r = vec![0.0; input.len()];
        ta_llvbars::<f64>(&ctx(lua), &mut r, &input, periods)?;
        Ok(NumArray::from(r))
      })?,
    )?;
    lua.globals().set(
      "LONGCROSS",
      lua.create_function(|lua, (a, b, n): (NumArray, NumArray, usize)| {
        let mut r = vec![false; a.len()];
        ta_longcross::<f64>(&ctx(lua), &mut r, &a, &b, n)?;
        Ok(BoolArray::from(r))
      })?,
    )?;
    lua.globals().set(
      "LWMA",
      lua.create_function(|lua, (input, periods): (NumArray, usize)| {
        let mut r = vec![0.0; input.len()];
        ta_lwma::<f64>(&ctx(lua), &mut r, &input, periods)?;
        Ok(NumArray::from(r))
      })?,
    )?;
    lua.globals().set(
      "MA",
      lua.create_function(|lua, (input, periods): (NumArray, usize)| {
        let mut r = vec![0.0; input.len()];
        ta_ma::<f64>(&ctx(lua), &mut r, &input, periods)?;
        Ok(NumArray::from(r))
      })?,
    )?;
    lua.globals().set(
      "MAX_DRAWDOWN",
      lua.create_function(|lua, (input, periods): (NumArray, usize)| {
        let mut r = vec![0.0; input.len()];
        ta_max_drawdown::<f64>(&ctx(lua), &mut r, &input, periods)?;
        Ok(NumArray::from(r))
      })?,
    )?;
    lua.globals().set(
      "MIN_MAX_DIFF",
      lua.create_function(|lua, (input, periods): (NumArray, usize)| {
        let mut r = vec![0.0; input.len()];
        ta_min_max_diff::<f64>(&ctx(lua), &mut r, &input, periods)?;
        Ok(NumArray::from(r))
      })?,
    )?;
    lua.globals().set(
      "MOMENT",
      lua.create_function(|lua, (input, periods, k): (NumArray, usize, usize)| {
        let mut r = vec![0.0; input.len()];
        ta_moment::<f64>(&ctx(lua), &mut r, &input, periods, k)?;
        Ok(NumArray::from(r))
      })?,
    )?;
    lua.globals().set(
      "NEUTRALIZE",
      lua.create_function(|lua, (category, input): (NumArray, NumArray)| {
        let mut r = vec![0.0; category.len()];
        ta_neutralize::<f64>(&ctx(lua), &mut r, &category, &input)?;
        Ok(NumArray::from(r))
      })?,
    )?;
    lua.globals().set(
      "PRODUCT",
      lua.create_function(|lua, (input, periods): (NumArray, usize)| {
        let mut r = vec![0.0; input.len()];
        ta_product::<f64>(&ctx(lua), &mut r, &input, periods)?;
        Ok(NumArray::from(r))
      })?,
    )?;
    lua.globals().set(
      "QUANTILE",
      lua.create_function(|lua, (input, periods, q): (NumArray, usize, f64)| {
        let mut r = vec![0.0; input.len()];
        ta_quantile::<f64>(&ctx(lua), &mut r, &input, periods, q)?;
        Ok(NumArray::from(r))
      })?,
    )?;
    lua.globals().set(
      "RANK",
      lua.create_function(|lua, (input, periods): (NumArray, usize)| {
        let mut r = vec![0.0; input.len()];
        ta_rank::<f64>(&ctx(lua), &mut r, &input, periods)?;
        Ok(NumArray::from(r))
      })?,
    )?;
    lua.globals().set(
      "RCROSS",
      lua.create_function(|lua, (a, b): (NumArray, NumArray)| {
        let mut r = vec![false; a.len()];
        ta_rcross::<f64>(&ctx(lua), &mut r, &a, &b)?;
        Ok(BoolArray::from(r))
      })?,
    )?;
    lua.globals().set(
      "REF",
      lua.create_function(|lua, (input, periods): (NumArray, usize)| {
        let mut r = vec![0.0; input.len()];
        ta_ref::<f64>(&ctx(lua), &mut r, &input, periods)?;
        Ok(NumArray::from(r))
      })?,
    )?;
    lua.globals().set(
      "REGBETA",
      lua.create_function(|lua, (y, x, periods): (NumArray, NumArray, usize)| {
        let mut r = vec![0.0; y.len()];
        ta_regbeta::<f64>(&ctx(lua), &mut r, &y, &x, periods)?;
        Ok(NumArray::from(r))
      })?,
    )?;
    lua.globals().set(
      "REGRESI",
      lua.create_function(|lua, (y, x, periods): (NumArray, NumArray, usize)| {
        let mut r = vec![0.0; y.len()];
        ta_regresi::<f64>(&ctx(lua), &mut r, &y, &x, periods)?;
        Ok(NumArray::from(r))
      })?,
    )?;
    lua.globals().set(
      "RLONGCROSS",
      lua.create_function(|lua, (a, b, n): (NumArray, NumArray, usize)| {
        let mut r = vec![false; a.len()];
        ta_rlongcross::<f64>(&ctx(lua), &mut r, &a, &b, n)?;
        Ok(BoolArray::from(r))
      })?,
    )?;
    lua.globals().set(
      "SCAN_ADD",
      lua.create_function(|lua, (input, condition): (NumArray, BoolArray)| {
        let mut r = vec![0.0; input.len()];
        ta_scan_add::<f64>(&ctx(lua), &mut r, &input, &condition)?;
        Ok(NumArray::from(r))
      })?,
    )?;
    lua.globals().set(
      "SCAN_MUL",
      lua.create_function(|lua, (input, condition): (NumArray, BoolArray)| {
        let mut r = vec![0.0; input.len()];
        ta_scan_mul::<f64>(&ctx(lua), &mut r, &input, &condition)?;
        Ok(NumArray::from(r))
      })?,
    )?;
    lua.globals().set(
      "SHARPE",
      lua.create_function(|lua, (input, periods): (NumArray, usize)| {
        let mut r = vec![0.0; input.len()];
        ta_sharpe::<f64>(&ctx(lua), &mut r, &input, periods)?;
        Ok(NumArray::from(r))
      })?,
    )?;
    lua.globals().set(
      "SKEWNESS",
      lua.create_function(|lua, (input, periods): (NumArray, usize)| {
        let mut r = vec![0.0; input.len()];
        ta_skewness::<f64>(&ctx(lua), &mut r, &input, periods)?;
        Ok(NumArray::from(r))
      })?,
    )?;
    lua.globals().set(
      "SLOPE",
      lua.create_function(|lua, (input, periods): (NumArray, usize)| {
        let mut r = vec![0.0; input.len()];
        ta_slope::<f64>(&ctx(lua), &mut r, &input, periods)?;
        Ok(NumArray::from(r))
      })?,
    )?;
    lua.globals().set(
      "SMA",
      lua.create_function(|lua, (input, n, m): (NumArray, usize, usize)| {
        let mut r = vec![0.0; input.len()];
        ta_sma::<f64>(&ctx(lua), &mut r, &input, n, m)?;
        Ok(NumArray::from(r))
      })?,
    )?;
    lua.globals().set(
      "STDDEV",
      lua.create_function(|lua, (input, periods): (NumArray, usize)| {
        let mut r = vec![0.0; input.len()];
        ta_stddev::<f64>(&ctx(lua), &mut r, &input, periods)?;
        Ok(NumArray::from(r))
      })?,
    )?;
    lua.globals().set(
      "SUM",
      lua.create_function(|lua, (input, periods): (NumArray, usize)| {
        let mut r = vec![0.0; input.len()];
        ta_sum::<f64>(&ctx(lua), &mut r, &input, periods)?;
        Ok(NumArray::from(r))
      })?,
    )?;
    lua.globals().set(
      "SUMBARS",
      lua.create_function(|lua, (input, amount): (NumArray, f64)| {
        let mut r = vec![0.0; input.len()];
        ta_sumbars::<f64>(&ctx(lua), &mut r, &input, amount)?;
        Ok(NumArray::from(r))
      })?,
    )?;
    lua.globals().set(
      "SUMIF",
      lua.create_function(|lua, (input, condition, periods): (NumArray, BoolArray, usize)| {
        let mut r = vec![0.0; input.len()];
        ta_sumif::<f64>(&ctx(lua), &mut r, &input, &condition, periods)?;
        Ok(NumArray::from(r))
      })?,
    )?;
    lua.globals().set(
      "VAR",
      lua.create_function(|lua, (input, periods): (NumArray, usize)| {
        let mut r = vec![0.0; input.len()];
        ta_var::<f64>(&ctx(lua), &mut r, &input, periods)?;
        Ok(NumArray::from(r))
      })?,
    )?;
    lua.globals().set(
      "WEIGHTED_DELAY",
      lua.create_function(|lua, (input, periods): (NumArray, usize)| {
        let mut r = vec![0.0; input.len()];
        ta_weighted_delay::<f64>(&ctx(lua), &mut r, &input, periods)?;
        Ok(NumArray::from(r))
      })?,
    )?;
    lua.globals().set(
      "ZSCORE",
      lua.create_function(|lua, (input, periods): (NumArray, usize)| {
        let mut r = vec![0.0; input.len()];
        ta_zscore::<f64>(&ctx(lua), &mut r, &input, periods)?;
        Ok(NumArray::from(r))
      })?,
    )?;
  Ok(())
}
