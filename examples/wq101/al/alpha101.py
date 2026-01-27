import numpy as np


# (rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) -0.5)
def alpha_001(ctx):
  _RETURNS = ctx("RETURNS")
  return (
    ctx.RANK(
      ctx.TS_ARGMAX(
        ctx.SIGNEDPOWER(
          np.where(_RETURNS < 0, ctx.STDDEV(_RETURNS, 20), ctx("CLOSE")), 2.0
        ),
        5,
      )
    )
    - 0.5
  )


# (-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))
def alpha_002(ctx):
  _OPEN = ctx("OPEN")
  return -1 * ctx.CORRELATION(
    ctx.RANK(ctx.DELTA(ctx.LOG(ctx("VOLUME")), 2)),
    ctx.RANK(ctx("CLOSE") - _OPEN / _OPEN),
    6,
  )


# (-1 * correlation(rank(open), rank(volume), 10))
def alpha_003(ctx):
  return -1 * ctx.CORRELATION(ctx.RANK(ctx("OPEN")), ctx.RANK(ctx("VOLUME")), 10)


# (-1 * Ts_Rank(rank(low), 9))
def alpha_004(ctx):
  return -1 * ctx.TS_RANK(ctx.RANK(ctx("LOW")), 9)


# (rank((open - (sum(vwap, 10) / 10))) * (-1 * abs(rank((close - vwap)))))
def alpha_005(ctx):
  _VWAP = ctx("VWAP")
  return (
    ctx.RANK(ctx("OPEN") - ctx.SUM(_VWAP, 10) / 10)
    * -1
    * ctx.ABS(ctx.RANK(ctx("CLOSE") - _VWAP))
  )


# (-1 * correlation(open, volume, 10))
def alpha_006(ctx):
  return -1 * ctx.CORRELATION(ctx("OPEN"), ctx("VOLUME"), 10)


# ((adv20 < volume) ? ((-1 * ts_rank(abs(delta(close, 7)), 60)) * sign(delta(close, 7))) : (-1* 1))
def alpha_007(ctx):
  _CLOSE = ctx("CLOSE")
  return np.where(
    ctx("ADV20") < ctx("VOLUME"),
    -1
    * ctx.TS_RANK(ctx.ABS(ctx.DELTA(_CLOSE, 7)), 60)
    * ctx.SIGN(ctx.DELTA(_CLOSE, 7)),
    -1 * 1,
  )


# (-1 * rank(((sum(open, 5) * sum(returns, 5)) - delay((sum(open, 5) * sum(returns, 5)),10))))
def alpha_008(ctx):
  _OPEN = ctx("OPEN")
  _RETURNS = ctx("RETURNS")
  return -1 * ctx.RANK(
    ctx.SUM(_OPEN, 5) * ctx.SUM(_RETURNS, 5)
    - ctx.DELAY(ctx.SUM(_OPEN, 5) * ctx.SUM(_RETURNS, 5), 10)
  )


# ((0 < ts_min(delta(close, 1), 5)) ? delta(close, 1) : ((ts_max(delta(close, 1), 5) < 0) ?delta(close, 1) : (-1 * delta(close, 1))))
def alpha_009(ctx):
  _CLOSE = ctx("CLOSE")
  return np.where(
    0 < ctx.TS_MIN(ctx.DELTA(_CLOSE, 1), 5),
    ctx.DELTA(_CLOSE, 1),
    np.where(
      ctx.TS_MAX(ctx.DELTA(_CLOSE, 1), 5) < 0,
      ctx.DELTA(_CLOSE, 1),
      -1 * ctx.DELTA(_CLOSE, 1),
    ),
  )


# rank(((0 < ts_min(delta(close, 1), 4)) ? delta(close, 1) : ((ts_max(delta(close, 1), 4) < 0)? delta(close, 1) : (-1 * delta(close, 1)))))
def alpha_010(ctx):
  _CLOSE = ctx("CLOSE")
  return ctx.RANK(
    np.where(
      0 < ctx.TS_MIN(ctx.DELTA(_CLOSE, 1), 4),
      ctx.DELTA(_CLOSE, 1),
      np.where(
        ctx.TS_MAX(ctx.DELTA(_CLOSE, 1), 4) < 0,
        ctx.DELTA(_CLOSE, 1),
        -1 * ctx.DELTA(_CLOSE, 1),
      ),
    )
  )


# ((rank(ts_max((vwap - close), 3)) + rank(ts_min((vwap - close), 3))) *rank(delta(volume, 3)))
def alpha_011(ctx):
  _CLOSE = ctx("CLOSE")
  _VWAP = ctx("VWAP")
  return ctx.RANK(ctx.TS_MAX(_VWAP - _CLOSE, 3)) + ctx.RANK(
    ctx.TS_MIN(_VWAP - _CLOSE, 3)
  ) * ctx.RANK(ctx.DELTA(ctx("VOLUME"), 3))


# (sign(delta(volume, 1)) * (-1 * delta(close, 1)))
def alpha_012(ctx):
  return ctx.SIGN(ctx.DELTA(ctx("VOLUME"), 1)) * -1 * ctx.DELTA(ctx("CLOSE"), 1)


# (-1 * rank(covariance(rank(close), rank(volume), 5)))
def alpha_013(ctx):
  return -1 * ctx.RANK(
    ctx.COVARIANCE(ctx.RANK(ctx("CLOSE")), ctx.RANK(ctx("VOLUME")), 5)
  )


# ((-1 * rank(delta(returns, 3))) * correlation(open, volume, 10))
def alpha_014(ctx):
  return (
    -1
    * ctx.RANK(ctx.DELTA(ctx("RETURNS"), 3))
    * ctx.CORRELATION(ctx("OPEN"), ctx("VOLUME"), 10)
  )


# (-1 * sum(rank(correlation(rank(high), rank(volume), 3)), 3))
def alpha_015(ctx):
  return -1 * ctx.SUM(
    ctx.RANK(ctx.CORRELATION(ctx.RANK(ctx("HIGH")), ctx.RANK(ctx("VOLUME")), 3)), 3
  )


# (-1 * rank(covariance(rank(high), rank(volume), 5)))
def alpha_016(ctx):
  return -1 * ctx.RANK(
    ctx.COVARIANCE(ctx.RANK(ctx("HIGH")), ctx.RANK(ctx("VOLUME")), 5)
  )


# (((-1 * rank(ts_rank(close, 10))) * rank(delta(delta(close, 1), 1))) *rank(ts_rank((volume / adv20), 5)))
def alpha_017(ctx):
  _CLOSE = ctx("CLOSE")
  return (
    -1
    * ctx.RANK(ctx.TS_RANK(_CLOSE, 10))
    * ctx.RANK(ctx.DELTA(ctx.DELTA(_CLOSE, 1), 1))
    * ctx.RANK(ctx.TS_RANK(ctx("VOLUME") / ctx("ADV20"), 5))
  )


# (-1 * rank(((stddev(abs((close - open)), 5) + (close - open)) + correlation(close, open,10))))
def alpha_018(ctx):
  _CLOSE = ctx("CLOSE")
  _OPEN = ctx("OPEN")
  return -1 * ctx.RANK(
    ctx.STDDEV(ctx.ABS(_CLOSE - _OPEN), 5)
    + _CLOSE
    - _OPEN
    + ctx.CORRELATION(_CLOSE, _OPEN, 10)
  )


# ((-1 * sign(((close - delay(close, 7)) + delta(close, 7)))) * (1 + rank((1 + sum(returns,250)))))
def alpha_019(ctx):
  _CLOSE = ctx("CLOSE")
  return -1 * ctx.SIGN(
    _CLOSE - ctx.DELAY(_CLOSE, 7) + ctx.DELTA(_CLOSE, 7)
  ) * 1 + ctx.RANK(1 + ctx.SUM(ctx("RETURNS"), 250))


# (((-1 * rank((open - delay(high, 1)))) * rank((open - delay(close, 1)))) * rank((open -delay(low, 1))))
def alpha_020(ctx):
  _OPEN = ctx("OPEN")
  return (
    -1
    * ctx.RANK(_OPEN - ctx.DELAY(ctx("HIGH"), 1))
    * ctx.RANK(_OPEN - ctx.DELAY(ctx("CLOSE"), 1))
    * ctx.RANK(_OPEN - ctx.DELAY(ctx("LOW"), 1))
  )


# ((((sum(close, 8) / 8) + stddev(close, 8)) < (sum(close, 2) / 2)) ? (-1 * 1) : (((sum(close,2) / 2) < ((sum(close, 8) / 8) - stddev(close, 8))) ? 1 : (((1 < (volume / adv20)) || ((volume /adv20) == 1)) ? 1 : (-1 * 1))))
def alpha_021(ctx):
  _ADV20 = ctx("ADV20")
  _CLOSE = ctx("CLOSE")
  _VOLUME = ctx("VOLUME")
  return np.where(
    ctx.SUM(_CLOSE, 8) / 8 + ctx.STDDEV(_CLOSE, 8) < ctx.SUM(_CLOSE, 2) / 2,
    -1 * 1,
    np.where(
      ctx.SUM(_CLOSE, 2) / 2 < ctx.SUM(_CLOSE, 8) / 8 - ctx.STDDEV(_CLOSE, 8),
      1,
      np.where(np.bitwise_or(1 < _VOLUME / _ADV20, _VOLUME / _ADV20 == 1), 1, -1 * 1),
    ),
  )


# (-1 * (delta(correlation(high, volume, 5), 5) * rank(stddev(close, 20))))
def alpha_022(ctx):
  return (
    -1
    * ctx.DELTA(ctx.CORRELATION(ctx("HIGH"), ctx("VOLUME"), 5), 5)
    * ctx.RANK(ctx.STDDEV(ctx("CLOSE"), 20))
  )


# (((sum(high, 20) / 20) < high) ? (-1 * delta(high, 2)) : 0)
def alpha_023(ctx):
  _HIGH = ctx("HIGH")
  return np.where(ctx.SUM(_HIGH, 20) / 20 < _HIGH, -1 * ctx.DELTA(_HIGH, 2), 0)


# ((((delta((sum(close, 100) / 100), 100) / delay(close, 100)) < 0.05) ||((delta((sum(close, 100) / 100), 100) / delay(close, 100)) == 0.05)) ? (-1 * (close - ts_min(close,100))) : (-1 * delta(close, 3)))
def alpha_024(ctx):
  _CLOSE = ctx("CLOSE")
  return np.where(
    np.bitwise_or(
      ctx.DELTA(ctx.SUM(_CLOSE, 100) / 100, 100) / ctx.DELAY(_CLOSE, 100) < 0.05,
      ctx.DELTA(ctx.SUM(_CLOSE, 100) / 100, 100) / ctx.DELAY(_CLOSE, 100) == 0.05,
    ),
    -1 * _CLOSE - ctx.TS_MIN(_CLOSE, 100),
    -1 * ctx.DELTA(_CLOSE, 3),
  )


# rank(((((-1 * returns) * adv20) * vwap) * (high - close)))
def alpha_025(ctx):
  return ctx.RANK(
    -1 * ctx("RETURNS") * ctx("ADV20") * ctx("VWAP") * ctx("HIGH") - ctx("CLOSE")
  )


# (-1 * ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3))
def alpha_026(ctx):
  return -1 * ctx.TS_MAX(
    ctx.CORRELATION(ctx.TS_RANK(ctx("VOLUME"), 5), ctx.TS_RANK(ctx("HIGH"), 5), 5), 3
  )


# ((0.5 < rank((sum(correlation(rank(volume), rank(vwap), 6), 2) / 2.0))) ? (-1 * 1) : 1)
def alpha_027(ctx):
  return np.where(
    0.5
    < ctx.RANK(
      ctx.SUM(ctx.CORRELATION(ctx.RANK(ctx("VOLUME")), ctx.RANK(ctx("VWAP")), 6), 2)
      / 2.0
    ),
    -1 * 1,
    1,
  )


# scale(((correlation(adv20, low, 5) + ((high + low) / 2)) - close))
def alpha_028(ctx):
  _LOW = ctx("LOW")
  return ctx.SCALE(
    ctx.CORRELATION(ctx("ADV20"), _LOW, 5) + ctx("HIGH") + _LOW / 2 - ctx("CLOSE")
  )


# (min(product(rank(rank(scale(log(sum(ts_min(rank(rank((-1 * rank(delta((close - 1),5))))), 2), 1))))), 1), 5) + ts_rank(delay((-1 * returns), 6), 5))
def alpha_029(ctx):
  return ctx.MIN(
    ctx.PRODUCT(
      ctx.RANK(
        ctx.RANK(
          ctx.SCALE(
            ctx.LOG(
              ctx.SUM(
                ctx.TS_MIN(
                  ctx.RANK(ctx.RANK(-1 * ctx.RANK(ctx.DELTA(ctx("CLOSE") - 1, 5)))), 2
                ),
                1,
              )
            )
          )
        )
      ),
      1,
    ),
    5,
  ) + ctx.TS_RANK(ctx.DELAY(-1 * ctx("RETURNS"), 6), 5)


# (((1.0 - rank(((sign((close - delay(close, 1))) + sign((delay(close, 1) - delay(close, 2)))) +sign((delay(close, 2) - delay(close, 3)))))) * sum(volume, 5)) / sum(volume, 20))
def alpha_030(ctx):
  _CLOSE = ctx("CLOSE")
  _VOLUME = ctx("VOLUME")
  return 1.0 - ctx.RANK(
    ctx.SIGN(_CLOSE - ctx.DELAY(_CLOSE, 1))
    + ctx.SIGN(ctx.DELAY(_CLOSE, 1) - ctx.DELAY(_CLOSE, 2))
    + ctx.SIGN(ctx.DELAY(_CLOSE, 2) - ctx.DELAY(_CLOSE, 3))
  ) * ctx.SUM(_VOLUME, 5) / ctx.SUM(_VOLUME, 20)


# ((rank(rank(rank(decay_linear((-1 * rank(rank(delta(close, 10)))), 10)))) + rank((-1 *delta(close, 3)))) + sign(scale(correlation(adv20, low, 12))))
def alpha_031(ctx):
  _CLOSE = ctx("CLOSE")
  return (
    ctx.RANK(
      ctx.RANK(
        ctx.RANK(ctx.DECAY_LINEAR(-1 * ctx.RANK(ctx.RANK(ctx.DELTA(_CLOSE, 10))), 10))
      )
    )
    + ctx.RANK(-1 * ctx.DELTA(_CLOSE, 3))
    + ctx.SIGN(ctx.SCALE(ctx.CORRELATION(ctx("ADV20"), ctx("LOW"), 12)))
  )


# (scale(((sum(close, 7) / 7) - close)) + (20 * scale(correlation(vwap, delay(close, 5),230))))
def alpha_032(ctx):
  _CLOSE = ctx("CLOSE")
  return ctx.SCALE(ctx.SUM(_CLOSE, 7) / 7 - _CLOSE) + 20 * ctx.SCALE(
    ctx.CORRELATION(ctx("VWAP"), ctx.DELAY(_CLOSE, 5), 230)
  )


# rank((-1 * ((1 - (open / close))^1)))
def alpha_033(ctx):
  return ctx.RANK(-1 * np.power(1 - ctx("OPEN") / ctx("CLOSE"), 1))


# rank(((1 - rank((stddev(returns, 2) / stddev(returns, 5)))) + (1 - rank(delta(close, 1)))))
def alpha_034(ctx):
  _RETURNS = ctx("RETURNS")
  return ctx.RANK(
    1
    - ctx.RANK(ctx.STDDEV(_RETURNS, 2) / ctx.STDDEV(_RETURNS, 5))
    + 1
    - ctx.RANK(ctx.DELTA(ctx("CLOSE"), 1))
  )


# ((Ts_Rank(volume, 32) * (1 - Ts_Rank(((close + high) - low), 16))) * (1 -Ts_Rank(returns, 32)))
def alpha_035(ctx):
  return (
    ctx.TS_RANK(ctx("VOLUME"), 32) * 1
    - ctx.TS_RANK(ctx("CLOSE") + ctx("HIGH") - ctx("LOW"), 16) * 1
    - ctx.TS_RANK(ctx("RETURNS"), 32)
  )


# (((((2.21 * rank(correlation((close - open), delay(volume, 1), 15))) + (0.7 * rank((open- close)))) + (0.73 * rank(Ts_Rank(delay((-1 * returns), 6), 5)))) + rank(abs(correlation(vwap,adv20, 6)))) + (0.6 * rank((((sum(close, 200) / 200) - open) * (close - open)))))
def alpha_036(ctx):
  _CLOSE = ctx("CLOSE")
  _OPEN = ctx("OPEN")
  return (
    2.21 * ctx.RANK(ctx.CORRELATION(_CLOSE - _OPEN, ctx.DELAY(ctx("VOLUME"), 1), 15))
    + 0.7 * ctx.RANK(_OPEN - _CLOSE)
    + 0.73 * ctx.RANK(ctx.TS_RANK(ctx.DELAY(-1 * ctx("RETURNS"), 6), 5))
    + ctx.RANK(ctx.ABS(ctx.CORRELATION(ctx("VWAP"), ctx("ADV20"), 6)))
    + 0.6 * ctx.RANK(ctx.SUM(_CLOSE, 200) / 200 - _OPEN * _CLOSE - _OPEN)
  )


# (rank(correlation(delay((open - close), 1), close, 200)) + rank((open - close)))
def alpha_037(ctx):
  _CLOSE = ctx("CLOSE")
  _OPEN = ctx("OPEN")
  return ctx.RANK(
    ctx.CORRELATION(ctx.DELAY(_OPEN - _CLOSE, 1), _CLOSE, 200)
  ) + ctx.RANK(_OPEN - _CLOSE)


# ((-1 * rank(Ts_Rank(close, 10))) * rank((close / open)))
def alpha_038(ctx):
  _CLOSE = ctx("CLOSE")
  return -1 * ctx.RANK(ctx.TS_RANK(_CLOSE, 10)) * ctx.RANK(_CLOSE / ctx("OPEN"))


# ((-1 * rank((delta(close, 7) * (1 - rank(decay_linear((volume / adv20), 9)))))) * (1 +rank(sum(returns, 250))))
def alpha_039(ctx):
  return -1 * ctx.RANK(
    ctx.DELTA(ctx("CLOSE"), 7) * 1
    - ctx.RANK(ctx.DECAY_LINEAR(ctx("VOLUME") / ctx("ADV20"), 9))
  ) * 1 + ctx.RANK(ctx.SUM(ctx("RETURNS"), 250))


# ((-1 * rank(stddev(high, 10))) * correlation(high, volume, 10))
def alpha_040(ctx):
  _HIGH = ctx("HIGH")
  return (
    -1 * ctx.RANK(ctx.STDDEV(_HIGH, 10)) * ctx.CORRELATION(_HIGH, ctx("VOLUME"), 10)
  )


# (((high * low)^0.5) - vwap)
def alpha_041(ctx):
  return np.power(ctx("HIGH") * ctx("LOW"), 0.5) - ctx("VWAP")


# (rank((vwap - close)) / rank((vwap + close)))
def alpha_042(ctx):
  _CLOSE = ctx("CLOSE")
  _VWAP = ctx("VWAP")
  return ctx.RANK(_VWAP - _CLOSE) / ctx.RANK(_VWAP + _CLOSE)


# (ts_rank((volume / adv20), 20) * ts_rank((-1 * delta(close, 7)), 8))
def alpha_043(ctx):
  return ctx.TS_RANK(ctx("VOLUME") / ctx("ADV20"), 20) * ctx.TS_RANK(
    -1 * ctx.DELTA(ctx("CLOSE"), 7), 8
  )


# (-1 * correlation(high, rank(volume), 5))
def alpha_044(ctx):
  return -1 * ctx.CORRELATION(ctx("HIGH"), ctx.RANK(ctx("VOLUME")), 5)


# (-1 * ((rank((sum(delay(close, 5), 20) / 20)) * correlation(close, volume, 2)) *rank(correlation(sum(close, 5), sum(close, 20), 2))))
def alpha_045(ctx):
  _CLOSE = ctx("CLOSE")
  return (
    -1
    * ctx.RANK(ctx.SUM(ctx.DELAY(_CLOSE, 5), 20) / 20)
    * ctx.CORRELATION(_CLOSE, ctx("VOLUME"), 2)
    * ctx.RANK(ctx.CORRELATION(ctx.SUM(_CLOSE, 5), ctx.SUM(_CLOSE, 20), 2))
  )


# ((0.25 < (((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10))) ?(-1 * 1) : (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < 0) ? 1 :((-1 * 1) * (close - delay(close, 1)))))
def alpha_046(ctx):
  _CLOSE = ctx("CLOSE")
  return np.where(
    0.25
    < ctx.DELAY(_CLOSE, 20)
    - ctx.DELAY(_CLOSE, 10) / 10
    - ctx.DELAY(_CLOSE, 10)
    - _CLOSE / 10,
    -1 * 1,
    np.where(
      ctx.DELAY(_CLOSE, 20)
      - ctx.DELAY(_CLOSE, 10) / 10
      - ctx.DELAY(_CLOSE, 10)
      - _CLOSE / 10
      < 0,
      1,
      -1 * 1 * _CLOSE - ctx.DELAY(_CLOSE, 1),
    ),
  )


# ((((rank((1 / close)) * volume) / adv20) * ((high * rank((high - close))) / (sum(high, 5) /5))) - rank((vwap - delay(vwap, 5))))
def alpha_047(ctx):
  _CLOSE = ctx("CLOSE")
  _HIGH = ctx("HIGH")
  _VWAP = ctx("VWAP")
  return ctx.RANK(1 / _CLOSE) * ctx("VOLUME") / ctx("ADV20") * _HIGH * ctx.RANK(
    _HIGH - _CLOSE
  ) / ctx.SUM(_HIGH, 5) / 5 - ctx.RANK(_VWAP - ctx.DELAY(_VWAP, 5))


# (indneutralize(((correlation(delta(close, 1), delta(delay(close, 1), 1), 250) *delta(close, 1)) / close), IndClass.subindustry) / sum(((delta(close, 1) / delay(close, 1))^2), 250))
def alpha_048(ctx):
  _CLOSE = ctx("CLOSE")
  return ctx.INDNEUTRALIZE(
    ctx.CORRELATION(ctx.DELTA(_CLOSE, 1), ctx.DELTA(ctx.DELAY(_CLOSE, 1), 1), 250)
    * ctx.DELTA(_CLOSE, 1)
    / _CLOSE,
    ctx("INDCLASS.SUBINDUSTRY"),
  ) / ctx.SUM(np.power(ctx.DELTA(_CLOSE, 1) / ctx.DELAY(_CLOSE, 1), 2), 250)


# (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 *0.1)) ? 1 : ((-1 * 1) * (close - delay(close, 1))))
def alpha_049(ctx):
  _CLOSE = ctx("CLOSE")
  return np.where(
    ctx.DELAY(_CLOSE, 20)
    - ctx.DELAY(_CLOSE, 10) / 10
    - ctx.DELAY(_CLOSE, 10)
    - _CLOSE / 10
    < -1 * 0.1,
    1,
    -1 * 1 * _CLOSE - ctx.DELAY(_CLOSE, 1),
  )


# (-1 * ts_max(rank(correlation(rank(volume), rank(vwap), 5)), 5))
def alpha_050(ctx):
  return -1 * ctx.TS_MAX(
    ctx.RANK(ctx.CORRELATION(ctx.RANK(ctx("VOLUME")), ctx.RANK(ctx("VWAP")), 5)), 5
  )


# (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 *0.05)) ? 1 : ((-1 * 1) * (close - delay(close, 1))))
def alpha_051(ctx):
  _CLOSE = ctx("CLOSE")
  return np.where(
    ctx.DELAY(_CLOSE, 20)
    - ctx.DELAY(_CLOSE, 10) / 10
    - ctx.DELAY(_CLOSE, 10)
    - _CLOSE / 10
    < -1 * 0.05,
    1,
    -1 * 1 * _CLOSE - ctx.DELAY(_CLOSE, 1),
  )


# ((((-1 * ts_min(low, 5)) + delay(ts_min(low, 5), 5)) * rank(((sum(returns, 240) -sum(returns, 20)) / 220))) * ts_rank(volume, 5))
def alpha_052(ctx):
  _LOW = ctx("LOW")
  _RETURNS = ctx("RETURNS")
  return -1 * ctx.TS_MIN(_LOW, 5) + ctx.DELAY(ctx.TS_MIN(_LOW, 5), 5) * ctx.RANK(
    ctx.SUM(_RETURNS, 240) - ctx.SUM(_RETURNS, 20) / 220
  ) * ctx.TS_RANK(ctx("VOLUME"), 5)


# (-1 * delta((((close - low) - (high - close)) / (close - low)), 9))
def alpha_053(ctx):
  _CLOSE = ctx("CLOSE")
  _LOW = ctx("LOW")
  return -1 * ctx.DELTA(_CLOSE - _LOW - ctx("HIGH") - _CLOSE / _CLOSE - _LOW, 9)


# ((-1 * ((low - close) * (open^5))) / ((low - high) * (close^5)))
def alpha_054(ctx):
  _CLOSE = ctx("CLOSE")
  _LOW = ctx("LOW")
  return (
    -1 * _LOW
    - _CLOSE * np.power(ctx("OPEN"), 5) / _LOW
    - ctx("HIGH") * np.power(_CLOSE, 5)
  )


# (-1 * correlation(rank(((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low,12)))), rank(volume), 6))
def alpha_055(ctx):
  _LOW = ctx("LOW")
  return -1 * ctx.CORRELATION(
    ctx.RANK(
      ctx("CLOSE")
      - ctx.TS_MIN(_LOW, 12) / ctx.TS_MAX(ctx("HIGH"), 12)
      - ctx.TS_MIN(_LOW, 12)
    ),
    ctx.RANK(ctx("VOLUME")),
    6,
  )


# (0 - (1 * (rank((sum(returns, 10) / sum(sum(returns, 2), 3))) * rank((returns * cap)))))
def alpha_056(ctx):
  _RETURNS = ctx("RETURNS")
  return 0 - 1 * ctx.RANK(
    ctx.SUM(_RETURNS, 10) / ctx.SUM(ctx.SUM(_RETURNS, 2), 3)
  ) * ctx.RANK(_RETURNS * ctx("CAP"))


# (0 - (1 * ((close - vwap) / decay_linear(rank(ts_argmax(close, 30)), 2))))
def alpha_057(ctx):
  _CLOSE = ctx("CLOSE")
  return (
    0
    - 1 * _CLOSE
    - ctx("VWAP") / ctx.DECAY_LINEAR(ctx.RANK(ctx.TS_ARGMAX(_CLOSE, 30)), 2)
  )


# (-1 * Ts_Rank(decay_linear(correlation(IndNeutralize(vwap, IndClass.sector), volume,3.92795), 7.89291), 5.50322))
def alpha_058(ctx):
  return -1 * ctx.TS_RANK(
    ctx.DECAY_LINEAR(
      ctx.CORRELATION(
        ctx.INDNEUTRALIZE(ctx("VWAP"), ctx("INDCLASS.SECTOR")), ctx("VOLUME"), 3.92795
      ),
      7.89291,
    ),
    5.50322,
  )


# (-1 * Ts_Rank(decay_linear(correlation(IndNeutralize(((vwap * 0.728317) + (vwap *(1 - 0.728317))), IndClass.industry), volume, 4.25197), 16.2289), 8.19648))
def alpha_059(ctx):
  _VWAP = ctx("VWAP")
  return -1 * ctx.TS_RANK(
    ctx.DECAY_LINEAR(
      ctx.CORRELATION(
        ctx.INDNEUTRALIZE(
          _VWAP * 0.728317 + _VWAP * 1 - 0.728317, ctx("INDCLASS.INDUSTRY")
        ),
        ctx("VOLUME"),
        4.25197,
      ),
      16.2289,
    ),
    8.19648,
  )


# (0 - (1 * ((2 * scale(rank(((((close - low) - (high - close)) / (high - low)) * volume)))) -scale(rank(ts_argmax(close, 10))))))
def alpha_060(ctx):
  _CLOSE = ctx("CLOSE")
  _HIGH = ctx("HIGH")
  _LOW = ctx("LOW")
  return (
    0
    - 1
    * 2
    * ctx.SCALE(ctx.RANK(_CLOSE - _LOW - _HIGH - _CLOSE / _HIGH - _LOW * ctx("VOLUME")))
    - ctx.SCALE(ctx.RANK(ctx.TS_ARGMAX(_CLOSE, 10)))
  )


# (rank((vwap - ts_min(vwap, 16.1219))) < rank(correlation(vwap, adv180, 17.9282)))
def alpha_061(ctx):
  _VWAP = ctx("VWAP")
  return ctx.RANK(_VWAP - ctx.TS_MIN(_VWAP, 16.1219)) < ctx.RANK(
    ctx.CORRELATION(_VWAP, ctx("ADV180"), 17.9282)
  )


# ((rank(correlation(vwap, sum(adv20, 22.4101), 9.91009)) < rank(((rank(open) +rank(open)) < (rank(((high + low) / 2)) + rank(high))))) * -1)
def alpha_062(ctx):
  _HIGH = ctx("HIGH")
  _OPEN = ctx("OPEN")
  return (
    ctx.RANK(ctx.CORRELATION(ctx("VWAP"), ctx.SUM(ctx("ADV20"), 22.4101), 9.91009))
    < ctx.RANK(
      ctx.RANK(_OPEN)
      + np.where(ctx.RANK(_OPEN) < ctx.RANK(_HIGH + ctx("LOW") / 2), 1.0, 0.0)
      + ctx.RANK(_HIGH)
    )
    * -1
  )


# ((rank(decay_linear(delta(IndNeutralize(close, IndClass.industry), 2.25164), 8.22237))- rank(decay_linear(correlation(((vwap * 0.318108) + (open * (1 - 0.318108))), sum(adv180,37.2467), 13.557), 12.2883))) * -1)
def alpha_063(ctx):
  return (
    ctx.RANK(
      ctx.DECAY_LINEAR(
        ctx.DELTA(ctx.INDNEUTRALIZE(ctx("CLOSE"), ctx("INDCLASS.INDUSTRY")), 2.25164),
        8.22237,
      )
    )
    - ctx.RANK(
      ctx.DECAY_LINEAR(
        ctx.CORRELATION(
          ctx("VWAP") * 0.318108 + ctx("OPEN") * 1 - 0.318108,
          ctx.SUM(ctx("ADV180"), 37.2467),
          13.557,
        ),
        12.2883,
      )
    )
    * -1
  )


# ((rank(correlation(sum(((open * 0.178404) + (low * (1 - 0.178404))), 12.7054),sum(adv120, 12.7054), 16.6208)) < rank(delta(((((high + low) / 2) * 0.178404) + (vwap * (1 -0.178404))), 3.69741))) * -1)
def alpha_064(ctx):
  _LOW = ctx("LOW")
  return (
    ctx.RANK(
      ctx.CORRELATION(
        ctx.SUM(ctx("OPEN") * 0.178404 + _LOW * 1 - 0.178404, 12.7054),
        ctx.SUM(ctx("ADV120"), 12.7054),
        16.6208,
      )
    )
    < ctx.RANK(
      ctx.DELTA(ctx("HIGH") + _LOW / 2 * 0.178404 + ctx("VWAP") * 1 - 0.178404, 3.69741)
    )
    * -1
  )


# ((rank(correlation(((open * 0.00817205) + (vwap * (1 - 0.00817205))), sum(adv60,8.6911), 6.40374)) < rank((open - ts_min(open, 13.635)))) * -1)
def alpha_065(ctx):
  _OPEN = ctx("OPEN")
  return (
    ctx.RANK(
      ctx.CORRELATION(
        _OPEN * 0.00817205 + ctx("VWAP") * 1 - 0.00817205,
        ctx.SUM(ctx("ADV60"), 8.6911),
        6.40374,
      )
    )
    < ctx.RANK(_OPEN - ctx.TS_MIN(_OPEN, 13.635)) * -1
  )


# ((rank(decay_linear(delta(vwap, 3.51013), 7.23052)) + Ts_Rank(decay_linear(((((low* 0.96633) + (low * (1 - 0.96633))) - vwap) / (open - ((high + low) / 2))), 11.4157), 6.72611)) * -1)
def alpha_066(ctx):
  _LOW = ctx("LOW")
  _VWAP = ctx("VWAP")
  return (
    ctx.RANK(ctx.DECAY_LINEAR(ctx.DELTA(_VWAP, 3.51013), 7.23052))
    + ctx.TS_RANK(
      ctx.DECAY_LINEAR(
        _LOW * 0.96633
        + _LOW * 1
        - 0.96633
        - _VWAP / ctx("OPEN")
        - ctx("HIGH")
        + _LOW / 2,
        11.4157,
      ),
      6.72611,
    )
    * -1
  )


# ((rank((high - ts_min(high, 2.14593)))^rank(correlation(IndNeutralize(vwap,IndClass.sector), IndNeutralize(adv20, IndClass.subindustry), 6.02936))) * -1)
def alpha_067(ctx):
  _HIGH = ctx("HIGH")
  return (
    np.power(
      ctx.RANK(_HIGH - ctx.TS_MIN(_HIGH, 2.14593)),
      ctx.RANK(
        ctx.CORRELATION(
          ctx.INDNEUTRALIZE(ctx("VWAP"), ctx("INDCLASS.SECTOR")),
          ctx.INDNEUTRALIZE(ctx("ADV20"), ctx("INDCLASS.SUBINDUSTRY")),
          6.02936,
        )
      ),
    )
    * -1
  )


# ((Ts_Rank(correlation(rank(high), rank(adv15), 8.91644), 13.9333) <rank(delta(((close * 0.518371) + (low * (1 - 0.518371))), 1.06157))) * -1)
def alpha_068(ctx):
  return (
    ctx.TS_RANK(
      ctx.CORRELATION(ctx.RANK(ctx("HIGH")), ctx.RANK(ctx("ADV15")), 8.91644), 13.9333
    )
    < ctx.RANK(ctx.DELTA(ctx("CLOSE") * 0.518371 + ctx("LOW") * 1 - 0.518371, 1.06157))
    * -1
  )


# ((rank(ts_max(delta(IndNeutralize(vwap, IndClass.industry), 2.72412),4.79344))^Ts_Rank(correlation(((close * 0.490655) + (vwap * (1 - 0.490655))), adv20, 4.92416),9.0615)) * -1)
def alpha_069(ctx):
  _VWAP = ctx("VWAP")
  return (
    np.power(
      ctx.RANK(
        ctx.TS_MAX(
          ctx.DELTA(ctx.INDNEUTRALIZE(_VWAP, ctx("INDCLASS.INDUSTRY")), 2.72412),
          4.79344,
        )
      ),
      ctx.TS_RANK(
        ctx.CORRELATION(
          ctx("CLOSE") * 0.490655 + _VWAP * 1 - 0.490655, ctx("ADV20"), 4.92416
        ),
        9.0615,
      ),
    )
    * -1
  )


# ((rank(delta(vwap, 1.29456))^Ts_Rank(correlation(IndNeutralize(close,IndClass.industry), adv50, 17.8256), 17.9171)) * -1)
def alpha_070(ctx):
  return (
    np.power(
      ctx.RANK(ctx.DELTA(ctx("VWAP"), 1.29456)),
      ctx.TS_RANK(
        ctx.CORRELATION(
          ctx.INDNEUTRALIZE(ctx("CLOSE"), ctx("INDCLASS.INDUSTRY")),
          ctx("ADV50"),
          17.8256,
        ),
        17.9171,
      ),
    )
    * -1
  )


# max(Ts_Rank(decay_linear(correlation(Ts_Rank(close, 3.43976), Ts_Rank(adv180,12.0647), 18.0175), 4.20501), 15.6948), Ts_Rank(decay_linear((rank(((low + open) - (vwap +vwap)))^2), 16.4662), 4.4388))
def alpha_071(ctx):
  _VWAP = ctx("VWAP")
  return ctx.MAX(
    ctx.TS_RANK(
      ctx.DECAY_LINEAR(
        ctx.CORRELATION(
          ctx.TS_RANK(ctx("CLOSE"), 3.43976),
          ctx.TS_RANK(ctx("ADV180"), 12.0647),
          18.0175,
        ),
        4.20501,
      ),
      15.6948,
    ),
    ctx.TS_RANK(
      ctx.DECAY_LINEAR(
        np.power(ctx.RANK(ctx("LOW") + ctx("OPEN") - _VWAP + _VWAP), 2), 16.4662
      ),
      4.4388,
    ),
  )


# (rank(decay_linear(correlation(((high + low) / 2), adv40, 8.93345), 10.1519)) /rank(decay_linear(correlation(Ts_Rank(vwap, 3.72469), Ts_Rank(volume, 18.5188), 6.86671),2.95011)))
def alpha_072(ctx):
  return ctx.RANK(
    ctx.DECAY_LINEAR(
      ctx.CORRELATION(ctx("HIGH") + ctx("LOW") / 2, ctx("ADV40"), 8.93345), 10.1519
    )
  ) / ctx.RANK(
    ctx.DECAY_LINEAR(
      ctx.CORRELATION(
        ctx.TS_RANK(ctx("VWAP"), 3.72469), ctx.TS_RANK(ctx("VOLUME"), 18.5188), 6.86671
      ),
      2.95011,
    )
  )


# (max(rank(decay_linear(delta(vwap, 4.72775), 2.91864)),Ts_Rank(decay_linear(((delta(((open * 0.147155) + (low * (1 - 0.147155))), 2.03608) / ((open *0.147155) + (low * (1 - 0.147155)))) * -1), 3.33829), 16.7411)) * -1)
def alpha_073(ctx):
  _LOW = ctx("LOW")
  _OPEN = ctx("OPEN")
  return (
    ctx.MAX(
      ctx.RANK(ctx.DECAY_LINEAR(ctx.DELTA(ctx("VWAP"), 4.72775), 2.91864)),
      ctx.TS_RANK(
        ctx.DECAY_LINEAR(
          ctx.DELTA(_OPEN * 0.147155 + _LOW * 1 - 0.147155, 2.03608) / _OPEN * 0.147155
          + _LOW * 1
          - 0.147155 * -1,
          3.33829,
        ),
        16.7411,
      ),
    )
    * -1
  )


# ((rank(correlation(close, sum(adv30, 37.4843), 15.1365)) <rank(correlation(rank(((high * 0.0261661) + (vwap * (1 - 0.0261661)))), rank(volume), 11.4791)))* -1)
def alpha_074(ctx):
  return (
    ctx.RANK(ctx.CORRELATION(ctx("CLOSE"), ctx.SUM(ctx("ADV30"), 37.4843), 15.1365))
    < ctx.RANK(
      ctx.CORRELATION(
        ctx.RANK(ctx("HIGH") * 0.0261661 + ctx("VWAP") * 1 - 0.0261661),
        ctx.RANK(ctx("VOLUME")),
        11.4791,
      )
    )
    * -1
  )


# (rank(correlation(vwap, volume, 4.24304)) < rank(correlation(rank(low), rank(adv50),12.4413)))
def alpha_075(ctx):
  return ctx.RANK(ctx.CORRELATION(ctx("VWAP"), ctx("VOLUME"), 4.24304)) < ctx.RANK(
    ctx.CORRELATION(ctx.RANK(ctx("LOW")), ctx.RANK(ctx("ADV50")), 12.4413)
  )


# (max(rank(decay_linear(delta(vwap, 1.24383), 11.8259)),Ts_Rank(decay_linear(Ts_Rank(correlation(IndNeutralize(low, IndClass.sector), adv81,8.14941), 19.569), 17.1543), 19.383)) * -1)
def alpha_076(ctx):
  return (
    ctx.MAX(
      ctx.RANK(ctx.DECAY_LINEAR(ctx.DELTA(ctx("VWAP"), 1.24383), 11.8259)),
      ctx.TS_RANK(
        ctx.DECAY_LINEAR(
          ctx.TS_RANK(
            ctx.CORRELATION(
              ctx.INDNEUTRALIZE(ctx("LOW"), ctx("INDCLASS.SECTOR")),
              ctx("ADV81"),
              8.14941,
            ),
            19.569,
          ),
          17.1543,
        ),
        19.383,
      ),
    )
    * -1
  )


# min(rank(decay_linear(((((high + low) / 2) + high) - (vwap + high)), 20.0451)),rank(decay_linear(correlation(((high + low) / 2), adv40, 3.1614), 5.64125)))
def alpha_077(ctx):
  _HIGH = ctx("HIGH")
  _LOW = ctx("LOW")
  return ctx.MIN(
    ctx.RANK(ctx.DECAY_LINEAR(_HIGH + _LOW / 2 + _HIGH - ctx("VWAP") + _HIGH, 20.0451)),
    ctx.RANK(
      ctx.DECAY_LINEAR(ctx.CORRELATION(_HIGH + _LOW / 2, ctx("ADV40"), 3.1614), 5.64125)
    ),
  )


# (rank(correlation(sum(((low * 0.352233) + (vwap * (1 - 0.352233))), 19.7428),sum(adv40, 19.7428), 6.83313))^rank(correlation(rank(vwap), rank(volume), 5.77492)))
def alpha_078(ctx):
  _VWAP = ctx("VWAP")
  return np.power(
    ctx.RANK(
      ctx.CORRELATION(
        ctx.SUM(ctx("LOW") * 0.352233 + _VWAP * 1 - 0.352233, 19.7428),
        ctx.SUM(ctx("ADV40"), 19.7428),
        6.83313,
      )
    ),
    ctx.RANK(ctx.CORRELATION(ctx.RANK(_VWAP), ctx.RANK(ctx("VOLUME")), 5.77492)),
  )


# (rank(delta(IndNeutralize(((close * 0.60733) + (open * (1 - 0.60733))),IndClass.sector), 1.23438)) < rank(correlation(Ts_Rank(vwap, 3.60973), Ts_Rank(adv150,9.18637), 14.6644)))
def alpha_079(ctx):
  return ctx.RANK(
    ctx.DELTA(
      ctx.INDNEUTRALIZE(
        ctx("CLOSE") * 0.60733 + ctx("OPEN") * 1 - 0.60733, ctx("INDCLASS.SECTOR")
      ),
      1.23438,
    )
  ) < ctx.RANK(
    ctx.CORRELATION(
      ctx.TS_RANK(ctx("VWAP"), 3.60973), ctx.TS_RANK(ctx("ADV150"), 9.18637), 14.6644
    )
  )


# ((rank(Sign(delta(IndNeutralize(((open * 0.868128) + (high * (1 - 0.868128))),IndClass.industry), 4.04545)))^Ts_Rank(correlation(high, adv10, 5.11456), 5.53756)) * -1)
def alpha_080(ctx):
  _HIGH = ctx("HIGH")
  return (
    np.power(
      ctx.RANK(
        ctx.SIGN(
          ctx.DELTA(
            ctx.INDNEUTRALIZE(
              ctx("OPEN") * 0.868128 + _HIGH * 1 - 0.868128, ctx("INDCLASS.INDUSTRY")
            ),
            4.04545,
          )
        )
      ),
      ctx.TS_RANK(ctx.CORRELATION(_HIGH, ctx("ADV10"), 5.11456), 5.53756),
    )
    * -1
  )


# ((rank(Log(product(rank((rank(correlation(vwap, sum(adv10, 49.6054),8.47743))^4)), 14.9655))) < rank(correlation(rank(vwap), rank(volume), 5.07914))) * -1)
def alpha_081(ctx):
  _VWAP = ctx("VWAP")
  return (
    ctx.RANK(
      ctx.LOG(
        ctx.PRODUCT(
          ctx.RANK(
            np.power(
              ctx.RANK(ctx.CORRELATION(_VWAP, ctx.SUM(ctx("ADV10"), 49.6054), 8.47743)),
              4,
            )
          ),
          14.9655,
        )
      )
    )
    < ctx.RANK(ctx.CORRELATION(ctx.RANK(_VWAP), ctx.RANK(ctx("VOLUME")), 5.07914)) * -1
  )


# (min(rank(decay_linear(delta(open, 1.46063), 14.8717)),Ts_Rank(decay_linear(correlation(IndNeutralize(volume, IndClass.sector), ((open * 0.634196) +(open * (1 - 0.634196))), 17.4842), 6.92131), 13.4283)) * -1)
def alpha_082(ctx):
  _OPEN = ctx("OPEN")
  return (
    ctx.MIN(
      ctx.RANK(ctx.DECAY_LINEAR(ctx.DELTA(_OPEN, 1.46063), 14.8717)),
      ctx.TS_RANK(
        ctx.DECAY_LINEAR(
          ctx.CORRELATION(
            ctx.INDNEUTRALIZE(ctx("VOLUME"), ctx("INDCLASS.SECTOR")),
            _OPEN * 0.634196 + _OPEN * 1 - 0.634196,
            17.4842,
          ),
          6.92131,
        ),
        13.4283,
      ),
    )
    * -1
  )


# ((rank(delay(((high - low) / (sum(close, 5) / 5)), 2)) * rank(rank(volume))) / (((high -low) / (sum(close, 5) / 5)) / (vwap - close)))
def alpha_083(ctx):
  _CLOSE = ctx("CLOSE")
  _HIGH = ctx("HIGH")
  _LOW = ctx("LOW")
  return (
    ctx.RANK(ctx.DELAY(_HIGH - _LOW / ctx.SUM(_CLOSE, 5) / 5, 2))
    * ctx.RANK(ctx.RANK(ctx("VOLUME")))
    / _HIGH
    - _LOW / ctx.SUM(_CLOSE, 5) / 5 / ctx("VWAP")
    - _CLOSE
  )


# SignedPower(Ts_Rank((vwap - ts_max(vwap, 15.3217)), 20.7127), delta(close,4.96796))
def alpha_084(ctx):
  _VWAP = ctx("VWAP")
  return ctx.SIGNEDPOWER(
    ctx.TS_RANK(_VWAP - ctx.TS_MAX(_VWAP, 15.3217), 20.7127),
    ctx.DELTA(ctx("CLOSE"), 4.96796),
  )


# (rank(correlation(((high * 0.876703) + (close * (1 - 0.876703))), adv30,9.61331))^rank(correlation(Ts_Rank(((high + low) / 2), 3.70596), Ts_Rank(volume, 10.1595),7.11408)))
def alpha_085(ctx):
  _HIGH = ctx("HIGH")
  return np.power(
    ctx.RANK(
      ctx.CORRELATION(
        _HIGH * 0.876703 + ctx("CLOSE") * 1 - 0.876703, ctx("ADV30"), 9.61331
      )
    ),
    ctx.RANK(
      ctx.CORRELATION(
        ctx.TS_RANK(_HIGH + ctx("LOW") / 2, 3.70596),
        ctx.TS_RANK(ctx("VOLUME"), 10.1595),
        7.11408,
      )
    ),
  )


# ((Ts_Rank(correlation(close, sum(adv20, 14.7444), 6.00049), 20.4195) < rank(((open+ close) - (vwap + open)))) * -1)
def alpha_086(ctx):
  _CLOSE = ctx("CLOSE")
  _OPEN = ctx("OPEN")
  return (
    ctx.TS_RANK(
      ctx.CORRELATION(_CLOSE, ctx.SUM(ctx("ADV20"), 14.7444), 6.00049), 20.4195
    )
    < ctx.RANK(_OPEN + _CLOSE - ctx("VWAP") + _OPEN) * -1
  )


# (max(rank(decay_linear(delta(((close * 0.369701) + (vwap * (1 - 0.369701))),1.91233), 2.65461)), Ts_Rank(decay_linear(abs(correlation(IndNeutralize(adv81,IndClass.industry), close, 13.4132)), 4.89768), 14.4535)) * -1)
def alpha_087(ctx):
  _CLOSE = ctx("CLOSE")
  return (
    ctx.MAX(
      ctx.RANK(
        ctx.DECAY_LINEAR(
          ctx.DELTA(_CLOSE * 0.369701 + ctx("VWAP") * 1 - 0.369701, 1.91233), 2.65461
        )
      ),
      ctx.TS_RANK(
        ctx.DECAY_LINEAR(
          ctx.ABS(
            ctx.CORRELATION(
              ctx.INDNEUTRALIZE(ctx("ADV81"), ctx("INDCLASS.INDUSTRY")), _CLOSE, 13.4132
            )
          ),
          4.89768,
        ),
        14.4535,
      ),
    )
    * -1
  )


# min(rank(decay_linear(((rank(open) + rank(low)) - (rank(high) + rank(close))),8.06882)), Ts_Rank(decay_linear(correlation(Ts_Rank(close, 8.44728), Ts_Rank(adv60,20.6966), 8.01266), 6.65053), 2.61957))
def alpha_088(ctx):
  _CLOSE = ctx("CLOSE")
  return ctx.MIN(
    ctx.RANK(
      ctx.DECAY_LINEAR(
        ctx.RANK(ctx("OPEN"))
        + ctx.RANK(ctx("LOW"))
        - ctx.RANK(ctx("HIGH"))
        + ctx.RANK(_CLOSE),
        8.06882,
      )
    ),
    ctx.TS_RANK(
      ctx.DECAY_LINEAR(
        ctx.CORRELATION(
          ctx.TS_RANK(_CLOSE, 8.44728), ctx.TS_RANK(ctx("ADV60"), 20.6966), 8.01266
        ),
        6.65053,
      ),
      2.61957,
    ),
  )


# (Ts_Rank(decay_linear(correlation(((low * 0.967285) + (low * (1 - 0.967285))), adv10,6.94279), 5.51607), 3.79744) - Ts_Rank(decay_linear(delta(IndNeutralize(vwap,IndClass.industry), 3.48158), 10.1466), 15.3012))
def alpha_089(ctx):
  _LOW = ctx("LOW")
  return ctx.TS_RANK(
    ctx.DECAY_LINEAR(
      ctx.CORRELATION(_LOW * 0.967285 + _LOW * 1 - 0.967285, ctx("ADV10"), 6.94279),
      5.51607,
    ),
    3.79744,
  ) - ctx.TS_RANK(
    ctx.DECAY_LINEAR(
      ctx.DELTA(ctx.INDNEUTRALIZE(ctx("VWAP"), ctx("INDCLASS.INDUSTRY")), 3.48158),
      10.1466,
    ),
    15.3012,
  )


# ((rank((close - ts_max(close, 4.66719)))^Ts_Rank(correlation(IndNeutralize(adv40,IndClass.subindustry), low, 5.38375), 3.21856)) * -1)
def alpha_090(ctx):
  _CLOSE = ctx("CLOSE")
  return (
    np.power(
      ctx.RANK(_CLOSE - ctx.TS_MAX(_CLOSE, 4.66719)),
      ctx.TS_RANK(
        ctx.CORRELATION(
          ctx.INDNEUTRALIZE(ctx("ADV40"), ctx("INDCLASS.SUBINDUSTRY")),
          ctx("LOW"),
          5.38375,
        ),
        3.21856,
      ),
    )
    * -1
  )


# ((Ts_Rank(decay_linear(decay_linear(correlation(IndNeutralize(close,IndClass.industry), volume, 9.74928), 16.398), 3.83219), 4.8667) -rank(decay_linear(correlation(vwap, adv30, 4.01303), 2.6809))) * -1)
def alpha_091(ctx):
  return (
    ctx.TS_RANK(
      ctx.DECAY_LINEAR(
        ctx.DECAY_LINEAR(
          ctx.CORRELATION(
            ctx.INDNEUTRALIZE(ctx("CLOSE"), ctx("INDCLASS.INDUSTRY")),
            ctx("VOLUME"),
            9.74928,
          ),
          16.398,
        ),
        3.83219,
      ),
      4.8667,
    )
    - ctx.RANK(
      ctx.DECAY_LINEAR(ctx.CORRELATION(ctx("VWAP"), ctx("ADV30"), 4.01303), 2.6809)
    )
    * -1
  )


# min(Ts_Rank(decay_linear(((((high + low) / 2) + close) < (low + open)), 14.7221),18.8683), Ts_Rank(decay_linear(correlation(rank(low), rank(adv30), 7.58555), 6.94024),6.80584))
def alpha_092(ctx):
  _LOW = ctx("LOW")
  return ctx.MIN(
    ctx.TS_RANK(
      ctx.DECAY_LINEAR(
        ctx("HIGH") + _LOW / 2 + np.where(ctx("CLOSE") < _LOW, 1.0, 0.0) + ctx("OPEN"),
        14.7221,
      ),
      18.8683,
    ),
    ctx.TS_RANK(
      ctx.DECAY_LINEAR(
        ctx.CORRELATION(ctx.RANK(_LOW), ctx.RANK(ctx("ADV30")), 7.58555), 6.94024
      ),
      6.80584,
    ),
  )


# (Ts_Rank(decay_linear(correlation(IndNeutralize(vwap, IndClass.industry), adv81,17.4193), 19.848), 7.54455) / rank(decay_linear(delta(((close * 0.524434) + (vwap * (1 -0.524434))), 2.77377), 16.2664)))
def alpha_093(ctx):
  _VWAP = ctx("VWAP")
  return ctx.TS_RANK(
    ctx.DECAY_LINEAR(
      ctx.CORRELATION(
        ctx.INDNEUTRALIZE(_VWAP, ctx("INDCLASS.INDUSTRY")), ctx("ADV81"), 17.4193
      ),
      19.848,
    ),
    7.54455,
  ) / ctx.RANK(
    ctx.DECAY_LINEAR(
      ctx.DELTA(ctx("CLOSE") * 0.524434 + _VWAP * 1 - 0.524434, 2.77377), 16.2664
    )
  )


# ((rank((vwap - ts_min(vwap, 11.5783)))^Ts_Rank(correlation(Ts_Rank(vwap,19.6462), Ts_Rank(adv60, 4.02992), 18.0926), 2.70756)) * -1)
def alpha_094(ctx):
  _VWAP = ctx("VWAP")
  return (
    np.power(
      ctx.RANK(_VWAP - ctx.TS_MIN(_VWAP, 11.5783)),
      ctx.TS_RANK(
        ctx.CORRELATION(
          ctx.TS_RANK(_VWAP, 19.6462), ctx.TS_RANK(ctx("ADV60"), 4.02992), 18.0926
        ),
        2.70756,
      ),
    )
    * -1
  )


# (rank((open - ts_min(open, 12.4105))) < Ts_Rank((rank(correlation(sum(((high + low)/ 2), 19.1351), sum(adv40, 19.1351), 12.8742))^5), 11.7584))
def alpha_095(ctx):
  _OPEN = ctx("OPEN")
  return ctx.RANK(_OPEN - ctx.TS_MIN(_OPEN, 12.4105)) < ctx.TS_RANK(
    np.power(
      ctx.RANK(
        ctx.CORRELATION(
          ctx.SUM(ctx("HIGH") + ctx("LOW") / 2, 19.1351),
          ctx.SUM(ctx("ADV40"), 19.1351),
          12.8742,
        )
      ),
      5,
    ),
    11.7584,
  )


# (max(Ts_Rank(decay_linear(correlation(rank(vwap), rank(volume), 3.83878),4.16783), 8.38151), Ts_Rank(decay_linear(Ts_ArgMax(correlation(Ts_Rank(close, 7.45404),Ts_Rank(adv60, 4.13242), 3.65459), 12.6556), 14.0365), 13.4143)) * -1)
def alpha_096(ctx):
  return (
    ctx.MAX(
      ctx.TS_RANK(
        ctx.DECAY_LINEAR(
          ctx.CORRELATION(ctx.RANK(ctx("VWAP")), ctx.RANK(ctx("VOLUME")), 3.83878),
          4.16783,
        ),
        8.38151,
      ),
      ctx.TS_RANK(
        ctx.DECAY_LINEAR(
          ctx.TS_ARGMAX(
            ctx.CORRELATION(
              ctx.TS_RANK(ctx("CLOSE"), 7.45404),
              ctx.TS_RANK(ctx("ADV60"), 4.13242),
              3.65459,
            ),
            12.6556,
          ),
          14.0365,
        ),
        13.4143,
      ),
    )
    * -1
  )


# ((rank(decay_linear(delta(IndNeutralize(((low * 0.721001) + (vwap * (1 - 0.721001))),IndClass.industry), 3.3705), 20.4523)) - Ts_Rank(decay_linear(Ts_Rank(correlation(Ts_Rank(low,7.87871), Ts_Rank(adv60, 17.255), 4.97547), 18.5925), 15.7152), 6.71659)) * -1)
def alpha_097(ctx):
  _LOW = ctx("LOW")
  return (
    ctx.RANK(
      ctx.DECAY_LINEAR(
        ctx.DELTA(
          ctx.INDNEUTRALIZE(
            _LOW * 0.721001 + ctx("VWAP") * 1 - 0.721001, ctx("INDCLASS.INDUSTRY")
          ),
          3.3705,
        ),
        20.4523,
      )
    )
    - ctx.TS_RANK(
      ctx.DECAY_LINEAR(
        ctx.TS_RANK(
          ctx.CORRELATION(
            ctx.TS_RANK(_LOW, 7.87871), ctx.TS_RANK(ctx("ADV60"), 17.255), 4.97547
          ),
          18.5925,
        ),
        15.7152,
      ),
      6.71659,
    )
    * -1
  )


# (rank(decay_linear(correlation(vwap, sum(adv5, 26.4719), 4.58418), 7.18088)) -rank(decay_linear(Ts_Rank(Ts_ArgMin(correlation(rank(open), rank(adv15), 20.8187), 8.62571),6.95668), 8.07206)))
def alpha_098(ctx):
  return ctx.RANK(
    ctx.DECAY_LINEAR(
      ctx.CORRELATION(ctx("VWAP"), ctx.SUM(ctx("ADV5"), 26.4719), 4.58418), 7.18088
    )
  ) - ctx.RANK(
    ctx.DECAY_LINEAR(
      ctx.TS_RANK(
        ctx.TS_ARGMIN(
          ctx.CORRELATION(ctx.RANK(ctx("OPEN")), ctx.RANK(ctx("ADV15")), 20.8187),
          8.62571,
        ),
        6.95668,
      ),
      8.07206,
    )
  )


# ((rank(correlation(sum(((high + low) / 2), 19.8975), sum(adv60, 19.8975), 8.8136)) <rank(correlation(low, volume, 6.28259))) * -1)
def alpha_099(ctx):
  _LOW = ctx("LOW")
  return (
    ctx.RANK(
      ctx.CORRELATION(
        ctx.SUM(ctx("HIGH") + _LOW / 2, 19.8975), ctx.SUM(ctx("ADV60"), 19.8975), 8.8136
      )
    )
    < ctx.RANK(ctx.CORRELATION(_LOW, ctx("VOLUME"), 6.28259)) * -1
  )


# (0 - (1 * (((1.5 * scale(indneutralize(indneutralize(rank(((((close - low) - (high -close)) / (high - low)) * volume)), IndClass.subindustry), IndClass.subindustry))) -scale(indneutralize((correlation(close, rank(adv20), 5) - rank(ts_argmin(close, 30))),IndClass.subindustry))) * (volume / adv20))))
def alpha_100(ctx):
  _ADV20 = ctx("ADV20")
  _CLOSE = ctx("CLOSE")
  _HIGH = ctx("HIGH")
  _INDCLASS_SUBINDUSTRY = ctx("INDCLASS.SUBINDUSTRY")
  _LOW = ctx("LOW")
  _VOLUME = ctx("VOLUME")
  return (
    0
    - 1
    * 1.5
    * ctx.SCALE(
      ctx.INDNEUTRALIZE(
        ctx.INDNEUTRALIZE(
          ctx.RANK(_CLOSE - _LOW - _HIGH - _CLOSE / _HIGH - _LOW * _VOLUME),
          _INDCLASS_SUBINDUSTRY,
        ),
        _INDCLASS_SUBINDUSTRY,
      )
    )
    - ctx.SCALE(
      ctx.INDNEUTRALIZE(
        ctx.CORRELATION(_CLOSE, ctx.RANK(_ADV20), 5)
        - ctx.RANK(ctx.TS_ARGMIN(_CLOSE, 30)),
        _INDCLASS_SUBINDUSTRY,
      )
    )
    * _VOLUME
    / _ADV20
  )


# ((close - open) / ((high - low) + .001))
def alpha_101(ctx):
  return ctx("CLOSE") - ctx("OPEN") / ctx("HIGH") - ctx("LOW") + 0.001
