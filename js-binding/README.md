# Introduction

This is a javascript binding for mlang, you can use it to execute the `mlang` code to compute the indicator values. Project is wasm-pack based.

# Usage

## install
```
npm install js-alpha-lib
```

## usage

Assuming there is an IKLine definition here

```typescript
type IKLine = {
  time: Float64Array
  open: Float64Array
  high: Float64Array
  low: Float64Array
  close: Float64Array
  volume: Float64Array
  amount: Float64Array
}
```

use this library like this

```typescript
import { execute, JSLine, NamedArray, NamedValue } from 'js-alpha-lib'


function toNamedArray(kline: IKLine) {
  return [
    new NamedArray('open', kline.open),
    new NamedArray('high', kline.high),
    new NamedArray('low', kline.low),
    new NamedArray('close', kline.close),
    new NamedArray('volume', kline.volume),
    new NamedArray('amount', kline.amount),
    new NamedArray('time', kline.time),
  ]
}

function toNamedValues(params?: Record<string, number>) {
  if (!params) return []
  return Object.entries(params).map(([key, value]) => new NamedValue(key, value))
}

export type MLangLine = {
  kind: '' | 'icon' | 'text'
  name: string
  color?: string
  data: Float64Array
  when?: boolean[]
  extra?: string | number
}

function toMlangLine(line: JSLine): MLangLine {
  return {
    kind: line.kind as '' | 'icon' | 'text',
    name: line.name,
    color: line.color,
    data: line.data,
    when: line.when ? Array.from(line.when).map((v) => v > 0) : undefined,
    extra: line.ext_data as string | number,
  }
}

const code = `
MA5: MA(CLOSE, 5);
MA10: MA(CLOSE, 10);
`
const lines = toMLangLine(execute(code, toNamedArray(kline), toNamedValues(params)))

// now lines can be used to plot the chart

```

Because the `JSLine` is on wasm memory, so convert it into a more convenient form for use in the JavaScript environment.

## Additional Utilities

### `join_asof`
Aligns series values `v2` (with dates `d2`) to target dates `d1`.
- `method`: `0` (zero fill), `1` (nan fill), `2` (forward fill / ffill), `-2` (backward fill / bfill)

```typescript
import { join_asof } from 'js-alpha-lib'

const alignedV2 = join_asof(d1, d2, v2, method)
```

### `apply_split`
Performs price split adjustments or split factor calculations.
- `method`: `1` (forward split), `2` (forward split factor), `-1` (backward split), `-2` (backward split factor)
- Note: `dividend`, `transfer_shares`, `right_shares`, and `right_price` must have the same length as `price` (align them with `join_asof(price_dates, event_dates, values, 0)` first if needed).

```typescript
import { apply_split } from 'js-alpha-lib'

const result = apply_split(method, price, dividend, transfer_shares, right_shares, right_price)
```


