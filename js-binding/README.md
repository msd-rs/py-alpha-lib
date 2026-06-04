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


