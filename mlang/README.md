`mlang` is simple language for strategy calculation, it available for many stcok applications, such as THS, TDX, DZH and so on.

# Brief intro

The language is designed to be simple and easy to use, each variable is vector. support common arithmetic operations and logical operations, and it can be used for strategy calculation.

## Syntax rules


### Define variable

```mlang
VAR_NAME := EXPR;
VAR_NAME : EXPR;
```

both kind define a variable, the second kind mean this variable will be exported

### Expression

Function Call and Operator are supported. for a `KDJ` indicator, we can write it as:

```
RSV := (CLOSE - LLV(LOW, 9)) / (HHV(HIGH, 9) - LLV(LOW, 9)) * 100;
K : (REF(RSV, 1) * 2 + RSV) / 3;
D : (REF(K, 1) * 2 + K) / 3;
J : 3 * K - 2 * D;
```

All operators supported and all ta function in this project is supported.


# Usage

This library provide :

- Compile the `mlang` code into lua, so you can embed this into your lua application, see `to_lua` for detail.
- Provide a lightweight runtime for `mlang`, you can use it to execute the `mlang` code, see `MRuntime` for detail.
  - After running the code, it will return a list of exported variables called `line`, you can use it to draw the chart









