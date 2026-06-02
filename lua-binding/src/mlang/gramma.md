这是一个将股票公式转换成lua代码的解析器

# 公式的语法

- 变量定义: `NAME := EXPRESSION;`
- 线条定义: `NAME : EXPRESSION [,"COLOR", LINE_STYLE];`
- 表达式: 支持算术运算、逻辑运算, 函数调用, 以 ; 结尾
- 支持多行


示例：

```
MA5 := MA(C,5);
MA10 : MA(C,P1), RED;
UP:=CROSS(MA5,MA10);
DRAWICON(UP,LOW,1);
```

# 转换规则

## 转换后的函数签名
输出一个函数定义, 其原型如下
```lua
function compute(D, P)
  local lines = {}
  -- TODO: 实际的算法

  return lines
end
```
其中, D 是传入的数据, 它是一个table, 所有不是在算法中定义的变量, 可以通过 `D.varname` 来引用。
P 是传入的参数, 它是一个table, 所有的参数, 可以在 `P.varname` 来引用, 算法中P开头数字结尾的变量视为参数

## 线条定义
线条定义的会加入到 `lines` 数组中, 通过 `table.insert`, 值是一个table
- kind: 必填, 表示线的类型, 默认 折线(line)
- name: 必填, 变量名
- data: 必填, 表达式计算后的结果
- color: 颜色, 可以是字符串或者颜色枚举

DRAW开头的函数调用的结果总是会添加到 `lines` 数组中

## 比较运算的特殊规则

- 不要使用`==`，`!=`，`<>`，`<=`，`>=`，`<`，`>`，而是使用变量的eq，ne，le，lt，ge，gt方法
- 例如: `a == b` 应该转换为 `a:eq(b)`

## 其他运算, 直接使用运算符, 例如 +、-、*、/、%等

- 特殊情况: 变量和数字的运算, 变量总应该在左边, 例如 `1 + a` 应该转换为 `a + 1`

经过以上的说明, 示例转换后的lua代码如下

```lua
function compute(D, P)
  local lines = {}
  local ma5 = MA(D.C, 5)
  local ma10 = MA(D.C, P.P1)
  table.insert(lines, {kind="line", name="ma10", data = ma10, color = "red"})
  local up = CROSS(ma5, ma10)
  table.insert(lines, DRWAICON(up, D.L, 1))
  return lines
end
```



