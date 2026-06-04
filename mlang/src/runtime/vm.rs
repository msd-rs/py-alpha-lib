use anyhow::{Result, anyhow};
use pest::iterators::Pair;
use std::collections::HashMap;
use std::cell::RefCell;
use pest::Parser;

use crate::numarray::{NumArray, BoolArray};
use crate::runtime::mvalue::MValue;
use alpha_algo::Context;

#[derive(Debug, Clone, Default)]
pub struct Line {
  pub kind: String,
  pub name: String,
  pub data: Vec<f64>,
  pub color: Option<String>,
  pub when: Option<Vec<bool>>,
  pub ext_data: Option<Vec<u8>>,
}

#[derive(Debug, Clone)]
pub enum Expr {
  Num(f64),
  Str(String),
  SelfKw,
  Identifier(String),
  DottedName(String),
  Neg(Box<Expr>),
  Power(Box<Expr>, Box<Expr>),
  Mul(Box<Expr>, Box<Expr>),
  Div(Box<Expr>, Box<Expr>),
  Rem(Box<Expr>, Box<Expr>),
  Add(Box<Expr>, Box<Expr>),
  Sub(Box<Expr>, Box<Expr>),
  Eq(Box<Expr>, Box<Expr>),
  Ne(Box<Expr>, Box<Expr>),
  Lt(Box<Expr>, Box<Expr>),
  Le(Box<Expr>, Box<Expr>),
  Gt(Box<Expr>, Box<Expr>),
  Ge(Box<Expr>, Box<Expr>),
  And(Box<Expr>, Box<Expr>),
  Or(Box<Expr>, Box<Expr>),
  Ternary {
    cond: Box<Expr>,
    true_case: Box<Expr>,
    false_case: Box<Expr>,
  },
  FuncCall {
    name: String,
    args: Vec<FuncArg>,
  },
  ScanMul {
    operand: Box<Expr>,
    cond: Box<Expr>,
  },
  ScanAdd {
    operand: Box<Expr>,
    cond: Box<Expr>,
  },
}

#[derive(Debug, Clone)]
pub enum FuncArg {
  Unnamed(Expr),
  Named(String, Expr),
}

#[derive(Debug, Clone)]
pub enum Statement {
  VarDef {
    name: String,
    expr: Expr,
  },
  LineDef {
    name: String,
    expr: Expr,
    color: Option<String>,
    style: Option<String>,
  },
  ExprStmt(Expr),
}

fn build_expr(pair: Pair<crate::Rule>) -> Result<Expr> {
  match pair.as_rule() {
    crate::Rule::expr | crate::Rule::ternary => {
      let mut inner = pair.into_inner();
      let first = build_expr(inner.next().unwrap())?;
      if let Some(true_case_pair) = inner.next() {
        let false_case_pair = inner.next().unwrap();
        let true_case = build_expr(true_case_pair)?;
        let false_case = build_expr(false_case_pair)?;
        
        // SELF Pattern rewrite logic
        if let Expr::SelfKw = false_case {
          if let Expr::Mul(left, right) = &true_case {
            if let Expr::SelfKw = **left {
              return Ok(Expr::ScanMul { operand: right.clone(), cond: Box::new(first) });
            } else if let Expr::SelfKw = **right {
              return Ok(Expr::ScanMul { operand: left.clone(), cond: Box::new(first) });
            }
          } else if let Expr::Add(left, right) = &true_case {
            if let Expr::SelfKw = **left {
              return Ok(Expr::ScanAdd { operand: right.clone(), cond: Box::new(first) });
            } else if let Expr::SelfKw = **right {
              return Ok(Expr::ScanAdd { operand: left.clone(), cond: Box::new(first) });
            }
          }
        }
        
        Ok(Expr::Ternary {
          cond: Box::new(first),
          true_case: Box::new(true_case),
          false_case: Box::new(false_case),
        })
      } else {
        Ok(first)
      }
    }
    crate::Rule::logical_or => {
      let mut inner = pair.into_inner();
      let mut accum = build_expr(inner.next().unwrap())?;
      while let Some(next_pair) = inner.next() {
        let right = build_expr(next_pair)?;
        accum = Expr::Or(Box::new(accum), Box::new(right));
      }
      Ok(accum)
    }
    crate::Rule::logical_and => {
      let mut inner = pair.into_inner();
      let mut accum = build_expr(inner.next().unwrap())?;
      while let Some(next_pair) = inner.next() {
        let right = build_expr(next_pair)?;
        accum = Expr::And(Box::new(accum), Box::new(right));
      }
      Ok(accum)
    }
    crate::Rule::comparison => {
      let mut inner = pair.into_inner();
      let mut accum = build_expr(inner.next().unwrap())?;
      while let Some(op_pair) = inner.next() {
        let op = op_pair.as_str();
        let next_pair = inner.next().unwrap();
        let right = build_expr(next_pair)?;
        accum = match op {
          "==" => Expr::Eq(Box::new(accum), Box::new(right)),
          "!=" | "<>" => Expr::Ne(Box::new(accum), Box::new(right)),
          "<" => Expr::Lt(Box::new(accum), Box::new(right)),
          "<=" => Expr::Le(Box::new(accum), Box::new(right)),
          ">" => Expr::Gt(Box::new(accum), Box::new(right)),
          ">=" => Expr::Ge(Box::new(accum), Box::new(right)),
          _ => return Err(anyhow!("Unknown comparison operator: {}", op)),
        };
      }
      Ok(accum)
    }
    crate::Rule::sum => {
      let mut inner = pair.into_inner();
      let mut accum = build_expr(inner.next().unwrap())?;
      while let Some(op_pair) = inner.next() {
        let op = op_pair.as_str();
        let next_pair = inner.next().unwrap();
        let right = build_expr(next_pair)?;
        accum = match op {
          "+" => Expr::Add(Box::new(accum), Box::new(right)),
          "-" => Expr::Sub(Box::new(accum), Box::new(right)),
          _ => return Err(anyhow!("Unknown sum operator: {}", op)),
        };
      }
      Ok(accum)
    }
    crate::Rule::product => {
      let mut inner = pair.into_inner();
      let mut accum = build_expr(inner.next().unwrap())?;
      while let Some(op_pair) = inner.next() {
        let op = op_pair.as_str();
        let next_pair = inner.next().unwrap();
        let right = build_expr(next_pair)?;
        accum = match op {
          "*" => Expr::Mul(Box::new(accum), Box::new(right)),
          "/" => Expr::Div(Box::new(accum), Box::new(right)),
          "%" => Expr::Rem(Box::new(accum), Box::new(right)),
          _ => return Err(anyhow!("Unknown product operator: {}", op)),
        };
      }
      Ok(accum)
    }
    crate::Rule::power => {
      let mut inner = pair.into_inner();
      let mut accum = build_expr(inner.next().unwrap())?;
      while let Some(next_pair) = inner.next() {
        let right = build_expr(next_pair)?;
        accum = Expr::Power(Box::new(accum), Box::new(right));
      }
      Ok(accum)
    }
    crate::Rule::atom => {
      build_expr(pair.into_inner().next().unwrap())
    }
    crate::Rule::neg => {
      let inner = pair.into_inner().next().unwrap();
      Ok(Expr::Neg(Box::new(build_expr(inner)?)))
    }
    crate::Rule::self_kw => Ok(Expr::SelfKw),
    crate::Rule::dotted_name => {
      Ok(Expr::DottedName(pair.as_str().to_string()))
    }
    crate::Rule::func_call => {
      let mut inner = pair.into_inner();
      let name = inner.next().unwrap().as_str().to_string();
      let mut args = vec![];
      if let Some(args_pair) = inner.next() {
        for arg in args_pair.into_inner() {
          let mut arg_inner = arg.into_inner();
          let first = arg_inner.next().unwrap();
          if first.as_rule() == crate::Rule::identifier && arg_inner.peek().is_some() {
            let arg_name = first.as_str().to_string();
            let expr = build_expr(arg_inner.next().unwrap())?;
            args.push(FuncArg::Named(arg_name, expr));
          } else {
            args.push(FuncArg::Unnamed(build_expr(first)?));
          }
        }
      }
      Ok(Expr::FuncCall { name, args })
    }
    crate::Rule::identifier => {
      Ok(Expr::Identifier(pair.as_str().to_string()))
    }
    crate::Rule::number => {
      let val: f64 = pair.as_str().parse()?;
      Ok(Expr::Num(val))
    }
    crate::Rule::string => {
      let val = pair.as_str();
      let content = if val.starts_with('"') && val.ends_with('"') {
        &val[1..val.len() - 1]
      } else {
        val
      };
      Ok(Expr::Str(content.to_string()))
    }
    _ => Err(anyhow!("Unsupported rule in build_expr: {:?}", pair.as_rule())),
  }
}

fn build_statement(pair: Pair<crate::Rule>) -> Result<Statement> {
  let stmt_inner = pair.into_inner().next().unwrap();
  match stmt_inner.as_rule() {
    crate::Rule::var_def => {
      let mut inner = stmt_inner.into_inner();
      let name = inner.next().unwrap().as_str().to_string();
      let expr = build_expr(inner.next().unwrap())?;
      Ok(Statement::VarDef { name, expr })
    }
    crate::Rule::line_def => {
      let mut inner = stmt_inner.into_inner();
      let name = inner.next().unwrap().as_str().to_string();
      let expr = build_expr(inner.next().unwrap())?;
      let color = inner.next().map(|p| {
        let val = p.as_str().trim();
        if val.starts_with('"') && val.ends_with('"') {
          val[1..val.len() - 1].to_lowercase()
        } else {
          val.to_lowercase()
        }
      });
      let style = inner.next().map(|p| {
        let val = p.as_str().trim();
        if val.starts_with('"') && val.ends_with('"') {
          val[1..val.len() - 1].to_lowercase()
        } else {
          val.to_lowercase()
        }
      });
      Ok(Statement::LineDef { name, expr, color, style })
    }
    crate::Rule::expr_stmt => {
      let expr = build_expr(stmt_inner.into_inner().next().unwrap())?;
      Ok(Statement::ExprStmt(expr))
    }
    _ => Err(anyhow!("Unsupported statement rule: {:?}", stmt_inner.as_rule())),
  }
}

fn select_ternary(cond: &BoolArray, true_val: &MValue, false_val: &MValue) -> Result<MValue> {
  let len = cond.len();
  
  let is_numeric = |val: &MValue| match val {
    MValue::Num(_) | MValue::NumArray(_) => true,
    _ => false,
  };
  
  let is_boolean = |val: &MValue| match val {
    MValue::Bool(_) | MValue::BoolArray(_) => true,
    _ => false,
  };
  
  if is_numeric(true_val) && is_numeric(false_val) {
    let mut data = Vec::with_capacity(len);
    for i in 0..len {
      let t = match true_val {
        MValue::Num(n) => *n,
        MValue::NumArray(arr) => arr[i],
        _ => unreachable!(),
      };
      let f = match false_val {
        MValue::Num(n) => *n,
        MValue::NumArray(arr) => arr[i],
        _ => unreachable!(),
      };
      data.push(if cond[i] { t } else { f });
    }
    Ok(MValue::NumArray(NumArray::from(data)))
  } else if is_boolean(true_val) && is_boolean(false_val) {
    let mut data = Vec::with_capacity(len);
    for i in 0..len {
      let t = match true_val {
        MValue::Bool(b) => *b,
        MValue::BoolArray(arr) => arr[i],
        _ => unreachable!(),
      };
      let f = match false_val {
        MValue::Bool(b) => *b,
        MValue::BoolArray(arr) => arr[i],
        _ => unreachable!(),
      };
      data.push(if cond[i] { t } else { f });
    }
    Ok(MValue::BoolArray(BoolArray::from(data)))
  } else {
    Err(anyhow!("Incompatible types in ternary branches: {:?} and {:?}", true_val, false_val))
  }
}

fn is_parameter(s: &str) -> bool {
  if s.len() < 2 {
    return false;
  }
  let mut chars = s.chars();
  let first = chars.next().unwrap();
  (first == 'P' || first == 'p') && chars.all(|c| c.is_ascii_digit())
}

pub struct MRuntime {
  context: Context,
  variables: HashMap<String, MValue>,
  functions: HashMap<String, Box<dyn Fn(&Context, &[MValue], &mut Vec<Line>) -> Result<MValue>>>,
  lines: RefCell<Vec<Line>>,
}

impl MRuntime {
  pub fn new(context: Context) -> Self {
    let mut rt = Self {
      context,
      variables: HashMap::new(),
      functions: HashMap::new(),
      lines: RefCell::new(Vec::new()),
    };
    register_ta_functions(&mut rt);
    rt
  }

  pub fn register_func<F>(&mut self, name: &str, f: F)
  where
    F: Fn(&Context, &[MValue], &mut Vec<Line>) -> Result<MValue> + 'static,
  {
    self.functions.insert(name.to_uppercase(), Box::new(f));
  }

  pub fn lines(&self) -> Vec<Line> {
    self.lines.borrow().clone()
  }

  pub fn variables(&self) -> HashMap<String, MValue> {
    self.variables.clone()
  }

  pub fn execute(
    &mut self,
    code: &str,
    datas: &HashMap<String, NumArray>,
    params: &HashMap<String, f64>,
  ) -> Result<Vec<Line>> {
    self.variables.clear();
    self.lines.borrow_mut().clear();

    if code.trim().is_empty() {
      return Ok(self.lines());
    }

    let parsed = crate::MLangParser::parse(crate::Rule::program, code)
      .map_err(|e| anyhow!("Parse error: {}", e))?
      .next()
      .unwrap();

    let mut statements = vec![];
    for stmt_pair in parsed.into_inner() {
      if stmt_pair.as_rule() == crate::Rule::EOI {
        break;
      }
      statements.push(build_statement(stmt_pair)?);
    }

    for stmt in &statements {
      self.execute_statement(stmt, datas, params)?;
    }

    Ok(self.lines())
  }

  fn execute_statement(
    &mut self,
    stmt: &Statement,
    datas: &HashMap<String, NumArray>,
    params: &HashMap<String, f64>,
  ) -> Result<()> {
    match stmt {
      Statement::VarDef { name, expr } => {
        let val = self.eval_expr(expr, datas, params)?;
        self.variables.insert(name.to_lowercase(), val);
      }
      Statement::LineDef { name, expr, color, style: _ } => {
        let val = self.eval_expr(expr, datas, params)?;
        self.variables.insert(name.to_lowercase(), val.clone());
        
        let default_len = datas.values().next().map(|a| a.len()).unwrap_or(1);
        let arr = val.to_num_array(default_len)?;
        
        let line = Line {
          kind: "line".to_string(),
          name: name.to_lowercase(),
          data: arr.into(),
          color: color.clone(),
          ..Default::default()
        };
        self.lines.borrow_mut().push(line);
      }
      Statement::ExprStmt(expr) => {
        let _ = self.eval_expr(expr, datas, params)?;
      }
    }
    Ok(())
  }

  fn resolve_identifier(
    &self,
    name: &str,
    datas: &HashMap<String, NumArray>,
    params: &HashMap<String, f64>,
  ) -> Result<MValue> {
    let name_lower = name.to_lowercase();
    if let Some(val) = self.variables.get(&name_lower) {
      return Ok(val.clone());
    }

    if is_parameter(name) {
      let name_upper = name.to_uppercase();
      if let Some(&val) = params.get(&name_upper).or_else(|| params.get(&name_lower)) {
        return Ok(MValue::Num(val));
      }
      return Err(anyhow!("Parameter {} not found in params map", name));
    }

    let name_upper = name.to_uppercase();
    let data_key = match name_upper.as_str() {
      "C" | "CLOSE" => "C",
      "O" | "OPEN" => "O",
      "H" | "HIGH" => "H",
      "L" | "LOW" => "L",
      "V" | "VOLUME" | "VOL" => "V",
      _ => name_upper.as_str(),
    };

    if let Some(arr) = datas.get(data_key)
      .or_else(|| datas.get(&data_key.to_lowercase()))
      .or_else(|| datas.get(name))
      .or_else(|| datas.get(&name_lower))
    {
      return Ok(MValue::NumArray(arr.clone()));
    }

    Err(anyhow!("Identifier {} could not be resolved as variable, parameter, or data array", name))
  }

  fn eval_expr(
    &self,
    expr: &Expr,
    datas: &HashMap<String, NumArray>,
    params: &HashMap<String, f64>,
  ) -> Result<MValue> {
    match expr {
      Expr::Num(n) => Ok(MValue::Num(*n)),
      Expr::Str(s) => Ok(MValue::Str(s.clone())),
      Expr::SelfKw => Err(anyhow!("SELF keyword cannot be evaluated directly outside recursive contexts")),
      Expr::Identifier(name) => self.resolve_identifier(name, datas, params),
      Expr::DottedName(name) => self.resolve_identifier(name, datas, params),
      Expr::Neg(inner) => {
        let val = self.eval_expr(inner, datas, params)?;
        val.neg_op()
      }
      Expr::Power(left, right) => {
        let l = self.eval_expr(left, datas, params)?;
        let r = self.eval_expr(right, datas, params)?;
        l.pow(&r)
      }
      Expr::Mul(left, right) => {
        let l = self.eval_expr(left, datas, params)?;
        let r = self.eval_expr(right, datas, params)?;
        l.mul(&r)
      }
      Expr::Div(left, right) => {
        let l = self.eval_expr(left, datas, params)?;
        let r = self.eval_expr(right, datas, params)?;
        l.div(&r)
      }
      Expr::Rem(left, right) => {
        let l = self.eval_expr(left, datas, params)?;
        let r = self.eval_expr(right, datas, params)?;
        l.rem(&r)
      }
      Expr::Add(left, right) => {
        let l = self.eval_expr(left, datas, params)?;
        let r = self.eval_expr(right, datas, params)?;
        l.add(&r)
      }
      Expr::Sub(left, right) => {
        let l = self.eval_expr(left, datas, params)?;
        let r = self.eval_expr(right, datas, params)?;
        l.sub(&r)
      }
      Expr::Eq(left, right) => {
        let l = self.eval_expr(left, datas, params)?;
        let r = self.eval_expr(right, datas, params)?;
        l.eq_op(&r)
      }
      Expr::Ne(left, right) => {
        let l = self.eval_expr(left, datas, params)?;
        let r = self.eval_expr(right, datas, params)?;
        l.ne_op(&r)
      }
      Expr::Lt(left, right) => {
        let l = self.eval_expr(left, datas, params)?;
        let r = self.eval_expr(right, datas, params)?;
        l.lt_op(&r)
      }
      Expr::Le(left, right) => {
        let l = self.eval_expr(left, datas, params)?;
        let r = self.eval_expr(right, datas, params)?;
        l.le_op(&r)
      }
      Expr::Gt(left, right) => {
        let l = self.eval_expr(left, datas, params)?;
        let r = self.eval_expr(right, datas, params)?;
        l.gt_op(&r)
      }
      Expr::Ge(left, right) => {
        let l = self.eval_expr(left, datas, params)?;
        let r = self.eval_expr(right, datas, params)?;
        l.ge_op(&r)
      }
      Expr::And(left, right) => {
        let l = self.eval_expr(left, datas, params)?;
        let r = self.eval_expr(right, datas, params)?;
        l.and_op(&r)
      }
      Expr::Or(left, right) => {
        let l = self.eval_expr(left, datas, params)?;
        let r = self.eval_expr(right, datas, params)?;
        l.or_op(&r)
      }
      Expr::Ternary { cond, true_case, false_case } => {
        let c = self.eval_expr(cond, datas, params)?;
        match c {
          MValue::Bool(b) => {
            if b {
              self.eval_expr(true_case, datas, params)
            } else {
              self.eval_expr(false_case, datas, params)
            }
          }
          MValue::BoolArray(arr) => {
            let t = self.eval_expr(true_case, datas, params)?;
            let f = self.eval_expr(false_case, datas, params)?;
            select_ternary(&arr, &t, &f)
          }
          _ => Err(anyhow!("Ternary condition must evaluate to a boolean or boolean array")),
        }
      }
      Expr::ScanMul { operand, cond } => {
        let cond_val = self.eval_expr(cond, datas, params)?;
        let operand_val = self.eval_expr(operand, datas, params)?;
        
        let len = match (&operand_val, &cond_val) {
          (MValue::NumArray(a), _) => a.len(),
          (_, MValue::BoolArray(b)) => b.len(),
          _ => datas.values().next().map(|a| a.len()).unwrap_or(1),
        };
        
        let cond_arr = cond_val.to_bool_array(len)?;
        let op_arr = operand_val.to_num_array(len)?;
        
        let mut r = vec![0.0; len];
        alpha_algo::ta_scan_mul::<f64>(&self.context, &mut r, &op_arr, &cond_arr)?;
        Ok(MValue::NumArray(NumArray::from(r)))
      }
      Expr::ScanAdd { operand, cond } => {
        let cond_val = self.eval_expr(cond, datas, params)?;
        let operand_val = self.eval_expr(operand, datas, params)?;
        
        let len = match (&operand_val, &cond_val) {
          (MValue::NumArray(a), _) => a.len(),
          (_, MValue::BoolArray(b)) => b.len(),
          _ => datas.values().next().map(|a| a.len()).unwrap_or(1),
        };
        
        let cond_arr = cond_val.to_bool_array(len)?;
        let op_arr = operand_val.to_num_array(len)?;
        
        let mut r = vec![0.0; len];
        alpha_algo::ta_scan_add::<f64>(&self.context, &mut r, &op_arr, &cond_arr)?;
        Ok(MValue::NumArray(NumArray::from(r)))
      }
      Expr::FuncCall { name, args } => {
        let mut evaluated_args = vec![];
        for arg in args {
          match arg {
            FuncArg::Unnamed(expr) => {
              evaluated_args.push(self.eval_expr(expr, datas, params)?);
            }
            FuncArg::Named(_, expr) => {
              evaluated_args.push(self.eval_expr(expr, datas, params)?);
            }
          }
        }
        
        let name_upper = name.to_uppercase();
        if let Some(func) = self.functions.get(&name_upper) {
          let mut lines_borrow = self.lines.borrow_mut();
          func(&self.context, &evaluated_args, &mut *lines_borrow)
        } else {
          Err(anyhow!("Function {} is not registered in the runtime", name))
        }
      }
    }
  }
}

// Macros to register standard TA functions
macro_rules! reg_ta_1_arr_1_usize {
  ($rt:expr, $name:ident, $ta_fn:ident) => {
    $rt.register_func(stringify!($name), |ctx, args, _lines| {
      if args.len() != 2 {
        return Err(anyhow!("{} expects 2 arguments", stringify!($name)));
      }
      let default_len = match &args[0] {
        MValue::NumArray(arr) => arr.len(),
        _ => 1,
      };
      let input = args[0].to_num_array(default_len)?;
      let periods = match &args[1] {
        MValue::Num(n) => *n as usize,
        _ => return Err(anyhow!("{} second argument must be a number", stringify!($name))),
      };
      let mut r = vec![0.0; input.len()];
      $ta_fn::<f64>(ctx, &mut r, &input, periods)?;
      Ok(MValue::NumArray(NumArray::from(r)))
    });
  };
}

macro_rules! reg_ta_1_bool_arr_1_usize {
  ($rt:expr, $name:ident, $ta_fn:ident) => {
    $rt.register_func(stringify!($name), |ctx, args, _lines| {
      if args.len() != 2 {
        return Err(anyhow!("{} expects 2 arguments", stringify!($name)));
      }
      let default_len = match &args[0] {
        MValue::BoolArray(arr) => arr.len(),
        _ => 1,
      };
      let input = args[0].to_bool_array(default_len)?;
      let periods = match &args[1] {
        MValue::Num(n) => *n as usize,
        _ => return Err(anyhow!("{} second argument must be a number", stringify!($name))),
      };
      let mut r = vec![0.0; input.len()];
      $ta_fn::<f64>(ctx, &mut r, &input, periods)?;
      Ok(MValue::NumArray(NumArray::from(r)))
    });
  };
}

macro_rules! reg_ta_1_bool_arr {
  ($rt:expr, $name:ident, $ta_fn:ident) => {
    $rt.register_func(stringify!($name), |ctx, args, _lines| {
      if args.len() != 1 {
        return Err(anyhow!("{} expects 1 argument", stringify!($name)));
      }
      let default_len = match &args[0] {
        MValue::BoolArray(arr) => arr.len(),
        _ => 1,
      };
      let input = args[0].to_bool_array(default_len)?;
      let mut r = vec![0.0; input.len()];
      $ta_fn::<f64>(ctx, &mut r, &input)?;
      Ok(MValue::NumArray(NumArray::from(r)))
    });
  };
}

macro_rules! reg_ta_1_arr {
  ($rt:expr, $name:ident, $ta_fn:ident) => {
    $rt.register_func(stringify!($name), |ctx, args, _lines| {
      if args.len() != 1 {
        return Err(anyhow!("{} expects 1 argument", stringify!($name)));
      }
      let default_len = match &args[0] {
        MValue::NumArray(arr) => arr.len(),
        _ => 1,
      };
      let input = args[0].to_num_array(default_len)?;
      let mut r = vec![0.0; input.len()];
      $ta_fn::<f64>(ctx, &mut r, &input)?;
      Ok(MValue::NumArray(NumArray::from(r)))
    });
  };
}

macro_rules! reg_ta_2_arr_1_usize {
  ($rt:expr, $name:ident, $ta_fn:ident) => {
    $rt.register_func(stringify!($name), |ctx, args, _lines| {
      if args.len() != 3 {
        return Err(anyhow!("{} expects 3 arguments", stringify!($name)));
      }
      let default_len = match (&args[0], &args[1]) {
        (MValue::NumArray(a), _) => a.len(),
        (_, MValue::NumArray(b)) => b.len(),
        _ => 1,
      };
      let x = args[0].to_num_array(default_len)?;
      let y = args[1].to_num_array(default_len)?;
      let periods = match &args[2] {
        MValue::Num(n) => *n as usize,
        _ => return Err(anyhow!("{} third argument must be a number", stringify!($name))),
      };
      let mut r = vec![0.0; x.len()];
      $ta_fn::<f64>(ctx, &mut r, &x, &y, periods)?;
      Ok(MValue::NumArray(NumArray::from(r)))
    });
  };
}

macro_rules! reg_ta_2_arr_to_bool {
  ($rt:expr, $name:ident, $ta_fn:ident) => {
    $rt.register_func(stringify!($name), |ctx, args, _lines| {
      if args.len() != 2 {
        return Err(anyhow!("{} expects 2 arguments", stringify!($name)));
      }
      let default_len = match (&args[0], &args[1]) {
        (MValue::NumArray(a), _) => a.len(),
        (_, MValue::NumArray(b)) => b.len(),
        _ => 1,
      };
      let a = args[0].to_num_array(default_len)?;
      let b = args[1].to_num_array(default_len)?;
      let mut r = vec![false; a.len()];
      $ta_fn::<f64>(ctx, &mut r, &a, &b)?;
      Ok(MValue::BoolArray(BoolArray::from(r)))
    });
  };
}

macro_rules! reg_ta_1_arr_1_f64 {
  ($rt:expr, $name:ident, $ta_fn:ident) => {
    $rt.register_func(stringify!($name), |ctx, args, _lines| {
      if args.len() != 2 {
        return Err(anyhow!("{} expects 2 arguments", stringify!($name)));
      }
      let default_len = match &args[0] {
        MValue::NumArray(arr) => arr.len(),
        _ => 1,
      };
      let input = args[0].to_num_array(default_len)?;
      let val = match &args[1] {
        MValue::Num(n) => *n,
        _ => return Err(anyhow!("{} second argument must be a number", stringify!($name))),
      };
      let mut r = vec![0.0; input.len()];
      $ta_fn::<f64>(ctx, &mut r, &input, val)?;
      Ok(MValue::NumArray(NumArray::from(r)))
    });
  };
}

macro_rules! reg_ta_1_arr_2_usize {
  ($rt:expr, $name:ident, $ta_fn:ident) => {
    $rt.register_func(stringify!($name), |ctx, args, _lines| {
      if args.len() != 3 {
        return Err(anyhow!("{} expects 3 arguments", stringify!($name)));
      }
      let default_len = match &args[0] {
        MValue::NumArray(arr) => arr.len(),
        _ => 1,
      };
      let input = args[0].to_num_array(default_len)?;
      let val1 = match &args[1] {
        MValue::Num(n) => *n as usize,
        _ => return Err(anyhow!("{} second argument must be a number", stringify!($name))),
      };
      let val2 = match &args[2] {
        MValue::Num(n) => *n as usize,
        _ => return Err(anyhow!("{} third argument must be a number", stringify!($name))),
      };
      let mut r = vec![0.0; input.len()];
      $ta_fn::<f64>(ctx, &mut r, &input, val1, val2)?;
      Ok(MValue::NumArray(NumArray::from(r)))
    });
  };
}

macro_rules! reg_ta_1_arr_1_usize_other {
  ($rt:expr, $name:ident, $ta_fn:ident) => {
    $rt.register_func(stringify!($name), |ctx, args, _lines| {
      if args.len() != 2 {
        return Err(anyhow!("{} expects 2 arguments", stringify!($name)));
      }
      let default_len = match &args[0] {
        MValue::NumArray(arr) => arr.len(),
        _ => 1,
      };
      let input = args[0].to_num_array(default_len)?;
      let val = match &args[1] {
        MValue::Num(n) => *n as usize,
        _ => return Err(anyhow!("{} second argument must be a number", stringify!($name))),
      };
      let mut r = vec![0.0; input.len()];
      $ta_fn::<f64>(ctx, &mut r, &input, val)?;
      Ok(MValue::NumArray(NumArray::from(r)))
    });
  };
}

macro_rules! reg_ta_2_arr {
  ($rt:expr, $name:ident, $ta_fn:ident) => {
    $rt.register_func(stringify!($name), |ctx, args, _lines| {
      if args.len() != 2 {
        return Err(anyhow!("{} expects 2 arguments", stringify!($name)));
      }
      let default_len = match (&args[0], &args[1]) {
        (MValue::NumArray(a), _) => a.len(),
        (_, MValue::NumArray(b)) => b.len(),
        _ => 1,
      };
      let category = args[0].to_num_array(default_len)?;
      let input = args[1].to_num_array(default_len)?;
      let mut r = vec![0.0; category.len()];
      $ta_fn::<f64>(ctx, &mut r, &category, &input)?;
      Ok(MValue::NumArray(NumArray::from(r)))
    });
  };
}

macro_rules! reg_ta_1_arr_1_usize_1_f64 {
  ($rt:expr, $name:ident, $ta_fn:ident) => {
    $rt.register_func(stringify!($name), |ctx, args, _lines| {
      if args.len() != 3 {
        return Err(anyhow!("{} expects 3 arguments", stringify!($name)));
      }
      let default_len = match &args[0] {
        MValue::NumArray(arr) => arr.len(),
        _ => 1,
      };
      let input = args[0].to_num_array(default_len)?;
      let periods = match &args[1] {
        MValue::Num(n) => *n as usize,
        _ => return Err(anyhow!("{} second argument must be a number", stringify!($name))),
      };
      let q = match &args[2] {
        MValue::Num(n) => *n,
        _ => return Err(anyhow!("{} third argument must be a number", stringify!($name))),
      };
      let mut r = vec![0.0; input.len()];
      $ta_fn::<f64>(ctx, &mut r, &input, periods, q)?;
      Ok(MValue::NumArray(NumArray::from(r)))
    });
  };
}

macro_rules! reg_ta_2_arr_1_usize_to_bool {
  ($rt:expr, $name:ident, $ta_fn:ident) => {
    $rt.register_func(stringify!($name), |ctx, args, _lines| {
      if args.len() != 3 {
        return Err(anyhow!("{} expects 3 arguments", stringify!($name)));
      }
      let default_len = match (&args[0], &args[1]) {
        (MValue::NumArray(a), _) => a.len(),
        (_, MValue::NumArray(b)) => b.len(),
        _ => 1,
      };
      let a = args[0].to_num_array(default_len)?;
      let b = args[1].to_num_array(default_len)?;
      let n = match &args[2] {
        MValue::Num(num) => *num as usize,
        _ => return Err(anyhow!("{} third argument must be a number", stringify!($name))),
      };
      let mut r = vec![false; a.len()];
      $ta_fn::<f64>(ctx, &mut r, &a, &b, n)?;
      Ok(MValue::BoolArray(BoolArray::from(r)))
    });
  };
}

macro_rules! reg_ta_1_arr_1_bool_arr_1_usize {
  ($rt:expr, $name:ident, $ta_fn:ident) => {
    $rt.register_func(stringify!($name), |ctx, args, _lines| {
      if args.len() != 3 {
        return Err(anyhow!("{} expects 3 arguments", stringify!($name)));
      }
      let default_len = match (&args[0], &args[1]) {
        (MValue::NumArray(a), _) => a.len(),
        (_, MValue::BoolArray(b)) => b.len(),
        _ => 1,
      };
      let input = args[0].to_num_array(default_len)?;
      let condition = args[1].to_bool_array(default_len)?;
      let periods = match &args[2] {
        MValue::Num(n) => *n as usize,
        _ => return Err(anyhow!("{} third argument must be a number", stringify!($name))),
      };
      let mut r = vec![0.0; input.len()];
      $ta_fn::<f64>(ctx, &mut r, &input, &condition, periods)?;
      Ok(MValue::NumArray(NumArray::from(r)))
    });
  };
}

macro_rules! reg_ta_open_close_calc_delay_periods {
  ($rt:expr, $name:ident, $ta_fn:ident) => {
    $rt.register_func(stringify!($name), |ctx, args, _lines| {
      if args.len() != 5 {
        return Err(anyhow!("{} expects 5 arguments", stringify!($name)));
      }
      let default_len = match (&args[0], &args[1], &args[2]) {
        (MValue::NumArray(a), _, _) => a.len(),
        (_, MValue::NumArray(b), _) => b.len(),
        (_, _, MValue::NumArray(c)) => c.len(),
        _ => 1,
      };
      let open = args[0].to_num_array(default_len)?;
      let close = args[1].to_num_array(default_len)?;
      let is_calc = args[2].to_num_array(default_len)?;
      let delay = match &args[3] {
        MValue::Num(n) => *n as usize,
        _ => return Err(anyhow!("{} fourth argument must be a number", stringify!($name))),
      };
      let periods = match &args[4] {
        MValue::Num(n) => *n as usize,
        _ => return Err(anyhow!("{} fifth argument must be a number", stringify!($name))),
      };
      let mut r = vec![0.0; open.len()];
      $ta_fn::<f64>(ctx, &mut r, &open, &close, &is_calc, delay, periods)?;
      Ok(MValue::NumArray(NumArray::from(r)))
    });
  };
}

macro_rules! reg_ta_1_arr_2_usize_sma {
  ($rt:expr, $name:ident, $ta_fn:ident) => {
    $rt.register_func(stringify!($name), |ctx, args, _lines| {
      if args.len() != 3 {
        return Err(anyhow!("{} expects 3 arguments", stringify!($name)));
      }
      let default_len = match &args[0] {
        MValue::NumArray(arr) => arr.len(),
        _ => 1,
      };
      let input = args[0].to_num_array(default_len)?;
      let n = match &args[1] {
        MValue::Num(val) => *val as usize,
        _ => return Err(anyhow!("{} second argument must be a number", stringify!($name))),
      };
      let m = match &args[2] {
        MValue::Num(val) => *val as usize,
        _ => return Err(anyhow!("{} third argument must be a number", stringify!($name))),
      };
      let mut r = vec![0.0; input.len()];
      $ta_fn::<f64>(ctx, &mut r, &input, n, m)?;
      Ok(MValue::NumArray(NumArray::from(r)))
    });
  };
}

pub fn register_ta_functions(rt: &mut MRuntime) {
  use alpha_algo::*;

  reg_ta_2_arr_1_usize!(rt, ALPHA, ta_alpha);
  reg_ta_1_arr!(rt, BACKFILL, ta_backfill);
  reg_ta_1_bool_arr!(rt, BARSLAST, ta_barslast);
  reg_ta_1_bool_arr!(rt, BARSSINCE, ta_barssince);
  reg_ta_2_arr_1_usize!(rt, BETA, ta_beta);
  reg_ta_1_arr_1_usize_other!(rt, BINS, ta_bins);
  reg_ta_1_arr!(rt, CC_RANK, ta_cc_rank);
  reg_ta_1_arr!(rt, CC_ZSCORE, ta_cc_zscore);
  reg_ta_1_arr_1_usize!(rt, CORR, ta_corr);
  reg_ta_2_arr_1_usize!(rt, CORR2, ta_corr2);
  reg_ta_1_bool_arr_1_usize!(rt, COUNT, ta_count);
  reg_ta_1_arr_1_usize!(rt, COUNT_NANS, ta_count_nans);
  reg_ta_2_arr_1_usize!(rt, COV, ta_cov);
  reg_ta_2_arr_to_bool!(rt, CROSS, ta_cross);
  reg_ta_1_arr_1_f64!(rt, DMA, ta_dma);
  reg_ta_1_arr_1_usize!(rt, EMA, ta_ema);
  reg_ta_1_arr_2_usize!(rt, ENTROPY, ta_entropy);
  reg_ta_open_close_calc_delay_periods!(rt, FRET, ta_fret);
  reg_ta_2_arr!(rt, GROUP_RANK, ta_group_rank);
  reg_ta_2_arr!(rt, GROUP_ZSCORE, ta_group_zscore);
  reg_ta_1_arr_1_usize!(rt, HHV, ta_hhv);
  reg_ta_1_arr_1_usize!(rt, HHVBARS, ta_hhvbars);
  reg_ta_1_arr_1_usize!(rt, INTERCEPT, ta_intercept);
  reg_ta_1_arr_1_usize!(rt, KURTOSIS, ta_kurtosis);
  reg_ta_1_arr_1_usize!(rt, LLV, ta_llv);
  reg_ta_1_arr_1_usize!(rt, LLVBARS, ta_llvbars);
  reg_ta_2_arr_1_usize_to_bool!(rt, LONGCROSS, ta_longcross);
  reg_ta_1_arr_1_usize!(rt, LWMA, ta_lwma);
  reg_ta_1_arr_1_usize!(rt, MA, ta_ma);
  reg_ta_1_arr_1_usize!(rt, MAX_DRAWDOWN, ta_max_drawdown);
  reg_ta_1_arr_1_usize!(rt, MIN_MAX_DIFF, ta_min_max_diff);
  reg_ta_1_arr_2_usize!(rt, MOMENT, ta_moment);
  reg_ta_2_arr!(rt, NEUTRALIZE, ta_neutralize);
  reg_ta_1_arr_1_usize!(rt, PRODUCT, ta_product);
  reg_ta_1_arr_1_usize_1_f64!(rt, QUANTILE, ta_quantile);
  reg_ta_1_arr_1_usize!(rt, RANK, ta_rank);
  reg_ta_2_arr_to_bool!(rt, RCROSS, ta_rcross);
  reg_ta_1_arr_1_usize!(rt, REF, ta_ref);
  reg_ta_2_arr_1_usize!(rt, REGBETA, ta_regbeta);
  reg_ta_2_arr_1_usize!(rt, REGRESI, ta_regresi);
  reg_ta_2_arr_1_usize_to_bool!(rt, RLONGCROSS, ta_rlongcross);
  
  rt.register_func("SCAN_ADD", |ctx, args, _lines| {
    if args.len() != 2 { return Err(anyhow!("SCAN_ADD expects 2 arguments")); }
    let default_len = match (&args[0], &args[1]) {
      (MValue::NumArray(a), _) => a.len(),
      (_, MValue::BoolArray(b)) => b.len(),
      _ => 1,
    };
    let input = args[0].to_num_array(default_len)?;
    let cond = args[1].to_bool_array(default_len)?;
    let mut r = vec![0.0; input.len()];
    ta_scan_add::<f64>(ctx, &mut r, &input, &cond)?;
    Ok(MValue::NumArray(NumArray::from(r)))
  });

  rt.register_func("SCAN_MUL", |ctx, args, _lines| {
    if args.len() != 2 { return Err(anyhow!("SCAN_MUL expects 2 arguments")); }
    let default_len = match (&args[0], &args[1]) {
      (MValue::NumArray(a), _) => a.len(),
      (_, MValue::BoolArray(b)) => b.len(),
      _ => 1,
    };
    let input = args[0].to_num_array(default_len)?;
    let cond = args[1].to_bool_array(default_len)?;
    let mut r = vec![0.0; input.len()];
    ta_scan_mul::<f64>(ctx, &mut r, &input, &cond)?;
    Ok(MValue::NumArray(NumArray::from(r)))
  });

  reg_ta_1_arr_1_usize!(rt, SHARPE, ta_sharpe);
  reg_ta_1_arr_1_usize!(rt, SKEWNESS, ta_skewness);
  reg_ta_1_arr_1_usize!(rt, SLOPE, ta_slope);
  reg_ta_1_arr_2_usize_sma!(rt, SMA, ta_sma);
  reg_ta_1_arr_1_usize!(rt, STDDEV, ta_stddev);
  reg_ta_1_arr_1_usize!(rt, SUM, ta_sum);
  reg_ta_1_arr_1_f64!(rt, SUMBARS, ta_sumbars);
  reg_ta_1_arr_1_bool_arr_1_usize!(rt, SUMIF, ta_sumif);
  reg_ta_1_arr_1_usize!(rt, VAR, ta_var);
  reg_ta_1_arr_1_usize!(rt, WEIGHTED_DELAY, ta_weighted_delay);
  reg_ta_1_arr_1_usize!(rt, ZSCORE, ta_zscore);

  rt.register_func("DRAWICON", |_ctx, args, lines| {
    if args.len() != 3 { return Err(anyhow!("DRAWICON expects 3 arguments")); }
    let default_len = match (&args[0], &args[1]) {
      (MValue::BoolArray(a), _) => a.len(),
      (_, MValue::NumArray(b)) => b.len(),
      _ => 1,
    };
    let when = args[0].to_bool_array(default_len)?;
    let pos = args[1].to_num_array(default_len)?;
    let icon = match &args[2] {
      MValue::Num(n) => *n as u32,
      _ => return Err(anyhow!("DRAWICON third argument must be a number")),
    };
    lines.push(Line {
      kind: "icon".to_string(),
      name: "icon".to_string(),
      data: pos.into(),
      when: Some(when.into()),
      ext_data: Some(icon.to_le_bytes().to_vec()),
      ..Default::default()
    });
    Ok(MValue::Num(0.0))
  });

  rt.register_func("DRAWTEXT", |_ctx, args, lines| {
    if args.len() != 3 { return Err(anyhow!("DRAWTEXT expects 3 arguments")); }
    let default_len = match (&args[0], &args[1]) {
      (MValue::BoolArray(a), _) => a.len(),
      (_, MValue::NumArray(b)) => b.len(),
      _ => 1,
    };
    let when = args[0].to_bool_array(default_len)?;
    let pos = args[1].to_num_array(default_len)?;
    let text = match &args[2] {
      MValue::Str(s) => s.clone(),
      MValue::Num(n) => n.to_string(),
      _ => return Err(anyhow!("DRAWTEXT third argument must be a string or number")),
    };
    lines.push(Line {
      kind: "text".to_string(),
      name: "text".to_string(),
      data: pos.into(),
      when: Some(when.into()),
      ext_data: Some(text.as_bytes().to_vec()),
      ..Default::default()
    });
    Ok(MValue::Num(0.0))
  });
}
