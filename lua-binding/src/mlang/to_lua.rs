use anyhow::{Result, anyhow};
use pest::Parser;
use pest::iterators::Pair;
use std::collections::HashSet;

use super::{MLangParser, Rule};

fn find_defined_variables(pair: &Pair<Rule>, vars: &mut HashSet<String>) {
  match pair.as_rule() {
    Rule::var_def | Rule::line_def => {
      if let Some(ident) = pair.clone().into_inner().next() {
        vars.insert(ident.as_str().to_lowercase());
      }
    }
    _ => {
      for child in pair.clone().into_inner() {
        find_defined_variables(&child, vars);
      }
    }
  }
}

fn contains_variable(pair: &Pair<Rule>) -> bool {
  match pair.as_rule() {
    Rule::identifier | Rule::dotted_name | Rule::func_call | Rule::self_kw => true,
    _ => pair
      .clone()
      .into_inner()
      .any(|child| contains_variable(&child)),
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

fn format_color_or_style(val: &str) -> String {
  let trimmed = val.trim();
  if trimmed.starts_with('"') && trimmed.ends_with('"') {
    let content = &trimmed[1..trimmed.len() - 1];
    format!("\"{}\"", content.to_lowercase())
  } else {
    format!("\"{}\"", trimmed.to_lowercase())
  }
}

fn invert_comparison_op(op: &str) -> &str {
  match op {
    "==" => "==",
    "!=" => "!=",
    "<>" => "!=",
    "<" => ">",
    "<=" => ">=",
    ">" => "<",
    ">=" => "<=",
    _ => op,
  }
}

fn map_comparison_method(op: &str) -> &str {
  match op {
    "==" => "eq",
    "!=" | "<>" => "ne",
    "<" => "lt",
    "<=" => "le",
    ">" => "gt",
    ">=" => "ge",
    _ => panic!("Unknown comparison operator: {}", op),
  }
}

fn map_variable_op(op: &str) -> &str {
  match op {
    "&&" => "&",
    "||" => "|",
    _ => op,
  }
}

fn translate_ternary(cond: &str, true_case: &str, false_case: &str) -> String {
  if false_case == "SELF" {
    let trimmed = true_case.trim();

    let check_patterns = |s: &str| -> Option<String> {
      let s = s.trim();
      if s.ends_with("* SELF") {
        let operand = s[..s.len() - 6].trim();
        Some(format!("SCAN_MUL({}, {})", operand, cond))
      } else if s.starts_with("SELF *") {
        let operand = s[6..].trim();
        Some(format!("SCAN_MUL({}, {})", operand, cond))
      } else if s.ends_with("+ SELF") {
        let operand = s[..s.len() - 6].trim();
        Some(format!("SCAN_ADD({}, {})", operand, cond))
      } else if s.starts_with("SELF +") {
        let operand = s[6..].trim();
        Some(format!("SCAN_ADD({}, {})", operand, cond))
      } else {
        None
      }
    };

    if let Some(res) = check_patterns(trimmed) {
      return res;
    }

    if trimmed.starts_with('(') && trimmed.ends_with(')') {
      let inner = &trimmed[1..trimmed.len() - 1];
      if let Some(res) = check_patterns(inner) {
        return res;
      }
    }
  }
  format!("IF({}, {}, {})", cond, true_case, false_case)
}

fn translate_dotted_name(pair: Pair<Rule>, _vars: &HashSet<String>) -> String {
  let full_name = pair.as_str();
  format!("D.{}", full_name.to_uppercase())
}

fn translate_func_call(pair: Pair<Rule>, vars: &HashSet<String>) -> String {
  let mut inner = pair.into_inner();
  let name_ident = inner.next().unwrap();
  let name = name_ident.as_str();

  let args_str = if let Some(args_pair) = inner.next() {
    let mut parts = vec![];
    for arg in args_pair.into_inner() {
      let mut arg_inner = arg.into_inner();
      let first = arg_inner.next().unwrap();
      if first.as_rule() == Rule::identifier && arg_inner.peek().is_some() {
        let arg_name = first.as_str();
        let expr = arg_inner.next().unwrap();
        parts.push(format!("{}={}", arg_name, translate_expr(expr, vars)));
      } else {
        parts.push(translate_expr(first, vars));
      }
    }
    parts.join(", ")
  } else {
    "".to_string()
  };

  format!("{}({})", name, args_str)
}

fn is_draw_func(pair: &Pair<Rule>) -> bool {
  match pair.as_rule() {
    Rule::func_call => {
      let mut inner = pair.clone().into_inner();
      if let Some(ident) = inner.next() {
        let name = ident.as_str();
        return name.to_uppercase().starts_with("DRAW");
      }
      false
    }
    Rule::expr
    | Rule::ternary
    | Rule::logical_or
    | Rule::logical_and
    | Rule::comparison
    | Rule::sum
    | Rule::product
    | Rule::power
    | Rule::atom => {
      let mut inner = pair.clone().into_inner();
      if let (Some(first), None) = (inner.next(), inner.next()) {
        is_draw_func(&first)
      } else {
        false
      }
    }
    _ => false,
  }
}

fn translate_expr(pair: Pair<Rule>, vars: &HashSet<String>) -> String {
  match pair.as_rule() {
    Rule::expr | Rule::ternary => {
      let mut inner = pair.into_inner();
      let first = inner.next().unwrap();
      if let Some(true_case_pair) = inner.next() {
        let false_case_pair = inner.next().unwrap();
        let cond_str = translate_expr(first, vars);
        let true_str = translate_expr(true_case_pair, vars);
        let false_str = translate_expr(false_case_pair, vars);
        translate_ternary(&cond_str, &true_str, &false_str)
      } else {
        translate_expr(first, vars)
      }
    }
    Rule::logical_or
    | Rule::logical_and
    | Rule::comparison
    | Rule::sum
    | Rule::product
    | Rule::power => {
      let rule = pair.as_rule();
      let is_logical_or = rule == Rule::logical_or;
      let is_logical_and = rule == Rule::logical_and;
      let is_product = rule == Rule::product;
      let is_sum = rule == Rule::sum;
      let is_power = rule == Rule::power;

      let mut inner = pair.into_inner();
      let first_pair = inner.next().unwrap();
      let mut accum_str = translate_expr(first_pair.clone(), vars);
      let mut accum_contains_var = contains_variable(&first_pair);
      let mut has_rest = false;

      if is_logical_or || is_logical_and || is_power {
        let op = if is_logical_or {
          "||"
        } else if is_logical_and {
          "&&"
        } else {
          "^"
        };

        while let Some(next_pair) = inner.next() {
          has_rest = true;
          let right_str = translate_expr(next_pair.clone(), vars);
          let right_contains_var = contains_variable(&next_pair);

          if accum_contains_var || right_contains_var {
            if !accum_contains_var && right_contains_var {
              let mapped_op = map_variable_op(op);
              accum_str = format!("{} {} {}", right_str, mapped_op, accum_str);
            } else {
              let mapped_op = map_variable_op(op);
              accum_str = format!("{} {} {}", accum_str, mapped_op, right_str);
            }
          } else {
            let mapped_op = if is_logical_and {
              "and"
            } else if is_logical_or {
              "or"
            } else {
              op
            };
            accum_str = format!("{} {} {}", accum_str, mapped_op, right_str);
          }
          accum_contains_var = accum_contains_var || right_contains_var;
        }

        if has_rest {
          format!("({})", accum_str)
        } else {
          accum_str
        }
      } else {
        while let Some(op_pair) = inner.next() {
          has_rest = true;
          let op = op_pair.as_str();
          let next_pair = inner.next().unwrap();
          let right_str = translate_expr(next_pair.clone(), vars);
          let right_contains_var = contains_variable(&next_pair);

          let is_comparison = op_pair.as_rule() == Rule::comparison_op;

          if accum_contains_var || right_contains_var {
            if !accum_contains_var && right_contains_var {
              let inverted_op = if is_comparison {
                invert_comparison_op(op)
              } else {
                op
              };

              if is_comparison {
                accum_str = format!(
                  "{}:{}({})",
                  right_str,
                  map_comparison_method(inverted_op),
                  accum_str
                );
              } else {
                let mapped_op = map_variable_op(inverted_op);
                accum_str = format!("{} {} {}", right_str, mapped_op, accum_str);
              }
            } else {
              if is_comparison {
                accum_str = format!("{}:{}({})", accum_str, map_comparison_method(op), right_str);
              } else {
                let mapped_op = map_variable_op(op);
                accum_str = format!("{} {} {}", accum_str, mapped_op, right_str);
              }
            }
          } else {
            accum_str = format!("{} {} {}", accum_str, op, right_str);
          }
          accum_contains_var = accum_contains_var || right_contains_var;
        }

        if has_rest && (is_product || is_sum) {
          format!("({})", accum_str)
        } else {
          accum_str
        }
      }
    }
    Rule::atom => translate_expr(pair.into_inner().next().unwrap(), vars),
    Rule::neg => {
      let inner = pair.into_inner().next().unwrap();
      format!("-{}", translate_expr(inner, vars))
    }
    Rule::self_kw => "SELF".to_string(),
    Rule::dotted_name => translate_dotted_name(pair, vars),
    Rule::func_call => translate_func_call(pair, vars),
    Rule::identifier => {
      let name = pair.as_str();
      if name == "SELF" {
        "SELF".to_string()
      } else {
        let name_lower = name.to_lowercase();
        if vars.contains(&name_lower) {
          name_lower
        } else if is_parameter(name) {
          format!("P.{}", name.to_uppercase())
        } else {
          match name.to_uppercase().as_str() {
            "CLOSE" | "C" => "D.C".to_string(),
            "OPEN" | "O" => "D.O".to_string(),
            "HIGH" | "H" => "D.H".to_string(),
            "LOW" | "L" => "D.L".to_string(),
            "VOLUME" | "VOL" | "V" => "D.V".to_string(),
            _ => format!("D.{}", name.to_uppercase()),
          }
        }
      }
    }
    Rule::number | Rule::string => pair.as_str().to_string(),
    _ => pair.as_str().to_string(),
  }
}

pub fn to_lua(code: &str) -> Result<String> {
  if code.trim().is_empty() {
    return Ok(
      r#"function compute(D, P)
  local lines = {}
  return lines
end"#
        .to_string(),
    );
  }

  let parsed = MLangParser::parse(Rule::program, code)
    .map_err(|e| anyhow!("Parse error: {}", e))?
    .next()
    .unwrap();

  let mut defined_vars = HashSet::new();
  find_defined_variables(&parsed, &mut defined_vars);

  let mut body_lines = vec![];
  body_lines.push("  local lines = {}".to_string());

  for stmt in parsed.into_inner() {
    if stmt.as_rule() == Rule::EOI {
      break;
    }

    let stmt_inner = stmt.into_inner().next().unwrap();
    match stmt_inner.as_rule() {
      Rule::var_def => {
        let mut inner = stmt_inner.into_inner();
        let name_ident = inner.next().unwrap();
        let name = name_ident.as_str().to_lowercase();
        let expr = inner.next().unwrap();
        let expr_str = translate_expr(expr, &defined_vars);
        body_lines.push(format!("  local {} = {}", name, expr_str));
      }
      Rule::line_def => {
        let mut inner = stmt_inner.into_inner();
        let name_ident = inner.next().unwrap();
        let name = name_ident.as_str().to_lowercase();
        let expr = inner.next().unwrap();
        let expr_str = translate_expr(expr, &defined_vars);

        let color_opt = inner.next().map(|p| p.as_str());
        let style_opt = inner.next().map(|p| p.as_str());

        body_lines.push(format!("  local {} = {}", name, expr_str));

        let mut table_fields = vec![
          "kind=\"line\"".to_string(),
          format!("name=\"{}\"", name),
          format!("data = {}", name),
        ];
        if let Some(c) = color_opt {
          table_fields.push(format!("color = {}", format_color_or_style(c)));
        }
        if let Some(s) = style_opt {
          table_fields.push(format!("style = {}", format_color_or_style(s)));
        }

        body_lines.push(format!(
          "  table.insert(lines, {{{}}})",
          table_fields.join(", ")
        ));
      }
      Rule::expr_stmt => {
        let expr = stmt_inner.into_inner().next().unwrap();
        let is_draw = is_draw_func(&expr);
        let expr_str = translate_expr(expr, &defined_vars);
        if is_draw {
          body_lines.push(format!("  table.insert(lines, {})", expr_str));
        } else {
          body_lines.push(format!("  {}", expr_str));
        }
      }
      _ => {}
    }
  }

  body_lines.push("  return lines".to_string());

  let mut output = String::new();
  output.push_str("function compute(D, P)\n");
  for line in body_lines {
    output.push_str(&line);
    output.push('\n');
  }
  output.push_str("end");

  Ok(output)
}
