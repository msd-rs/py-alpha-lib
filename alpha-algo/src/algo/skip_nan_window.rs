// Copyright 2026 MSD-RS Project LiJia
// SPDX-License-Identifier: BSD-2-Clause

//! Skip nan window iterator, it keep the `window` size of no-nan values

use num_traits::Float;

/// Item of the window iterator
#[derive(Debug, Clone, Default, Copy, PartialEq)]
pub struct Item {
  /// Index of current start, ensure data[start] is normal
  pub start: usize,
  /// Index of previous start, ensure data[prev_start] is normal
  pub prev_start: usize,
  /// Index of current end, ensure data[end] is normal
  pub end: usize,
  /// Number of non-nan values in the window
  pub no_nan_count: usize,
}

impl Item {
  pub fn new(start: usize, prev_start: usize, end: usize, no_nan_count: usize) -> Self {
    Item {
      start,
      prev_start,
      end,
      no_nan_count,
    }
  }
}

/// Iterator over a window of data
///
/// The iterator yields tuples of (start, end, last_is_valid_number)
/// start always points to the first non-nan value in the window
/// end always points to the last value in the window, may be nan
/// in the range [start, end] there are mostly `window` values, but may be less
#[derive(Debug, Clone)]
pub struct SkipNanWindow<'a, NumT> {
  /// The size of the window
  window: usize,
  /// The data to iterate over
  data: &'a [NumT],
  /// The current item
  item: Item,
  /// Current cursor position in data
  cursor: usize,
}

impl<'a, NumT> SkipNanWindow<'a, NumT> {
  pub fn new(data: &'a [NumT], window: usize, skip: usize) -> Self {
    SkipNanWindow {
      window,
      data,
      item: Item::new(skip, skip, skip, 0),
      cursor: skip,
    }
  }
}

impl<NumT: Float> Iterator for SkipNanWindow<'_, NumT> {
  type Item = Item;

  fn next(&mut self) -> Option<Self::Item> {
    if self.cursor >= self.data.len() {
      return None;
    }

    // Update prev_start before modifying start
    self.item.prev_start = self.item.start;

    let val = self.data[self.cursor];
    // Check if the current value is valid (not NaN)
    if !val.is_nan() {
      self.item.no_nan_count += 1;
    }

    // Shrink window if needed (if valid count exceeds window size)
    while self.item.no_nan_count > self.window {
      if !self.data[self.item.start].is_nan() {
        self.item.no_nan_count -= 1;
      }
      self.item.start += 1;
    }

    // Ensure start points to the first non-nan value or catches up to cursor
    while self.item.start <= self.cursor && self.data[self.item.start].is_nan() {
      self.item.start += 1;
    }

    self.item.end = self.cursor;
    self.cursor += 1;

    Some(self.item)
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_skip_nan_window() {
    let data = vec![0.0, 1.0, f64::NAN, 3.0, 4.0, 5.0, 6.0, 7.0];
    let window = 3;
    let mut iter = SkipNanWindow::new(&data, window, 0);
    let mut items = vec![];
    while let Some(item) = iter.next() {
      items.push(item);
    }

    assert_eq!(items.len(), 8);
    assert_eq!(items[0], Item::new(0, 0, 0, 1));
    assert_eq!(items[1], Item::new(0, 0, 1, 2));
    assert_eq!(items[2], Item::new(0, 0, 2, 2));
    assert_eq!(items[3], Item::new(0, 0, 3, 3));
    assert_eq!(items[4], Item::new(1, 0, 4, 3));
    assert_eq!(items[5], Item::new(3, 1, 5, 3));
    assert_eq!(items[6], Item::new(4, 3, 6, 3));
    assert_eq!(items[7], Item::new(5, 4, 7, 3));
  }

  #[test]
  fn test_skip_nan_window_with_skip() {
    let data = vec![0.0, 1.0, f64::NAN, 3.0, 4.0, 5.0, 6.0, 7.0];
    let window = 3;
    let skip = 2;
    let mut iter = SkipNanWindow::new(&data, window, skip);
    let mut items = vec![];
    while let Some(item) = iter.next() {
      items.push(item);
    }

    assert_eq!(items.len(), 6);
    assert_eq!(items[0], Item::new(3, 2, 2, 0)); // The NAN skip
    assert_eq!(items[1], Item::new(3, 3, 3, 1)); // 3
    assert_eq!(items[2], Item::new(3, 3, 4, 2)); // 3, 4
    assert_eq!(items[3], Item::new(3, 3, 5, 3)); // 3, 4, 5
    assert_eq!(items[4], Item::new(4, 3, 6, 3)); // 4, 5, 6
    assert_eq!(items[5], Item::new(5, 4, 7, 3)); // 5, 6, 7
  }
}
