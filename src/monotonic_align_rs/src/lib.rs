use itertools;
use pyo3::prelude::*;
use std::cmp;

extern crate rayon;
use rayon::prelude::*;

fn maximum_path_each(
    path: &mut Vec<Vec<i32>>,
    value: &mut Vec<Vec<f32>>,
    t_y: &mut i32,
    t_x: &mut i32,
) {
    let max_neg_val = -1e9;

    let mut v_prev: f32;
    let mut v_cur: f32;

    for y in 0..*t_y {
        for x in cmp::max(0, *t_x + y - *t_y)..cmp::min(*t_x, y + 1) {
            if x == y {
                v_cur = max_neg_val;
            } else {
                v_cur = value[(y - 1) as usize][x as usize];
            }

            if x == 0 {
                if y == 0 {
                    v_prev = 0.;
                } else {
                    v_prev = max_neg_val;
                }
            } else {
                v_prev = value[(y - 1) as usize][(x - 1) as usize];
            }
            value[y as usize][x as usize] += if v_prev > v_cur { v_prev } else { v_cur };
        }
    }

    let mut index = (*t_x - 1) as usize;
    for y in (0..(*t_y as usize)).rev() {
        path[y][index] = 1;
        if index != 0
            && (index == y || value[(y - 1) as usize][index] < value[(y - 1) as usize][index - 1])
        {
            index = index - 1
        }
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn monotonic_align(_py: Python, m: &PyModule) -> PyResult<()> {
    #[pyfn(m)]
    fn maximum_path(
        paths: Vec<Vec<Vec<i32>>>,
        values: Vec<Vec<Vec<f32>>>,
        t_ys: Vec<i32>,
        t_xs: Vec<i32>,
    ) -> PyResult<()> {
        itertools::izip!(paths, values, t_ys, t_xs)
            .collect::<Vec<_>>()
            .par_iter_mut()
            .map(|(path, value, t_y, t_x)| maximum_path_each(path, value, t_y, t_x))
            .collect::<Vec<_>>();
        Ok(())
    }
    Ok(())
}
