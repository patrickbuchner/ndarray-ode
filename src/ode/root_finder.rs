use crate::prelude::*;
use ndarray::{Array1, Array2};
use ndarray_linalg::{error::Result, Inverse, Norm};

/// Newton method for iteratively finding the next state for our problem.
#[allow(non_snake_case)]
#[inline]
pub fn newton<Res>(
    rtol: f64,
    residual: &Res,
    mut x1: Array1<AD>,
    J: &mut Array2<f64>,
    slope_buffer: &mut Array1<AD>,
) -> Result<Array1<AD>>
where
    Res: Residual + std::marker::Sync,
{
    // println!("x1: {x1:?}");
    // 2. Obtain Jacobian
    jacobian_res(residual, x1.to_f64().view(), J, slope_buffer);
    // println!("J: {:?}", J);
    let mut DG_inv = J.inv()?;
    // println!("j^-1: {:?}", DG_inv);
    // x1 - (x0+h*f(x1)) = g(x1) != 0
    let mut G = x1.clone();
    residual.eval(x1.view(), &mut G);
    // println!("G: {:?}", G);
    let mut num_iter: usize = 0;
    let mut err = G.to_f64().norm();
    // 3. Iteration
    while err >= rtol && num_iter <= 10 {
        // prod = f'^-1 * f(x1) = f(x1)/ f'(x1)
        // Newton manipulation
        let DGG = DG_inv.dot(&G.to_f64());
        // x1_new = x1 - f(x1)/f'(x1)
        // println!("DGG: {:?}", DGG);
        // println!("x1: {x1:?}");
        x1 = x1 - DGG;
        // println!("x1: {x1:?}");
        jacobian_res(residual, x1.to_f64().view(), J, slope_buffer);
        // println!("J: {:?}", J);
        DG_inv = J.inv()?;
        // println!("j^-1: {:?}", DG_inv);
        // g(x1) != 0
        residual.eval(x1.view(), &mut G);
        // println!("G: {:?}", G);
        // println!();
        err = G.to_f64().norm();
        num_iter += 1;
    }
    println!();
    Ok(x1)
}
