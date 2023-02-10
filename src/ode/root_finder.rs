use crate::{ad::*, ode::Residual};
use ndarray::Array1;
use ndarray_linalg::{error::Result, Inverse, Norm};

/// Newton method for iteratively finding the next state for our problem.
#[allow(non_snake_case)]
#[inline]
pub fn newton<Res>(rtol: f64, residual: &Res, mut x1: Array1<f64>) -> Result<Array1<f64>>
where
    Res: Residual + std::marker::Sync,
{
    // 2. Obtain Jacobian
    let mut Dg = jacobian_res(residual, x1.view());
    let mut DG_inv = Dg.inv()?;
    // x1 - (x0+h*f(x1)) = g(x1) != 0
    let mut G = residual.eval(x1.to_ad().view()).to_f64();
    let mut num_iter: usize = 0;
    let mut err = G.norm();
    // 3. Iteration
    while err >= rtol && num_iter <= 10 {
        // prod = f'^-1 * f(x1) = f(x1)/ f'(x1)
        // Newton manipulation
        let DGG = DG_inv.dot(&G);
        // x1_new = x1 - f(x1)/f'(x1)
        x1 = x1 - DGG;
        Dg = jacobian_res(residual, x1.view());
        DG_inv = Dg.inv()?;
        // g(x1) != 0
        G = residual.eval(x1.to_ad().view()).to_f64();
        err = G.norm();
        num_iter += 1;
    }
    Ok(x1)
}
