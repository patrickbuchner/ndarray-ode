use ndarray::{Array1, Array2, ArrayView1, Axis};
use rayon::prelude::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

use crate::ad::*;
/// Jacobian Matrix
///
/// # Description
/// Exact jacobian matrix using Automatic Differenitation
///
/// # Type
/// (Vector, F) -> Matrix where F: Fn(&Vec<AD>) -> Vec<AD>
///
/// # Examples
/// ```
/// use ndarray::{array, Array1, ArrayView1};
/// use ndarray_ode::prelude::*;
/// fn main() {
///     let x = array![1., 1.];
///     let j = jacobian(f, x.view());
///     
///     let expected = array!([1.0, -1.0], [1.0, 2.0]);
///     println!("{j:?}");
///
///     //      c[0] c[1]
///     // r[0]    1   -1
///     // r[1]    1    2
/// }
/// fn f(xs: ArrayView1<AD>) -> Array1<AD> {
///     let x = xs[0];
///     let y = xs[1];
///
///     array![x - y, x + 2. * y]
/// }
/// ```
#[allow(non_snake_case)]
pub fn jacobian<F: Fn(ArrayView1<AD>) -> Array1<AD>>(f: F, x: ArrayView1<f64>) -> Array2<f64> {
    let l = x.len();
    let mut x_ad: Array1<AD> = x.iter().map(|&x| AD1(x, 0f64)).collect();
    let mut J = Array2::zeros((l, l));

    for (i, mut col) in J.axis_iter_mut(Axis(1)).enumerate() {
        x_ad[i][1] = 1f64;
        let slopes: Vec<f64> = f(x_ad.view()).iter().map(|ad| ad.dx()).collect();
        for (c, s) in col.iter_mut().zip(slopes.iter()) {
            *c = *s;
        }
        x_ad[i][1] = 0f64;
    }
    J
}

/// Jacobian Matrix
///
/// # Description
/// Exact jacobian matrix using Automatic Differenitation
/// Use it only after measuring that it is better than the normal jacobian function.
///
/// # Type
/// (Vector, F) -> Matrix where F: Fn(&Vec<AD>) -> Vec<AD>
///
/// # Examples
/// ```
/// use ndarray::{array, Array1, ArrayView1};
/// use ndarray_ode::prelude::*;
/// fn main() {
///     let x = array![1., 1.];
///     let j = jacobian_par(f, x.view());
///     
///     let expected = array!([1.0, -1.0], [1.0, 2.0]);
///     println!("{j:?}");
///
///     //      c[0] c[1]
///     // r[0]    1   -1
///     // r[1]    1    2
/// }
/// fn f(xs: ArrayView1<AD>) -> Array1<AD> {
///     let x = xs[0];
///     let y = xs[1];
///
///     array![x - y, x + 2. * y]
/// }
/// ```
#[allow(non_snake_case)]
pub fn jacobian_par<F: Fn(ArrayView1<AD>) -> Array1<AD> + std::marker::Sync>(
    f: F,
    x: ArrayView1<f64>,
) -> Array2<f64> {
    let l = x.len();
    let mut J = Array2::zeros((l, l));
    J.axis_iter_mut(Axis(1))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut col)| {
            let mut x_ad: Array1<AD> = x.iter().map(|&x| AD1(x, 0f64)).collect();
            x_ad[i][1] = 1f64;
            let slopes: Vec<f64> = f(x_ad.view()).iter().map(|ad| ad.dx()).collect();
            for (c, s) in col.iter_mut().zip(slopes.iter()) {
                *c = *s;
            }
            x_ad[i][1] = 0f64;
        });
    J
}
/// Jacobian Matrix for [crate::ode::traits::Residual]
///
/// # Description
/// Exact jacobian matrix using Automatic Differenitation
///
/// # Type
/// (Vector, F) -> Matrix where F: Fn(&Vec<AD>) -> Vec<AD>
///
/// # Examples
/// ```
/// use ndarray::{array, Array1, ArrayView1};
/// use ndarray_ode::prelude::*;
/// fn main() {
///     let x = array![1.0, 1.0];
///     let h = 0.1;
///     let mut res = SymplecticEuler::new(h, f);
///     res.update(x.to_ad());
///     let j = jacobian_res(&res, x.view());
///     
///     let expected = array!([1.0, 1.0], [0.0, -1.0]);;
///     println!("{j:?}");
///
///     //      c[0] c[1]
///     // r[0]    1    1
///     // r[1]    0   -1
/// }
/// fn f(xs: ArrayView1<AD>) -> Array1<AD> {
///     let x = xs[0];
///     let y = xs[1];
///
///     array![x - y, x + 2. * y]
/// }
/// ```
#[allow(non_snake_case)]
pub fn jacobian_res<Res>(f: &Res, x: ArrayView1<f64>, J: &mut Array2<f64>, slopes: &mut Array1<AD>)
where
    Res: crate::ode::Residual + std::marker::Sync,
{
    // let l = x.len();
    // let mut J = Array2::zeros((l, l));
    let mut x_ad: Array1<AD> = x.iter().map(|&x| AD1(x, 0f64)).collect();

    J.axis_iter_mut(Axis(1))
        // .into_par_iter()
        .enumerate()
        .for_each(|(i, mut col)| {
            x_ad[i][1] = 1f64;
            f.eval(x_ad.view(), slopes);
            for (c, s) in col.iter_mut().zip(slopes.iter()) {
                *c = s.dx();
            }
        });
}

#[cfg(test)]
mod test {
    use ndarray::{array, Array1, Array2, ArrayView1};

    use crate::ad::*;

    use super::{jacobian, jacobian_par};

    #[test]
    fn jacobian_test() {
        let x = array![1., 1.];
        let j = jacobian(f, x.view());

        let expected = array!([1.0, -1.0], [1.0, 2.0]);
        assert_eq!(expected, j);
        //      c[0] c[1]
        // r[0]    1   -1
        // r[1]    1    2
    }

    #[test]
    fn jacobian_par_test() {
        let x = array![1., 1.];
        let j = jacobian_par(f, x.view());

        let expected = array!([1.0, -1.0], [1.0, 2.0]);
        assert_eq!(expected, j);
        //      c[0] c[1]
        // r[0]    1   -1
        // r[1]    1    2
    }
    fn f(xs: ArrayView1<AD>) -> Array1<AD> {
        let x = xs[0];
        let y = xs[1];

        array![x - y, x + 2. * y]
    }

    fn f2(xs: ArrayView1<AD>, update: &mut Array1<AD>) {
        let x = xs[0];
        let y = xs[1];

        update[0] = x - y;
        update[1] = x + 2. * y;
    }
    #[test]
    fn jacobian_res_test() {
        use crate::prelude::*;
        let x0 = array![1.0, 1.0];
        let h = 1.;
        let mut euler = SymplecticEuler::new(h, f2);
        euler.update(x0.to_ad());

        let l = x0.len();
        #[allow(non_snake_case)]
        let mut J = Array2::zeros((l, l));
        let mut slope_buffer = x0.to_ad();
        jacobian_res(&euler, x0.view(), &mut J, &mut slope_buffer);
        let expected = array!([1.0, 2.0], [0.0, -1.0]);
        assert_eq!(expected, J);
        //      c[0] c[1]
        // r[0]    1    1
        // r[1]    0   -1
    }
}
