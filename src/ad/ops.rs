use std::ops::Div;

use ndarray::{Array1, ArrayView1};

use crate::ad::*;

pub fn factorial(n: usize) -> usize {
    let mut p = 1usize;
    for i in 1..(n + 1) {
        p *= i;
    }
    p
}
pub fn double_factorial(n: usize) -> usize {
    let mut s = 1usize;
    let mut n = n;
    while n >= 2 {
        s *= n;
        n -= 2;
    }
    s
}
/// Permutation
#[allow(non_snake_case)]
pub fn P(n: usize, r: usize) -> usize {
    let mut p = 1usize;
    for i in 0..r {
        p *= n - i;
    }
    p
}

/// Combination
#[allow(non_snake_case)]
pub fn C(n: usize, r: usize) -> usize {
    if r > n / 2 {
        return C(n, n - r);
    }

    P(n, r) / factorial(r)
}
/// Combination with Repetition
#[allow(non_snake_case)]
pub fn H(n: usize, r: usize) -> usize {
    C(n + r - 1, r)
}

pub trait ExpLogOps: Sized {
    fn exp(&self) -> Self;
    fn ln(&self) -> Self;
    fn log(&self, base: f64) -> Self;
    fn log2(&self) -> Self {
        self.log(2f64)
    }
    fn log10(&self) -> Self {
        self.log(10f64)
    }
}

pub trait PowOps: Sized {
    fn powi(&self, n: i32) -> Self;
    fn powf(&self, f: f64) -> Self;
    fn pow(&self, f: Self) -> Self;
    fn sqrt(&self) -> Self {
        self.powf(0.5)
    }
}

pub trait TrigOps: Sized + Div<Output = Self> {
    fn sin_cos(&self) -> (Self, Self);
    fn sin(&self) -> Self {
        let (s, _) = self.sin_cos();
        s
    }
    fn cos(&self) -> Self {
        let (_, c) = self.sin_cos();
        c
    }
    fn tan(&self) -> Self {
        let (s, c) = self.sin_cos();
        s / c
    }
    fn sinh_cosh(&self) -> (Self, Self);
    fn sinh(&self) -> Self {
        let (s, _) = self.sinh_cosh();
        s
    }
    fn cosh(&self) -> Self {
        let (_, c) = self.sinh_cosh();
        c
    }
    fn tanh(&self) -> Self {
        let (s, c) = self.sinh_cosh();
        s / c
    }
    fn asin(&self) -> Self;
    fn acos(&self) -> Self;
    fn atan(&self) -> Self;
    fn asin_acos(&self) -> (Self, Self) {
        (self.asin(), self.acos())
    }
    fn asinh(&self) -> Self;
    fn acosh(&self) -> Self;
    fn atanh(&self) -> Self;
    fn asinh_acosh(&self) -> (Self, Self) {
        (self.asinh(), self.acosh())
    }
}

/// Stable implementations for nightly-only features
///
/// # Implemented List
///
/// * `StableFn` : Make `FnOnce` to stable
/// Stable Fn trait
///
/// # Description
/// Implement `FnOnce` is still nighlty only feature. This trait is alternative to `FnOnce` trait.
pub trait StableFn<T> {
    type Output;
    fn call_stable(&self, target: T) -> Self::Output;
}

pub trait ADArray {
    type Array;
    fn to_ad(&self) -> Self::Array;
}
impl ADArray for Array1<f64> {
    type Array = Array1<AD>;
    fn to_ad(&self) -> Self::Array {
        self.iter().map(|t| AD::from(*t)).collect()
    }
}
impl ADArray for ArrayView1<'_, f64> {
    type Array = Array1<AD>;
    fn to_ad(&self) -> Self::Array {
        self.iter().map(|t| AD::from(*t)).collect()
    }
}
pub trait F64Array {
    type Array;
    fn to_f64(&self) -> Self::Array;
}

impl F64Array for Array1<AD> {
    type Array = Array1<f64>;
    fn to_f64(&self) -> Self::Array {
        self.iter().map(|t| t.x()).collect()
    }
}
impl F64Array for ArrayView1<'_, AD> {
    type Array = Array1<f64>;
    fn to_f64(&self) -> Self::Array {
        self.iter().map(|t| t.x()).collect()
    }
}

pub trait ScalarMultiplyf64 {
    type Array;
    fn scalar(&self, ar: Self::Array) -> Self::Array;
}

impl ScalarMultiplyf64 for f64 {
    type Array = Array1<f64>;
    fn scalar(&self, mut ar: Self::Array) -> Self::Array {
        ar.iter_mut().for_each(|x| *x *= self);
        ar
    }
}

// impl ScalarMultiplyf64 for AD {
//     type Array = Array1<f64>;
//     fn scalar(&self, mut ar: Self::Array) -> Self::Array {
//         ar.par_iter_mut().for_each(|x| *x = *x * self.x());
//         ar
//     }
// }

pub trait ScalarMultiplyAD {
    type Array;
    fn scalar(&self, ar: Self::Array) -> Self::Array;
}
// impl ScalarMultiplyAD for f64 {
//     type Array = Array1<AD>;
//     fn scalar(&self, mut ar: Self::Array) -> Self::Array {
//         ar.par_iter_mut().for_each(|x| *x = *x * AD::AD0(*self));
//         ar
//     }
// }

impl ScalarMultiplyAD for AD {
    type Array = Array1<AD>;
    fn scalar(&self, mut ar: Self::Array) -> Self::Array {
        ar.iter_mut().for_each(|x| *x = *x * self.x());
        ar
    }
}
