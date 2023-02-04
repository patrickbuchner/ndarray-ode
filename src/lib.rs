//! A library to make it more streamlined to solve odes
//!  
//! The library has currently support for ndarrays.
//! It supports one and two step methods but can be easily adapted to more.
//! 
//! ## Design
//! The ode solver expects a residual which will be reduced to zero with a given ɛ.
//! The residual is a trait implementing an evaluation function which takes in the autodiff from peroxides dual number AD.
//! This library contains a copy of peroxides implementation adapted to work with ndarrays.
//! Plotting of the results is done via python. Therefore the results need to be stored in a common datafile,
//! which currently is a parquet file via a small wrapper around polars dataframe.
//! 
//! ## Keppler problem example:
//! ```
//! use ndarray::*;
//! use ndarray_ode::{
//!     ad::*,
//!     ode::{solver::*, *},
//! };
//! const DOF: usize = 4;
//! #[allow(non_upper_case_globals)]
//! const μ: f64 = 1.;
//! 
//! fn keppler(x: ArrayView1<AD>) -> Array1<AD> {
//! let (x, y, px, py) = (x[0], x[1], x[2], x[3]);
//! let factor = -μ * (x * x + y * y).powi(3).sqrt();
//! array![px, py, factor * x, factor * y]
//! }
//! 
//! fn main() {
//!     let h = 0.5;
//!     #[allow(non_snake_case)]
//!     let T = 2.0;
//!     let x0 = array![1.0, 0.0, 0.0, 1.0];
//!     assert![x0.len() == DOF];
//! 
//!     let euler = SymplecticEuler::new(x0.to_ad(), h, keppler);
//!     let mut ode = Ode::new(euler, x0.to_ad());
//!     ode.set_step_size(h).set_t(T);
//!     
//!     let (time, result) = ode.run();
//! }
//! ```
#![allow(uncommon_codepoints)]
pub mod ad;
pub mod ode;
pub mod plot;
