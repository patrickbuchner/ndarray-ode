use ndarray::Array1;

mod traits;
pub use traits::*;
pub mod root_finder;
pub use root_finder::*;
mod one_step;
pub mod solver;
use one_step::*;

pub mod two_step;

pub struct Ode;
impl Ode {
    pub fn explicit<Scheme>(scheme: Scheme, initial: Array1<f64>) -> OdeEx<Scheme>
    where
        Scheme: Explicit + std::marker::Sync,
    {
        OdeEx::new(scheme, initial)
    }
    pub fn implicit<Scheme>(scheme: Scheme, initial: Array1<f64>) -> OdeIm<Scheme>
    where
        Scheme: Implicit + std::marker::Sync + Residual1Step,
    {
        OdeIm::new(scheme, initial)
    }
}
