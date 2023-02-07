use ndarray::{Array1, Array2, ArrayView1};

use crate::ad::*;

pub trait Explicit {
    fn next(&self, x: ArrayView1<f64>) -> Array1<f64>;
}

pub trait Implicit: Residual {}

/// Defines what an ODE solver needs
pub trait ODE<Scheme>
{
    /// Set the step size `h` which should be used throughout the whole computation.
    fn set_step_size(&mut self, h: f64) -> &mut Self;
    /// Set the total runtime of the simulation.
    #[allow(non_snake_case)]
    fn set_t(&mut self, T: f64) -> &mut Self;
    /// To deactivate tqdm during runtime
    fn set_with_tqdm(&mut self, with_tqdm:bool) ->&mut Self;
    /// Consumes the defined ODE and runs the simulation.
    ///
    /// It returns a vector of all time steps and an array which corresponds to the state at each timestep.
    ///
    /// columns: state
    ///
    /// rows: timestep
    fn run(self) -> (Vec<f64>, Array2<f64>);
}


pub trait OneStep {
    type Scheme;
    fn new(scheme: Self::Scheme, initial: Array1<f64>) -> Self;
}
pub trait TwoStep {
    type Scheme;
    fn new(residual: Self::Scheme, x0: Array1<f64>, x1: Array1<f64>) -> Self;
}

/// Defines a function that needs to evaluate to zero.
/// It was introduced for implicit methods.
pub trait Residual {
    /// A function that shall evaluate with the corrext `x` to zero.
    fn eval(&self, x_next: Array1<AD>) -> Array1<AD>;
}
/// Updates current x0, so that the residual for the next step can be calculated.
pub trait Residual1Step {
    fn update(&mut self, x0: Array1<AD>);
}

/// Updates current x0 and x1, so that the residual for the next step can be calculated.
pub trait Residual2Step {
    fn new(x0: Array1<AD>, x1: Array1<AD>, h: f64) -> Self;
    fn update(&mut self, x0: Array1<AD>, x1: Array1<AD>);
}

