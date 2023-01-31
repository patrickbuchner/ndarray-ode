use ndarray::{Array1, Array2};

use crate::ad::*;

pub trait ODE< Res>
where
    Res: Residual + std::marker::Sync,
{
    fn set_step_size(&mut self, h: f64) -> &mut Self;
    #[allow(non_snake_case)]
    fn set_t(&mut self, T: f64) -> &mut Self;
    fn set_epsilon(&mut self, ɛ: f64) -> &mut Self;
    fn run(self) -> (Vec<f64>, Array2<f64>);
}

pub trait Residual
{
    fn eval(&self, x_next: Array1<AD>) -> Array1<AD>;
}
pub trait Residual1Step
{
    fn update(&mut self, x0: Array1<AD>);
}

pub trait Residual2Step
{
    fn new(x0: Array1<AD>, x1: Array1<AD>, h: f64) -> Self;
    fn update(&mut self, x0: Array1<AD>, x1: Array1<AD>);
}

