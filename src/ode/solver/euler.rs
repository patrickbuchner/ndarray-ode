use ndarray::{array, s, Array1, ArrayView1, Axis};

use crate::{ad::*, ode::*};

pub struct ImplicitEuler<Flow>
where
    Flow: Fn(ArrayView1<AD>) -> Array1<AD>,
{
    x0_owned: Array1<AD>,
    flow: Flow,
    h: f64,
}
impl<Flow> ImplicitEuler<Flow>
where
    Flow: Fn(ArrayView1<AD>) -> Array1<AD>,
{
    pub fn new(h: f64, flow: Flow) -> Self {
        ImplicitEuler {
            x0_owned: array![],
            h,
            flow,
        }
    }
}
impl<Flow> Residual for ImplicitEuler<Flow>
where
    Flow: Fn(ArrayView1<AD>) -> Array1<AD>,
{
    fn eval(&self, x1: Array1<AD>) -> Array1<AD> {
        let h = AD::AD0(self.h);
        let x0 = self.x0_owned.view();
        let flow = (self.flow)(x1.view());
        x1 - x0 - h.scalar(flow)
    }
}

impl<Flow> Residual1Step for ImplicitEuler<Flow>
where
    Flow: Fn(ArrayView1<AD>) -> Array1<AD>,
{
    fn update(&mut self, x0: Array1<AD>) {
        self.x0_owned = x0;
    }
}
impl<Flow> Implicit for ImplicitEuler<Flow> where Flow: Fn(ArrayView1<AD>) -> Array1<AD> {}

pub struct ExplicitEuler<Flow>
where
    Flow: Fn(ArrayView1<f64>) -> Array1<f64>,
{
    flow: Flow,
    h: f64,
}

impl<Flow> ExplicitEuler<Flow>
where
    Flow: Fn(ArrayView1<f64>) -> Array1<f64>,
{
    pub fn new(h: f64, flow: Flow) -> Self {
        ExplicitEuler { flow, h }
    }
}
impl<Flow> Explicit for ExplicitEuler<Flow>
where
    Flow: Fn(ArrayView1<f64>) -> Array1<f64>,
{
    fn next(&self, x: ArrayView1<f64>) -> Array1<f64> {
        let h = self.h;
        x.to_owned() + h.scalar((self.flow)(x))
    }
}

pub struct SymplecticEuler<Flow>
where
    Flow: Fn(ArrayView1<AD>) -> Array1<AD>,
{
    x0_owned: Array1<AD>,
    flow: Flow,
    h: f64,
}
impl<Flow> SymplecticEuler<Flow>
where
    Flow: Fn(ArrayView1<AD>) -> Array1<AD>,
{
    pub fn new(h: f64, flow: Flow) -> Self {
        SymplecticEuler {
            x0_owned: array![],
            h,
            flow,
        }
    }
}
impl<Flow> Residual for SymplecticEuler<Flow>
where
    Flow: Fn(ArrayView1<AD>) -> Array1<AD>,
{
    #[inline]
    fn eval(&self, x1: Array1<AD>) -> Array1<AD> {
        let h = AD::AD0(self.h);
        let x0 = self.x0_owned.view();
        let mut q0 = x0.slice(s![..x0.len() / 2_usize]).to_owned();
        // let _p0 = x0.slice(s![x0.len() / 2 as usize..]);
        // let _q1 = x1.slice(s![..x0.len() / 2 as usize]);
        let p1 = x1.slice(s![x0.len() / 2_usize..]);
        _ = q0.append(Axis(0), p1.view());
        x1 - x0 - h.scalar((self.flow)(q0.view()))
    }
}

impl<Flow> Residual1Step for SymplecticEuler<Flow>
where
    Flow: Fn(ArrayView1<AD>) -> Array1<AD>,
{
    fn update(&mut self, x0: Array1<AD>) {
        self.x0_owned = x0;
    }
}

impl<Flow> Implicit for SymplecticEuler<Flow> where Flow: Fn(ArrayView1<AD>) -> Array1<AD> {}
