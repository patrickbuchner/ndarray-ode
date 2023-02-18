use ndarray::{Array1, ArrayView1};

use crate::ode::*;
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
    #[inline]
    fn next(&self, x: ArrayView1<f64>) -> Array1<f64> {
        let h = self.h;
        let mut flow = (self.flow)(x);
        // Zip::from(x)
        //     .and(&mut flow)
        //     .for_each(|&x0, f| *f = x0 + h * *f);
        flow.iter_mut()
            .zip(x.iter())
            .for_each(|(f, &x)| *f = x + h * *f);
        flow
    }
}
