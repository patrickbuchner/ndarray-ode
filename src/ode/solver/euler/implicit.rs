use ndarray::{array, Array1, ArrayView1, Zip};

use crate::{ad::*, ode::*};

pub struct ImplicitEuler<Flow>
where
    Flow: Fn(ArrayView1<AD>, &mut Array1<AD>),
{
    x0_owned: Array1<AD>,
    flow: Flow,
    h: AD,
}
impl<Flow> ImplicitEuler<Flow>
where
    Flow: Fn(ArrayView1<AD>, &mut Array1<AD>),
{
    pub fn new(h: f64, flow: Flow) -> Self {
        ImplicitEuler {
            x0_owned: array![],
            h: AD::AD0(h),
            flow,
        }
    }
}
impl<Flow> Residual for ImplicitEuler<Flow>
where
    Flow: Fn(ArrayView1<AD>, &mut Array1<AD>),
{
    #[inline]
    fn eval(&self, x1: ArrayView1<AD>, update: &mut Array1<AD>) {
        let h = self.h;
        let x0 = self.x0_owned.view();
        (self.flow)(x1, update);
        // x1 - x0 - h.scalar(flow)
        Zip::from(x1)
            .and(x0)
            .and(update)
            .for_each(|&x1, &x0, f| *f = x1 - x0 - h * *f);
    }
}

impl<Flow> Residual1Step for ImplicitEuler<Flow>
where
    Flow: Fn(ArrayView1<AD>, &mut Array1<AD>),
{
    #[inline]
    fn update(&mut self, x0: Array1<AD>) {
        self.x0_owned = x0;
    }
}
impl<Flow> Implicit for ImplicitEuler<Flow> where Flow: Fn(ArrayView1<AD>, &mut Array1<AD>) {}
