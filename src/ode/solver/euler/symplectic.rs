use ndarray::{array, s, Array1, ArrayView1, Axis, Zip};

use crate::{ad::*, ode::*};
pub struct SymplecticEuler<Flow>
where
    Flow: Fn(ArrayView1<AD>, &mut Array1<AD>),
{
    x0_owned: Array1<AD>,
    flow: Flow,
    h: AD,
}
impl<Flow> SymplecticEuler<Flow>
where
    Flow: Fn(ArrayView1<AD>, &mut Array1<AD>),
{
    pub fn new(h: f64, flow: Flow) -> Self {
        SymplecticEuler {
            x0_owned: array![],
            h: AD::AD0(h),
            flow,
        }
    }
}
impl<Flow> Residual for SymplecticEuler<Flow>
where
    Flow: Fn(ArrayView1<AD>, &mut Array1<AD>),
{
    #[inline]
    fn eval(&self, x1: ArrayView1<AD>, update: &mut Array1<AD>) {
        let h = self.h;
        let x0 = self.x0_owned.view();
        let mut q0 = x0.slice(s![..x0.len() / 2_usize]).to_owned();
        // let _p0 = x0.slice(s![x0.len() / 2 as usize..]);
        // let _q1 = x1.slice(s![..x0.len() / 2 as usize]);
        let p1 = x1.slice(s![x0.len() / 2_usize..]);
        _ = q0.append(Axis(0), p1);
        (self.flow)(q0.view(), update);
        Zip::from(x1)
            .and(x0)
            .and(update)
            .for_each(|&x1, &x0, f| *f = x1 - x0 - h * *f);
        // izip!(x1.iter(), x0.iter(), f.iter_mut()).for_each(|(&x1, &x0, f)| *f = x1 - x0 - h * *f);
        // x1 - x0 - h.scalar((self.flow)(q0.view()))
    }
}

impl<Flow> Residual1Step for SymplecticEuler<Flow>
where
    Flow: Fn(ArrayView1<AD>, &mut Array1<AD>),
{
    #[inline]
    fn update(&mut self, x0: Array1<AD>) {
        self.x0_owned = x0;
    }
}

impl<Flow> Implicit for SymplecticEuler<Flow> where Flow: Fn(ArrayView1<AD>, &mut Array1<AD>) {}
