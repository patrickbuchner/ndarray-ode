use crate::{ad::*, ode::*};
use ndarray::*;
use tqdm::tqdm;
#[allow(non_snake_case)]
pub struct Ode<Res>
where
    Res: Residual + std::marker::Sync + Residual1Step,
{
    residual: Res,
    initial: Array1<AD>,
    h: f64,
    T: f64,
    ɛ: f64,
}
impl<Res> Ode<Res>
where
    Res: Residual + std::marker::Sync + Residual1Step,
{
    pub fn new(residual: Res, initial: Array1<AD>) -> Self {
        Ode {
            residual,
            initial,
            h: 0.1,
            T: 1.0,
            ɛ: f64::EPSILON,
        }
    }
}

impl<Res> ODE<Res> for Ode<Res>
where
    Res: Residual + std::marker::Sync + Residual1Step,
{
    fn set_step_size(&mut self, h: f64) -> &mut Self {
        self.h = h;
        self
    }

    #[allow(non_snake_case)]
    fn set_t(&mut self, T: f64) -> &mut Self {
        self.T = T;
        self
    }

    fn set_epsilon(&mut self, ɛ: f64) -> &mut Self {
        self.ɛ = ɛ;
        self
    }

    fn run(mut self) -> (Vec<f64>, Array2<f64>) {
        let n: f64 = self.T / self.h;
        let n = n.floor() as usize;
        let mut x0 = self.initial.clone().to_f64();

        let mut result: Array2<f64> = Array::zeros((0, x0.len()));
        result.push_row(self.initial.to_f64().view()).unwrap();
        let mut time = vec![0.0; n];

        for t in tqdm(1..n) {
            self.residual.update(x0.to_ad());
            match newton(f64::EPSILON, &self.residual, x0.clone()) {
                Ok(x1) => {
                    result.push_row(x1.view()).unwrap();
                    x0 = x1;
                    time[t] = t as f64 * self.h;
                }
                Err(_) => todo!(),
            }
        }
        (time, result)
    }
}
