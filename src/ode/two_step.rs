use crate::{ad::*, ode::*};
use ndarray::*;
use tqdm::tqdm;
/// This ode solver uses a two step scheme, through which one may get better solutions
/// but also needs to define different residual functions.
#[allow(non_snake_case)]
pub struct OdeTwoStep<Scheme>
where
    Scheme: Implicit + std::marker::Sync + Residual2Step,
{
    scheme: Scheme,
    x0: Array1<AD>,
    x1: Array1<AD>,
    h: f64,
    T: f64,
    _ɛ: f64,
}
impl<Scheme> OdeTwoStep<Scheme>
where
    Scheme: Implicit + std::marker::Sync + Residual2Step,
{
    pub fn new(scheme: Scheme, x0: Array1<AD>, x1: Array1<AD>) -> Self {
        OdeTwoStep {
            scheme,
            x0,
            x1,
            h: 0.1,
            T: 1.0,
            _ɛ: f64::EPSILON,
        }
    }
}

impl<Scheme> ODE<Scheme> for OdeTwoStep<Scheme>
where
    Scheme: Implicit + std::marker::Sync + Residual2Step,
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

    fn run(mut self) -> (Vec<f64>, Array2<f64>) {
        let n: f64 = self.T / self.h;
        let n = n.floor() as usize;
        let mut x0 = self.x0.clone().to_f64();
        let mut x1 = self.x1.clone().to_f64();

        let mut result: Array2<f64> = Array::zeros((0, x0.len()));
        result.push_row(self.x0.to_f64().view()).unwrap();
        result.push_row(self.x1.to_f64().view()).unwrap();
        let mut time = vec![0.0; n];
        time[1] = 1. * self.h;

        let l = x0.len();
        #[allow(non_snake_case)]
        let mut J = Array2::zeros((l, l));
        let mut slope_buffer = x0.to_ad();

        for t in tqdm(2..n) {
            self.scheme.update(x0.to_ad(), x1.to_ad());
            match newton(
                f64::EPSILON,
                &self.scheme,
                x1.to_ad(),
                &mut J,
                &mut slope_buffer,
            ) {
                Ok(x2) => {
                    let x2 = x2.to_f64();
                    result.push_row(x2.view()).unwrap();
                    x0 = x1;
                    x1 = x2;
                    time[t] = t as f64 * self.h;
                }
                Err(e) => eprintln!("Did not converge, how to handle it?\n{e}"),
            }
        }
        (time, result)
    }

    fn set_with_progress(&mut self, _with_tqdm: bool) -> &mut Self {
        todo!("Implement setting the progress, currently always show it.")
    }
}
