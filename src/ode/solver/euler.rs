mod explicit;
mod implicit;
mod symplectic;
pub use explicit::ExplicitEuler;
pub use implicit::ImplicitEuler;
pub use symplectic::SymplecticEuler;
//"-C", "-Ctarget-cpu=native", "-Clink-arg=-Wl,--no-rosegment"
