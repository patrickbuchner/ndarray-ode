/// Given a path to a png file, this command will open it automatically.
pub fn open_png(path_to_plot: &str) {
    use std::process::Command;

    #[cfg(target_family = "unix")]
    Command::new("/usr/bin/xdg-open")
        .arg(path_to_plot)
        .output()
        .unwrap();
}
/// pass the path a python file and run it with some arguments or an empty list of arguments.
pub fn python(file: std::path::PathBuf, arguments: &[&str]) {
    use std::io::Write;
    use std::process::Command;
    let output = Command::new("python").arg(file).args(arguments).output();
    match output {
        Ok(o) => {
            if !o.status.success() {
                std::io::stdout().write_all(&o.stdout).unwrap();
                std::io::stderr().write_all(&o.stderr).unwrap();
            }
        }
        Err(e) => {
            println!("{e:#}");
        }
    }
}

mod dataframe;
pub use dataframe::Dataframe;
mod series;
pub use series::Series;
