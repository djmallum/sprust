mod config_read;
mod files;
mod mean;

use clap::{Parser, Subcommand};
use files::find_files;
use std::error::Error;
use crate::mean::{
    compute_and_save_mean,
    Dimension,
};

#[derive(Parser)]
#[command(name = "spins_rust")]
#[command(version = "0.1")]
#[command(about = "For running bulk calculations on spins outputs", long_about = None)]
struct Cli {
    #[arg(short)]
    path: Option<String>,
    #[command(subcommand)]
    command: Calculations,
}

#[derive(Subcommand)]
enum Calculations {
    /// Compute the mean along the X dimension
    Mean{
        /// Variable to operate on
        variable: String,
        /// Dimension to operate on
        #[arg(short)]
        dimension: Option<String>,
    },
}


fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();
    let path = match &cli.path {
        Some(p) => p,
        None => "."
    };

    let config = config_read::new()?;
    match cli.command {
        Calculations::Mean { variable, dimension } => {
            let var_name = format!("{}.", variable);
            let files = find_files(&path,&var_name)?;
            println!("Found {:?} files", files.len());
            let dimension = match dimension {
                Some(d) => d.parse::<Dimension>()?,
                None => Dimension::X,
            };

            let config = &config["default"];
            compute_and_save_mean(files, config, dimension)?;
        }
    }
    Ok(())
}



