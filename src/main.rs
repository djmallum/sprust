mod config_read;
mod files;

use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::fmt::Formatter;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::str::FromStr;
use clap::{Parser, Subcommand};
use ndarray::{Array2, Axis};
use files::{find_files};
use crate::config_read::new_ini;

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

enum Dimension {
    X,
    #[allow(dead_code)]
    Y,
    #[allow(dead_code)]
    Z,
}

impl FromStr for Dimension {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "x" | "X" => Ok(Dimension::X),
            "y" | "Y" => todo!("Only operations along X are implemented"),
            "z" | "Z" => todo!("Only operations along X are implemented"),
            _ => Err(format!("Unknown dimension: {}", s)),
        }
    }
}

impl fmt::Display for Dimension {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let name = match self {
            Dimension::X => "X",
            Dimension::Y => "Y",
            Dimension::Z => "Z",
        };
        write!(f, "{}", name)
    }
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

fn compute_and_save_mean(
    files: Vec<PathBuf>,
    config: &HashMap<String, Option<String>>,
    dimension: Dimension,
) -> Result<(), Box<dyn Error>> {
    let nx: usize = get(config, "Nx").parse()?;
    let nz: usize = get(config, "Nz").parse()?;
    let file_prefix = format!("t_{dimension}_mean");
    files.iter().enumerate()
        .map(|(i, f)| {
            let t0 = std::fs::read(&f)?;
            let floats: &[f64] = bytemuck::cast_slice(&t0);
            let axis = match dimension {
                Dimension::X => {Axis(0)}
                Dimension::Y => {Axis(1)}
                Dimension::Z => {Axis(2)}
            };
            let array = Array2::from_shape_vec((nx, nz), floats.to_vec())?
                .mean_axis(axis).unwrap();
            let file_name = format!("{file_prefix}.{i}");
            let file = File::create(file_name)?;
            let mut w = BufWriter::new(file);
            // TODO Save mean into single binary file
            // KE and other transformations should be a single file per time step
            // But dimension reduction should put them all into a single file
            for &v in array.iter() {
                w.write_all(&v.to_le_bytes())?;        // little-endian f64
            }
            w.flush()?;
            Ok(())
        })
        .collect::<Result<(), Box<dyn std::error::Error>>>()?;

    build_config(config,dimension,&file_prefix)?;
    Ok(())
}

fn build_config(
    config: &HashMap<String, Option<String>>,
    dimension: Dimension,
    file_prefix: &str,
) -> Result<(),Box<dyn Error>> {
    let mut ini = new_ini(config)?;
    let key = match dimension {
        Dimension::X => "Nx",
        Dimension::Y => "Ny",
        Dimension::Z => "Nz",
    };
    ini.set("default", key, Some("1".into()));

    ini.write(format!("{file_prefix}.conf"))?;
    Ok(())
}

fn get<'a>(config: &'a HashMap<String, Option<String>>, var_name: &str) -> &'a str {
    let nx = config.get(var_name).and_then(|o| o.as_ref());
    let nx = match nx {
        Some(val) => val,
        None => panic!(),
    };
    nx
}


