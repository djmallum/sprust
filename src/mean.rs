use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use ndarray::{Array2, Axis};
use crate::config_write::{build_config, get};
use crate::Dimension;

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
