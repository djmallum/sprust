use std::fs;
use std::path::PathBuf;

pub(crate) fn find_files(dir: &str, file_prefix: &str) -> std::io::Result<Vec<PathBuf>> {
    let mut paths: Vec<PathBuf> = fs::read_dir(dir)?
        .filter_map(|entry| entry.ok())            // skip unreadable entries
        .map(|entry| entry.path())
        .filter(|p| p.is_file())
        .filter(|p| is_file_prefix(p, file_prefix))
        .collect();

    paths.sort();                                  // deterministic order
    Ok(paths)
}

fn is_file_prefix(p: &std::path::Path,file_prefix: &str) -> bool {
    let name = match p.file_name().and_then(|n| n.to_str()) {
        Some(n) => n,
        None => return false,
    };
    // match "t" followed by a dot and digits
    if let Some(rest) = name.strip_prefix(file_prefix) {
        !rest.is_empty() && rest.chars().all(|c| c.is_ascii_digit())
    } else {
        false
    }
}
