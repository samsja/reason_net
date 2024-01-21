use clap::Parser;
use indicatif::ProgressBar;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rustc_hash::FxHashSet;

/// Simple program to greet a person
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long)]
    min: u32,

    #[arg(long)]
    max: u32,

    #[arg(long)]
    size: u32,

    #[arg(long)]
    seed: u64,

    #[arg(long)]
    save_file_path: String,

    #[arg(long, default_value = "+-*/%")]
    operators: String,
}

fn generate_operator(a: i64, b: i64, operator: char) -> String {
    match operator {
        '+' => format!("{a}+{b}={}", a + b),
        '-' => format!("{a}-{b}={}", a - b),
        '*' => format!("{a}*{b}={}", a * b),
        '/' => format!("{a}/{b}={}", a / b),
        '%' => format!("{a}%{b}={}", a % b),
        _ => panic!("Invalid operand"),
    }
}

fn generate_datapoint(min: u32, max: u32, rng: &mut StdRng, operators: String) -> String {
    let min_val = 10i64.pow(min);
    let max_val = 10i64.pow(max);

    let a: i64 = rng.gen_range(min_val..max_val);
    let b: i64 = rng.gen_range(min_val..max_val);

    let operand = rng.gen_range(0..operators.len());

    let operator = operators.chars().nth(operand).unwrap();

    let exo = generate_operator(a, b, operator);

    exo
}

fn generate_all(conf: Args) {
    let mut rng = StdRng::seed_from_u64(conf.seed);

    let mut unique_exos: FxHashSet<String> = FxHashSet::default();

    let progress_bar = ProgressBar::new(conf.size as u64);

    while unique_exos.len() < conf.size as usize {
        let datapoint: String =
            generate_datapoint(conf.min, conf.max, &mut rng, conf.operators.clone());
        let added = unique_exos.insert(datapoint);
        if added {
            progress_bar.inc(1);
        }
    }

    progress_bar.finish_with_message("done");

    std::fs::write(
        conf.save_file_path,
        unique_exos.into_iter().collect::<Vec<String>>().join("\n"),
    )
    .unwrap();
}

fn main() {
    let args = Args::parse();
    generate_all(args);
}
