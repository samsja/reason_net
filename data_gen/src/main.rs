use clap::Parser;
use indicatif::ProgressBar;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
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

    let operators: Vec<char> = conf.operators.chars().collect();

    let max = 999;

    let range1 = 9..max;
    let range2 = 9..max;

    let mut digit_data: Vec<String> = Vec::with_capacity(operators.len() * max * max);

    for operator in operators {
        for i in range1.clone() {
            for j in range2.clone() {
                digit_data.push(generate_operator(i as i64, j as i64, operator))
            }
        }
    }

    let unique_exos: Vec<String> = unique_exos.into_iter().collect();

    let mut exo_to_save = unique_exos
        .into_iter()
        .chain(digit_data.into_iter())
        .collect::<Vec<String>>();

    exo_to_save.shuffle(&mut rng);

    std::fs::write(conf.save_file_path, exo_to_save.join("\n")).unwrap();
}

fn main() {
    let args = Args::parse();

    if args.min > args.max {
        panic!("min should be less than max");
    }

    if args.min < 3 {
        panic!("min should be greater than 3");
    }

    generate_all(args);
}
