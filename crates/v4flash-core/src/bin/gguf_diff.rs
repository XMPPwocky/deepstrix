//! gguf-diff — compare two GGUFs' tensor inventories (dtype/shape/size),
//! aggregated by role (`blk.<n>.` collapsed to `blk.N.`). Built for
//! comparing quant mixes of the same checkpoint (e.g. antirez 0731 vs
//! unsloth UD-IQ2_XXS); split GGUFs accepted on either side.
//!
//! Usage: gguf-diff <a.gguf> <b.gguf> [--all]
//!   --all  also print roles that are identical in both files

use std::collections::BTreeMap;
use std::process::ExitCode;

use v4flash_core::mapped::MappedGguf;

fn role_of(name: &str) -> String {
    if let Some(rest) = name.strip_prefix("blk.") {
        if let Some(dot) = rest.find('.') {
            if rest[..dot].bytes().all(|b| b.is_ascii_digit()) {
                return format!("blk.N.{}", &rest[dot + 1..]);
            }
        }
    }
    name.to_string()
}

#[derive(Default)]
struct RoleAgg {
    /// dtype name -> tensor count
    types: BTreeMap<&'static str, usize>,
    bytes: u64,
    count: usize,
}

fn aggregate(m: &MappedGguf) -> BTreeMap<String, RoleAgg> {
    let mut out: BTreeMap<String, RoleAgg> = BTreeMap::new();
    for t in m.gguf().tensors() {
        let agg = out.entry(role_of(&t.name)).or_default();
        *agg.types.entry(t.dtype.name()).or_default() += 1;
        agg.bytes += t.byte_size;
        agg.count += 1;
    }
    out
}

fn fmt_types(a: &RoleAgg) -> String {
    a.types
        .iter()
        .map(|(n, c)| if *c > 1 { format!("{n}×{c}") } else { n.to_string() })
        .collect::<Vec<_>>()
        .join("+")
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: gguf-diff A.gguf B.gguf [--all]");
        return ExitCode::from(2);
    }
    let show_all = args.iter().any(|a| a == "--all");
    let (a, b) = match (MappedGguf::open(&args[1]), MappedGguf::open(&args[2])) {
        (Ok(a), Ok(b)) => (a, b),
        (Err(e), _) | (_, Err(e)) => {
            eprintln!("gguf-diff: {e:#}");
            return ExitCode::from(1);
        }
    };
    let (agg_a, agg_b) = (aggregate(&a), aggregate(&b));

    println!(
        "{:<44} {:<22} {:<22} {:>9} {:>9} {:>8}",
        "role", "A", "B", "A MiB", "B MiB", "Δ MiB"
    );
    let mut total_a = 0u64;
    let mut total_b = 0u64;
    let mut roles: Vec<&String> = agg_a.keys().chain(agg_b.keys()).collect();
    roles.sort();
    roles.dedup();
    // Sort by |delta| descending so the interesting rows lead.
    let mut rows: Vec<(&String, Option<&RoleAgg>, Option<&RoleAgg>)> = roles
        .into_iter()
        .map(|r| (r, agg_a.get(r), agg_b.get(r)))
        .collect();
    rows.sort_by_key(|(_, a, b)| {
        let ab = a.map_or(0, |x| x.bytes) as i64;
        let bb = b.map_or(0, |x| x.bytes) as i64;
        -(ab - bb).abs()
    });
    for (role, ra, rb) in rows {
        let (ab, bb) = (ra.map_or(0, |x| x.bytes), rb.map_or(0, |x| x.bytes));
        total_a += ab;
        total_b += bb;
        let ta = ra.map(fmt_types).unwrap_or_else(|| "—".into());
        let tb = rb.map(fmt_types).unwrap_or_else(|| "—".into());
        let same = ta == tb && ab == bb;
        if same && !show_all {
            continue;
        }
        println!(
            "{}{:<42} {:<22} {:<22} {:>9.1} {:>9.1} {:>+8.1}",
            if same { "  " } else { "* " },
            role,
            ta,
            tb,
            ab as f64 / (1 << 20) as f64,
            bb as f64 / (1 << 20) as f64,
            (bb as f64 - ab as f64) / (1 << 20) as f64,
        );
    }
    println!(
        "\nTOTAL: A {:.2} GiB, B {:.2} GiB, Δ {:+.2} GiB",
        total_a as f64 / (1u64 << 30) as f64,
        total_b as f64 / (1u64 << 30) as f64,
        (total_b as f64 - total_a as f64) / (1u64 << 30) as f64,
    );
    ExitCode::SUCCESS
}
