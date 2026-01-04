use rand::rngs::OsRng;
use secp256k1::{Secp256k1, SecretKey, PublicKey};
use sha3::{Digest, Keccak256};
use bs58::encode;
use rayon::prelude::*;
use clap::Parser;
use std::sync::{Arc, atomic::{AtomicUsize, AtomicBool, Ordering}};
use ocl::{ProQue, Buffer};
use tokio::runtime::Runtime;
use reqwest::Client;
use serde::Deserialize;
use indicatif::{ProgressBar, ProgressStyle};
use std::thread;

#[derive(Parser, Debug)]
struct Args {
    #[arg(short, long, default_value = "T")]
    prefix: String,
    #[arg(short, long, default_value = "")]
    suffix: String,
    #[arg(short, long, default_value_t = 1)]
    count: usize,
    #[arg(short, long, default_value_t = 4)]
    cpu_threads: usize,
    #[arg(short, long, default_value_t = 8192)]
    gpu_batch: usize,
    #[arg(long, default_value_t = false, help = "Enable TRX balance check")]
    check_balance: bool,
}

// ------------------
// TRX 地址生成
// ------------------
fn private_to_trx_address(priv_key: &SecretKey) -> String {
    let secp = Secp256k1::new();
    let pubkey = PublicKey::from_secret_key(&secp, priv_key);
    let pubkey_bytes = pubkey.serialize_uncompressed();
    let keccak = Keccak256::digest(&pubkey_bytes[1..]);
    let mut addr = vec![0x41u8];
    addr.extend_from_slice(&keccak[12..]);
    let checksum = &sha3::Sha256::digest(&sha3::Sha256::digest(&addr))[..4];
    addr.extend_from_slice(checksum);
    encode(addr).into_string()
}

fn check_vanity(addr: &str, prefix: &str, suffix: &str) -> bool {
    addr.starts_with(prefix) && addr.ends_with(suffix)
}

// ------------------
// TRON API 查询余额
// ------------------
#[derive(Deserialize)]
struct AccountData {
    balance: Option<u64>,
}

#[derive(Deserialize)]
struct AccountResponse {
    data: Option<Vec<AccountData>>,
}

async fn get_balance(client: &Client, address: &str) -> u64 {
    let url = format!("https://api.trongrid.io/v1/accounts/{}", address);
    match client.get(&url).send().await {
        Ok(resp) => {
            if let Ok(json) = resp.json::<AccountResponse>().await {
                if let Some(data) = json.data {
                    if let Some(account) = data.first() {
                        return account.balance.unwrap_or(0);
                    }
                }
            }
            0
        }
        Err(_) => 0,
    }
}

// ------------------
// GPU Worker
// ------------------
fn gpu_worker(prefix: String, suffix: String, batch_size: usize,
              found_count: Arc<AtomicUsize>, max_count: usize,
              stop_flag: Arc<AtomicBool>, client: Arc<Client>, rt: Arc<Runtime>,
              check_balance: bool) {

    let kernel_source = r#"
        __kernel void gen_random(__global ulong *out, uint seed) {
            int gid = get_global_id(0);
            ulong val = (ulong)(seed + gid) * 6364136223846793005UL + 1UL;
            out[gid] = val;
        }
    "#;

    let pro_que = ProQue::builder()
        .src(kernel_source)
        .dims(batch_size)
        .build().unwrap();

    let buffer = Buffer::<u64>::builder()
        .queue(pro_que.queue().clone())
        .flags(ocl::flags::MEM_WRITE_ONLY)
        .len(batch_size)
        .build().unwrap();

    while !stop_flag.load(Ordering::Relaxed) {
        let seed: u32 = rand::random();
        let kernel = pro_que.kernel_builder("gen_random")
            .arg(&buffer)
            .arg(seed)
            .build().unwrap();
        unsafe { kernel.enq().unwrap(); }

        let mut vals = vec![0u64; batch_size];
        buffer.read(&mut vals).enq().unwrap();

        for val in vals {
            let mut priv_bytes = val.to_be_bytes().to_vec();
            priv_bytes.resize(32, 0);
            let sk = SecretKey::from_slice(&priv_bytes).unwrap_or_else(|_| SecretKey::new(&mut OsRng));
            let addr = private_to_trx_address(&sk);

            if check_vanity(&addr, &prefix, &suffix) {
                if check_balance {
                    let balance = rt.block_on(get_balance(&client, &addr));
                    if balance > 0 {
                        println!("✅ GPU Found: {} | {} | Balance: {} TRX", addr, sk, balance as f64/1_000_000.0);
                        found_count.fetch_add(1, Ordering::Relaxed);
                    }
                } else {
                    println!("✅ GPU Found: {} | {}", addr, sk);
                    found_count.fetch_add(1, Ordering::Relaxed);
                }
                if found_count.load(Ordering::Relaxed) >= max_count {
                    stop_flag.store(true, Ordering::Relaxed);
                    return;
                }
            }
        }
    }
}

// ------------------
// CPU Worker
// ------------------
fn cpu_worker(prefix: String, suffix: String,
              found_count: Arc<AtomicUsize>, max_count: usize,
              stop_flag: Arc<AtomicBool>, client: Arc<Client>, rt: Arc<Runtime>,
              check_balance: bool) {
    let secp = Secp256k1::new();
    let mut rng = OsRng;

    while !stop_flag.load(Ordering::Relaxed) {
        let sk = SecretKey::new(&mut rng);
        let addr = private_to_trx_address(&sk);

        if check_vanity(&addr, &prefix, &suffix) {
            if check_balance {
                let balance = rt.block_on(get_balance(&client, &addr));
                if balance > 0 {
                    println!("✅ CPU Found: {} | {} | Balance: {} TRX", addr, sk, balance as f64/1_000_000.0);
                    found_count.fetch_add(1, Ordering::Relaxed);
                }
            } else {
                println!("✅ CPU Found: {} | {}", addr, sk);
                found_count.fetch_add(1, Ordering::Relaxed);
            }

            if found_count.load(Ordering::Relaxed) >= max_count {
                stop_flag.store(true, Ordering::Relaxed);
                return;
            }
        }
    }
}

// ------------------
// 主函数
// ------------------
fn main() {
    let args = Args::parse();
    let found_count = Arc::new(AtomicUsize::new(0));
    let stop_flag = Arc::new(AtomicBool::new(false));
    let client = Arc::new(Client::new());
    let rt = Arc::new(Runtime::new().unwrap());

    let pb = ProgressBar::new(args.count as u64);
    pb.set_style(ProgressStyle::with_template(
        "[{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})"
    ).unwrap());

    // GPU线程
    let gpu_prefix = args.prefix.clone();
    let gpu_suffix = args.suffix.clone();
    let found_count_clone = found_count.clone();
    let stop_flag_clone = stop_flag.clone();
    let client_clone = client.clone();
    let rt_clone = rt.clone();
    let gpu_batch = args.gpu_batch;
    let check_balance_flag = args.check_balance;

    let gpu_handle = thread::spawn(move || {
        gpu_worker(gpu_prefix, gpu_suffix, gpu_batch,
                   found_count_clone, args.count, stop_flag_clone, client_clone, rt_clone, check_balance_flag);
    });

    // CPU线程
    let mut cpu_handles = Vec::new();
    for _ in 0..args.cpu_threads {
        let cpu_prefix = args.prefix.clone();
        let cpu_suffix = args.suffix.clone();
        let found_count_clone = found_count.clone();
        let stop_flag_clone = stop_flag.clone();
        let client_clone = client.clone();
        let rt_clone = rt.clone();
        let check_balance_flag = args.check_balance;

        cpu_handles.push(thread::spawn(move || {
            cpu_worker(cpu_prefix, cpu_suffix, found_count_clone, args.count, stop_flag_clone, client_clone, rt_clone, check_balance_flag);
        }));
    }

    // 进度条更新
    while !stop_flag.load(Ordering::Relaxed) {
        pb.set_position(found_count.load(Ordering::Relaxed) as u64);
        thread::sleep(std::time::Duration::from_millis(500));
        if found_count.load(Ordering::Relaxed) >= args.count {
            stop_flag.store(true, Ordering::Relaxed);
            break;
        }
    }

    for h in cpu_handles { h.join().unwrap(); }
    gpu_handle.join().unwrap();
    pb.finish_with_message("Search completed!");
}
