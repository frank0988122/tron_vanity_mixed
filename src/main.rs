use rand::rngs::OsRng;
use secp256k1::{Secp256k1, SecretKey, PublicKey};
use sha3::{Digest, Keccak256};
use bs58::encode;
use rayon::prelude::*;
use clap::Parser;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, AtomicBool, Ordering},
};
use std::thread;
use std::time::Duration;
use ocl::{ProQue, Buffer};

// ----------------------------
// CLI 参数
// ----------------------------
#[derive(Parser, Debug)]
#[command(author, version, about)]
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
}

// ----------------------------
// TRX 地址生成
// ----------------------------
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

// ----------------------------
// 前后缀匹配
// ----------------------------
fn check_vanity(address: &str, prefix: &str, suffix: &str) -> bool {
    address.starts_with(prefix) && address.ends_with(suffix)
}

// ----------------------------
// GPU Worker
// ----------------------------
fn gpu_worker(prefix: String, suffix: String, batch_size: usize, found_count: Arc<AtomicUsize>, max_count: usize, stop_flag: Arc<AtomicBool>) {
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
        .build().expect("Failed to build OpenCL ProQue");

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
            let mut extra = vec![0u8; 32 - priv_bytes.len()];
            priv_bytes.append(&mut extra);
            let sk = SecretKey::from_slice(&priv_bytes).unwrap_or_else(|_| SecretKey::new(&mut OsRng));
            let addr = private_to_trx_address(&sk);

            if check_vanity(&addr, &prefix, &suffix) {
                println!("✅ GPU Found: {} | {}", addr, sk);
                found_count.fetch_add(1, Ordering::Relaxed);
                if found_count.load(Ordering::Relaxed) >= max_count {
                    stop_flag.store(true, Ordering::Relaxed);
                    return;
                }
            }
        }
    }
}

// ----------------------------
// CPU Worker
// ----------------------------
fn cpu_worker(prefix: String, suffix: String, found_count: Arc<AtomicUsize>, max_count: usize, stop_flag: Arc<AtomicBool>) {
    let secp = Secp256k1::new();
    let mut rng = OsRng;

    while !stop_flag.load(Ordering::Relaxed) {
        let sk = SecretKey::new(&mut rng);
        let addr = private_to_trx_address(&sk);

        if check_vanity(&addr, &prefix, &suffix) {
            println!("✅ CPU Found: {} | {}", addr, sk);
            found_count.fetch_add(1, Ordering::Relaxed);
            if found_count.load(Ordering::Relaxed) >= max_count {
                stop_flag.store(true, Ordering::Relaxed);
                return;
            }
        }
    }
}

// ----------------------------
// 主函数
// ----------------------------
fn main() {
    let args = Args::parse();

    let found_count = Arc::new(AtomicUsize::new(0));
    let stop_flag = Arc::new(AtomicBool::new(false));

    println!("Starting CPU + GPU mixed TRX vanity search");
    println!("Prefix: {}, Suffix: {}, Count: {}, CPU threads: {}, GPU batch: {}",
             args.prefix, args.suffix, args.count, args.cpu_threads, args.gpu_batch);

    // --------------------------
    // 启动 GPU 线程
    // --------------------------
    let gpu_prefix = args.prefix.clone();
    let gpu_suffix = args.suffix.clone();
    let found_count_clone = found_count.clone();
    let stop_flag_clone = stop_flag.clone();
    let gpu_batch = args.gpu_batch;

    let gpu_handle = thread::spawn(move || {
        gpu_worker(gpu_prefix, gpu_suffix, gpu_batch, found_count_clone, args.count, stop_flag_clone);
    });

    // --------------------------
    // 启动 CPU 多线程
    // --------------------------
    let mut cpu_handles = Vec::new();
    for _ in 0..args.cpu_threads {
        let cpu_prefix = args.prefix.clone();
        let cpu_suffix = args.suffix.clone();
        let found_count_clone = found_count.clone();
        let stop_flag_clone = stop_flag.clone();

        cpu_handles.push(thread::spawn(move || {
            cpu_worker(cpu_prefix, cpu_suffix, found_count_clone, args.count, stop_flag_clone);
        }));
    }

    // 等待线程结束
    for h in cpu_handles {
        h.join().unwrap();
    }
    gpu_handle.join().unwrap();

    println!("Search finished! Total found: {}", found_count.load(Ordering::Relaxed));
}
