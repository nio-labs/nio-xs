use nioxs_engine::ast_engine::memory_mb;

fn main() {
    let (rss, hwm) = memory_mb();
    println!("Memory: RSS={:.2} MB, HWM={:.2} MB", rss, hwm);
    #[cfg(target_os = "linux")]
    unsafe {
        libc::malloc_trim(0);
    }
}
