use nioxs_storage::NioDb;
use serde_json::json;
use std::time::Duration;
use tokio::fs;

pub async fn start_background_watcher(niodb: NioDb, workspace_root: String, deployment_id: String) {
    tokio::spawn(async move {
        loop {
            // Background daemon file-watcher loop
            if let Ok(mut entries) = fs::read_dir(&workspace_root).await {
                while let Ok(Some(entry)) = entries.next_entry().await {
                    let path = entry.path();
                    if path.is_file() {
                        if let Some(ext) = path.extension() {
                            if ext == "rs" || ext == "ts" || ext == "py" || ext == "md" {
                                if let Ok(content) = fs::read_to_string(&path).await {
                                    let key = format!("file_index:{}", path.display());
                                    let data = json!({
                                        "schema_version": 2,
                                        "deployment_id": deployment_id,
                                        "path": path.display().to_string(),
                                        "size": content.len(),
                                        // Just index the presence and size to stay lightweight
                                    });
                                    let _ =
                                        niodb.append_record("workspace_index", &key, &data).await;
                                }
                            }
                        }
                    }
                }
            }
            tokio::time::sleep(Duration::from_secs(30)).await;
        }
    });
}
