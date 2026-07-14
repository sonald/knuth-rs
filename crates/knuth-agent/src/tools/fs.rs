use std::path::Path;

use ai::Tool;
use async_trait::async_trait;
use encoding_rs::{Encoding, GB18030, UTF_16BE, UTF_16LE, UTF_8};
use once_cell::sync::Lazy;
use tokio::fs;
use tokio_util::sync::CancellationToken;

use super::{AgentTool, ToolInput, ToolOutcome};

const MAX_READ_BYTES: usize = 32 * 1024;

fn required_string<'a>(input: &'a ToolInput, name: &str) -> Result<&'a str, String> {
    input
        .get(name)
        .and_then(|value| value.as_str())
        .filter(|value| !value.is_empty())
        .ok_or_else(|| format!("{name} must be a non-empty string"))
}

fn output(value: String) -> ToolOutcome {
    ToolOutcome::Success(serde_json::json!({ "output": value }))
}

pub struct ReadFileTool {}

#[async_trait]
impl AgentTool for ReadFileTool {
    fn schema(&self) -> &Tool {
        &READ_FILE_SCHEMA
    }

    async fn invoke(
        &self,
        input: ToolInput,
        cancel_token: CancellationToken,
    ) -> Result<ToolOutcome, String> {
        let path = required_string(&input, "path")?;
        let offset = input.get("offset").map_or(Ok(1), |value| {
            value
                .as_u64()
                .filter(|value| *value >= 1)
                .ok_or_else(|| "offset must be an integer >= 1".to_string())
        })? as usize;
        let limit = input.get("limit").map_or(Ok(200), |value| {
            value
                .as_u64()
                .filter(|value| *value >= 1)
                .ok_or_else(|| "limit must be an integer >= 1".to_string())
        })? as usize;

        let content = tokio::select! {
            _ = cancel_token.cancelled() => return Err("File read cancelled".to_string()),
            result = fs::read_to_string(path) => result.map_err(|error| error.to_string())?,
        };
        let lines: Vec<&str> = content.split_inclusive('\n').collect();
        let selected = lines.iter().skip(offset - 1).take(limit);
        let mut rendered = Vec::new();
        let mut bytes = 0;

        for (index, line) in selected.enumerate() {
            let line_number = offset + index;
            let line_bytes = line.len();
            if line_bytes > MAX_READ_BYTES {
                return Err(format!(
                    "Line {line_number} is {line_bytes} bytes, exceeding read_file max of {MAX_READ_BYTES} bytes; no content returned"
                ));
            }
            bytes += line_bytes;
            if bytes > MAX_READ_BYTES {
                return Err(format!(
                    "Requested content exceeds read_file max of {MAX_READ_BYTES} bytes ({bytes} bytes needed); no content returned"
                ));
            }
            rendered.push(format!(
                "{line_number:4}: {}",
                line.trim_end_matches(['\n', '\r'])
            ));
        }

        if rendered.is_empty() {
            return Ok(output(format!(
                "No content found in the specified range (file has {} total lines)",
                lines.len()
            )));
        }

        let end_line = offset + rendered.len() - 1;
        Ok(output(format!(
            "File({path}) - Lines {offset}-{end_line} of {} total:\n{}",
            lines.len(),
            rendered.join("\n")
        )))
    }
}

pub struct WriteFileTool {}

#[async_trait]
impl AgentTool for WriteFileTool {
    fn schema(&self) -> &Tool {
        &WRITE_FILE_SCHEMA
    }

    async fn invoke(
        &self,
        input: ToolInput,
        cancel_token: CancellationToken,
    ) -> Result<ToolOutcome, String> {
        let path = required_string(&input, "path")?;
        let content = input
            .get("content")
            .and_then(|value| value.as_str())
            .ok_or("content must be a string")?;

        tokio::select! {
            _ = cancel_token.cancelled() => return Err("File write cancelled".to_string()),
            result = async {
                if let Some(parent) = Path::new(path).parent().filter(|parent| !parent.as_os_str().is_empty()) {
                    fs::create_dir_all(parent).await?;
                }
                fs::write(path, content).await
            } => result.map_err(|error| error.to_string())?,
        }
        Ok(output(format!("Wrote {path}")))
    }
}

pub struct EditFileTool {}

#[async_trait]
impl AgentTool for EditFileTool {
    fn schema(&self) -> &Tool {
        &EDIT_FILE_SCHEMA
    }

    async fn invoke(
        &self,
        input: ToolInput,
        cancel_token: CancellationToken,
    ) -> Result<ToolOutcome, String> {
        let path = required_string(&input, "path")?;
        let old_string = required_string(&input, "old_string")?;
        let new_string = input
            .get("new_string")
            .and_then(|value| value.as_str())
            .ok_or("new_string must be a string")?;
        if old_string == new_string {
            return Err("new_string must be different from old_string".to_string());
        }
        let replace_all = input.get("replace_all").map_or(Ok(false), |value| {
            value
                .as_bool()
                .ok_or_else(|| "replace_all must be a boolean".to_string())
        })?;

        let raw = tokio::select! {
            _ = cancel_token.cancelled() => return Err("File edit cancelled".to_string()),
            result = fs::read(path) => result.map_err(|error| error.to_string())?,
        };
        let (text, encoding) = decode_text(&raw)?;
        let count = text.matches(old_string).count();
        if count == 0 {
            return Err("old_string was not found".to_string());
        }
        if count > 1 && !replace_all {
            return Err(format!(
                "old_string found {count} matches; set replace_all=true to replace all"
            ));
        }

        let edited = if replace_all {
            text.replace(old_string, new_string)
        } else {
            text.replacen(old_string, new_string, 1)
        };
        let encoded = encode_text(&edited, encoding)?;
        tokio::select! {
            _ = cancel_token.cancelled() => return Err("File edit cancelled".to_string()),
            result = fs::write(path, encoded) => result.map_err(|error| error.to_string())?,
        }
        Ok(output(format!(
            "Edited {path} (replacements={}, encoding={})",
            if replace_all { count } else { 1 },
            encoding.name
        )))
    }
}

#[derive(Clone, Copy)]
struct TextEncoding {
    encoding: &'static Encoding,
    bom: &'static [u8],
    name: &'static str,
}

fn decode_text(raw: &[u8]) -> Result<(String, TextEncoding), String> {
    let candidates = if raw.starts_with(&[0xEF, 0xBB, 0xBF]) {
        vec![(UTF_8, &raw[3..], &b"\xEF\xBB\xBF"[..], "utf-8-sig")]
    } else if raw.starts_with(&[0xFF, 0xFE]) {
        vec![(UTF_16LE, &raw[2..], &b"\xFF\xFE"[..], "utf-16")]
    } else if raw.starts_with(&[0xFE, 0xFF]) {
        vec![(UTF_16BE, &raw[2..], &b"\xFE\xFF"[..], "utf-16")]
    } else if raw.iter().take(4096).any(|byte| *byte == 0) {
        let (even_zeroes, odd_zeroes) = raw.iter().enumerate().fold((0, 0), |counts, (i, byte)| {
            if *byte != 0 {
                counts
            } else if i % 2 == 0 {
                (counts.0 + 1, counts.1)
            } else {
                (counts.0, counts.1 + 1)
            }
        });
        if odd_zeroes >= even_zeroes {
            vec![
                (UTF_16LE, raw, &b""[..], "utf-16-le"),
                (UTF_16BE, raw, &b""[..], "utf-16-be"),
            ]
        } else {
            vec![
                (UTF_16BE, raw, &b""[..], "utf-16-be"),
                (UTF_16LE, raw, &b""[..], "utf-16-le"),
            ]
        }
    } else {
        vec![
            (UTF_8, raw, &b""[..], "utf-8"),
            (GB18030, raw, &b""[..], "gb18030"),
        ]
    };

    for (encoding, bytes, bom, name) in candidates {
        let (text, had_errors) = encoding.decode_without_bom_handling(bytes);
        if !had_errors {
            return Ok((
                text.into_owned(),
                TextEncoding {
                    encoding,
                    bom,
                    name,
                },
            ));
        }
    }
    Err("file is not a supported text encoding; supported encodings are utf-8-sig, utf-8, utf-16, utf-16-le, utf-16-be, gb18030".to_string())
}

fn encode_text(text: &str, encoding: TextEncoding) -> Result<Vec<u8>, String> {
    let encoded = if encoding.encoding == UTF_16LE {
        text.encode_utf16()
            .flat_map(u16::to_le_bytes)
            .collect::<Vec<_>>()
    } else if encoding.encoding == UTF_16BE {
        text.encode_utf16()
            .flat_map(u16::to_be_bytes)
            .collect::<Vec<_>>()
    } else {
        let (encoded, _, had_errors) = encoding.encoding.encode(text);
        if had_errors {
            return Err(format!(
                "edited content cannot be encoded as {}",
                encoding.name
            ));
        }
        encoded.into_owned()
    };
    let mut result = Vec::with_capacity(encoding.bom.len() + encoded.len());
    result.extend_from_slice(encoding.bom);
    result.extend_from_slice(&encoded);
    Ok(result)
}

static READ_FILE_SCHEMA: Lazy<Tool> = Lazy::new(|| Tool {
    name: "read_file".to_string(),
    description: include_str!("descriptions/read_file.md").trim().to_string(),
    parameters: serde_json::json!({
        "type": "object",
        "properties": {
            "path": { "type": "string" },
            "offset": { "type": "integer", "default": 1 },
            "limit": { "type": "integer", "default": 200 }
        },
        "required": ["path"],
        "additionalProperties": false
    }),
});

static WRITE_FILE_SCHEMA: Lazy<Tool> = Lazy::new(|| Tool {
    name: "write_file".to_string(),
    description: include_str!("descriptions/write_file.md")
        .trim()
        .to_string(),
    parameters: serde_json::json!({
        "type": "object",
        "properties": {
            "path": { "type": "string" },
            "content": { "type": "string" }
        },
        "required": ["path", "content"],
        "additionalProperties": false
    }),
});

static EDIT_FILE_SCHEMA: Lazy<Tool> = Lazy::new(|| Tool {
    name: "edit_file".to_string(),
    description: include_str!("descriptions/edit_file.md").trim().to_string(),
    parameters: serde_json::json!({
        "type": "object",
        "properties": {
            "path": { "type": "string" },
            "old_string": { "type": "string" },
            "new_string": { "type": "string" },
            "replace_all": { "type": "boolean", "default": false }
        },
        "required": ["path", "old_string", "new_string"],
        "additionalProperties": false
    }),
});

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use uuid::Uuid;

    fn input(value: serde_json::Value) -> ToolInput {
        value.as_object().unwrap().clone()
    }

    fn temp_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("knuth-agent-{}-{name}", Uuid::new_v4()))
    }

    #[tokio::test]
    async fn write_then_read_file() {
        let path = temp_path("notes/hello.txt");
        let path_string = path.to_string_lossy();
        WriteFileTool {}
            .invoke(
                input(json!({ "path": path_string, "content": "alpha\nbeta\n" })),
                CancellationToken::new(),
            )
            .await
            .unwrap();
        let result = ReadFileTool {}
            .invoke(
                input(json!({ "path": path_string, "offset": 2, "limit": 1 })),
                CancellationToken::new(),
            )
            .await
            .unwrap();

        match result {
            ToolOutcome::Success(value) => {
                assert!(value["output"].as_str().unwrap().contains("   2: beta"))
            }
        }
        fs::remove_dir_all(path.parent().unwrap()).await.unwrap();
    }

    #[tokio::test]
    async fn edit_file_preserves_supported_encodings() {
        for encoding in [
            TextEncoding {
                encoding: UTF_16LE,
                bom: b"\xFF\xFE",
                name: "utf-16",
            },
            TextEncoding {
                encoding: GB18030,
                bom: b"",
                name: "gb18030",
            },
        ] {
            let path = temp_path("encoded.txt");
            fs::write(&path, encode_text("你好\nbeta\n", encoding).unwrap())
                .await
                .unwrap();
            EditFileTool {}
                .invoke(
                    input(json!({ "path": path, "old_string": "beta", "new_string": "BETA" })),
                    CancellationToken::new(),
                )
                .await
                .unwrap();

            let raw = fs::read(&path).await.unwrap();
            let (text, detected) = decode_text(&raw).unwrap();
            assert_eq!(text, "你好\nBETA\n");
            assert_eq!(detected.name, encoding.name);
            fs::remove_file(path).await.unwrap();
        }
    }
}
