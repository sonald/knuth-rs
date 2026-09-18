//! Minimal skill support.
//!
//! A skill is a directory holding a `SKILL.md`: YAML frontmatter with a `name`
//! and a `description`, followed by the instructions themselves. Skills are
//! discovered from `~/.agents/skills` and `<workspace>/.agents/skills`, and are
//! surfaced to the model through a single `skill` tool, so the instructions only
//! enter the context when the model actually asks for them.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use ai::Tool;
use async_trait::async_trait;
use knuth_agent::{
    AgentTool, ToolCapabilities, ToolDescription, ToolError, ToolInput, ToolResult, required_string,
};
use knuth_core::ToolOutcome;
use serde::Deserialize;
use tokio_util::sync::CancellationToken;
use tracing::{debug, warn};

/// Directory holding skills, relative to a workspace root or a home directory.
const SKILLS_DIR: &str = ".agents/skills";

/// File holding a skill's instructions.
const SKILL_FILE: &str = "SKILL.md";

/// Name of the tool that loads a skill.
const SKILL_TOOL_NAME: &str = "skill";

#[derive(Debug, Deserialize)]
struct SkillFrontmatter {
    name: Option<String>,
    description: Option<String>,
}

/// A skill discovered on disk.
#[derive(Debug, Clone)]
pub struct Skill {
    pub name: String,
    pub description: String,
    /// Path of the `SKILL.md` itself.
    pub path: PathBuf,
}

impl Skill {
    /// The directory the skill lives in. Instructions commonly point at sibling
    /// files, so the tool result reports it.
    fn dir(&self) -> &Path {
        self.path.parent().unwrap_or(Path::new(""))
    }
}

/// Every skill visible to a session, keyed by name.
#[derive(Debug, Clone, Default)]
pub struct SkillSet {
    skills: BTreeMap<String, Skill>,
}

impl SkillSet {
    /// Discovers skills from the workspace and from `~/.agents/skills`.
    pub fn discover(workspace: &Path) -> Self {
        let global = home_dir().map(|home| home.join(SKILLS_DIR));
        Self::discover_in(workspace, global.as_deref())
    }

    /// Both directories are optional, and a workspace skill shadows a global
    /// one with the same name.
    pub fn discover_in(workspace: &Path, global: Option<&Path>) -> Self {
        let mut skills = BTreeMap::new();
        if let Some(global) = global {
            load_dir(global, &mut skills);
        }
        load_dir(&workspace.join(SKILLS_DIR), &mut skills);
        Self { skills }
    }

    pub fn len(&self) -> usize {
        self.skills.len()
    }

    pub fn is_empty(&self) -> bool {
        self.skills.is_empty()
    }

    pub fn get(&self, name: &str) -> Option<&Skill> {
        self.skills.get(name)
    }

    pub fn names(&self) -> Vec<&str> {
        self.skills.keys().map(String::as_str).collect()
    }

    /// Renders the `- name: description` index the model chooses from.
    fn catalog(&self) -> String {
        self.skills
            .values()
            .map(|skill| format!("<skill><name>{}</name><description>{}</description></skill>\n", skill.name, skill.description))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// Loads every `<dir>/<name>/SKILL.md` into `skills`. Later directories win, so
/// the caller scans global skills before workspace ones.
fn load_dir(dir: &Path, skills: &mut BTreeMap<String, Skill>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(error) => {
            debug!("no skills in {}: {error}", dir.display());
            return;
        }
    };

    for entry in entries.flatten() {
        let path = entry.path().join(SKILL_FILE);
        if !path.is_file() {
            continue;
        }
        match load_skill(&path) {
            Ok(skill) => {
                debug!("loaded skill {} from {}", skill.name, path.display());
                skills.insert(skill.name.clone(), skill);
            }
            // A broken skill must not take the whole session down with it.
            Err(error) => warn!("skipping skill {}: {error}", path.display()),
        }
    }
}

fn load_skill(path: &Path) -> Result<Skill, String> {
    let text = std::fs::read_to_string(path).map_err(|error| error.to_string())?;
    let (frontmatter, _) =
        split_frontmatter(&text).ok_or_else(|| "missing `---` frontmatter".to_string())?;
    let frontmatter = parse_frontmatter(frontmatter)?;

    // Without a description the model has no way to tell what the skill is for.
    let description = frontmatter
        .description
        .map(|description| description.trim().to_string())
        .filter(|description| !description.is_empty())
        .ok_or_else(|| "frontmatter has no description".to_string())?;

    let name = frontmatter
        .name
        .map(|name| name.trim().to_string())
        .filter(|name| !name.is_empty())
        .or_else(|| {
            path.parent()
                .and_then(Path::file_name)
                .map(|name| name.to_string_lossy().into_owned())
        })
        .ok_or_else(|| "frontmatter has no name and the directory has no name".to_string())?;

    Ok(Skill {
        name,
        description,
        path: path.to_path_buf(),
    })
}

/// Parses the YAML frontmatter, retrying once with bare colons quoted.
///
/// Skill files are written for lenient parsers, so a description such as
/// `... a coding task: features, refactors ...` is ordinary in practice. YAML
/// reads that inner `: ` as the start of a nested mapping and rejects the whole
/// frontmatter, which would drop an otherwise fine skill.
///
/// The retry only runs once the strict parse has already failed, so it cannot
/// change how a well-formed file is read. When it fails too, the error reported
/// is the one from the original text.
fn parse_frontmatter(frontmatter: &str) -> Result<SkillFrontmatter, String> {
    match serde_yaml::from_str(frontmatter) {
        Ok(parsed) => Ok(parsed),
        Err(error) => serde_yaml::from_str(&quote_bare_colon_values(frontmatter))
            .map_err(|_| error.to_string()),
    }
}

/// Quotes top-level scalar values that contain a `: `.
///
/// Indented lines — nested keys and block scalar bodies — are left alone, as are
/// values that already carry YAML structure (quoted, block, or flow style).
fn quote_bare_colon_values(frontmatter: &str) -> String {
    frontmatter
        .lines()
        .map(|line| match line.split_once(": ") {
            Some((key, value))
                if !line.starts_with([' ', '\t'])
                    && !value.starts_with(['"', '\'', '|', '>', '[', '{', '&', '*'])
                    && value.contains(": ") =>
            {
                format!(
                    "{key}: \"{}\"",
                    value.replace('\\', "\\\\").replace('"', "\\\"")
                )
            }
            _ => line.to_string(),
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Splits a `SKILL.md` into its frontmatter and the body that follows it.
fn split_frontmatter(text: &str) -> Option<(&str, &str)> {
    let rest = text
        .strip_prefix("---\r\n")
        .or_else(|| text.strip_prefix("---\n"))?;

    let mut offset = 0;
    for line in rest.split_inclusive('\n') {
        if line.trim_end_matches(['\r', '\n']).trim_end() == "---" {
            return Some((&rest[..offset], &rest[offset + line.len()..]));
        }
        offset += line.len();
    }
    None
}

fn home_dir() -> Option<PathBuf> {
    home_dir_from_env(|name| std::env::var(name).ok())
}

fn home_dir_from_env<F>(env_var: F) -> Option<PathBuf>
where
    F: Fn(&str) -> Option<String>,
{
    #[cfg(target_os = "windows")]
    let home = env_var("USERPROFILE");
    #[cfg(not(target_os = "windows"))]
    let home = env_var("HOME");
    home.map(PathBuf::from)
}

/// Loads a skill's instructions on demand. The catalogue lives in the tool
/// description, which the model sees every turn, so no system prompt section is
/// needed to advertise it.
pub struct SkillTool {
    skills: Arc<SkillSet>,
    schema: Tool,
}

impl SkillTool {
    /// `None` when nothing was discovered, so no useless tool is registered.
    pub fn new(skills: SkillSet) -> Option<Self> {
        if skills.is_empty() {
            return None;
        }
        let schema = Tool {
            name: SKILL_TOOL_NAME.to_string(),
            description: format!(
                "Load the full instructions for a named skill. Call this before \
                 starting a task that one of the skills below covers.\n\n\
                 <available_skills>\n{}</available_skills>\n",
                skills.catalog()
            ),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Skill name, exactly as listed in the tool description."
                    }
                },
                "required": ["name"],
                "additionalProperties": false
            }),
        };
        Some(Self {
            skills: Arc::new(skills),
            schema,
        })
    }
}

#[async_trait]
impl AgentTool for SkillTool {
    fn schema(&self) -> &Tool {
        &self.schema
    }

    fn description(&self) -> ToolDescription {
        ToolDescription {
            id: SKILL_TOOL_NAME.into(),
            introduction: None,
            capabilities: ToolCapabilities::READ_FILE,
        }
    }

    async fn execute(
        &self,
        input: ToolInput,
        _cancel_token: CancellationToken,
    ) -> Result<ToolResult, ToolError> {
        let name = required_string(&input, "name")?;
        let Some(skill) = self.skills.get(name) else {
            return Err(ToolError::ArgumentError(format!(
                "unknown skill \"{name}\"; available skills: {}",
                self.skills.names().join(", ")
            )));
        };

        // Unlike `read_file` this reads the whole file: a skill is meant to be
        // loaded in full, and some exceed the 32KB read_file limit.
        let text = tokio::fs::read_to_string(&skill.path)
            .await
            .map_err(|error| {
                ToolError::Message(format!("failed to read {}: {error}", skill.path.display()))
            })?;
        let body = split_frontmatter(&text).map_or(text.as_str(), |(_, body)| body);

        Ok(ToolResult {
            outcome: ToolOutcome::ExecSuccess,
            content: format!(
                "Skill \"{}\" (from {}):\n\n{}",
                skill.name,
                skill.dir().display(),
                body.trim()
            ),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use uuid::Uuid;

    fn temp_root(label: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!("knuth-cli-skill-{label}-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        root
    }

    /// Writes a skill into the skills directory `skills_dir` itself, which is
    /// what `discover_in` takes for the global side and what a workspace root
    /// holds under `.agents/skills`.
    fn write_skill(skills_dir: &Path, dir: &str, content: &str) {
        let dir = skills_dir.join(dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(SKILL_FILE), content).unwrap();
    }

    /// The directory `discover_in` scans inside a workspace root.
    fn workspace_skills(workspace: &Path) -> PathBuf {
        workspace.join(SKILLS_DIR)
    }

    fn skill_file(name: &str, description: &str) -> String {
        format!("---\nname: {name}\ndescription: {description}\n---\n\n# {name}\n\nbody text\n")
    }

    fn input(value: serde_json::Value) -> ToolInput {
        value.as_object().unwrap().clone()
    }

    #[test]
    fn workspace_skill_shadows_global_one_with_the_same_name() {
        let global = temp_root("global");
        let workspace = temp_root("workspace");
        write_skill(&global, "shared", &skill_file("shared", "from global"));
        write_skill(&global, "only-global", &skill_file("only-global", "global"));
        write_skill(
            &workspace_skills(&workspace),
            "shared",
            &skill_file("shared", "from workspace"),
        );

        let skills = SkillSet::discover_in(&workspace, Some(&global));

        assert_eq!(skills.len(), 2);
        assert_eq!(skills.get("shared").unwrap().description, "from workspace");
        assert!(skills.get("only-global").is_some());

        std::fs::remove_dir_all(global).unwrap();
        std::fs::remove_dir_all(workspace).unwrap();
    }

    #[test]
    fn skips_skills_with_unusable_frontmatter() {
        let workspace = temp_root("broken");
        write_skill(
            &workspace_skills(&workspace),
            "good",
            &skill_file("good", "usable"),
        );
        write_skill(
            &workspace_skills(&workspace),
            "no-frontmatter",
            "# Just a heading\n",
        );
        write_skill(
            &workspace,
            "no-description",
            "---\nname: no-description\n---\n\nbody\n",
        );
        write_skill(
            &workspace_skills(&workspace),
            "bad-yaml",
            "---\nname: [unclosed\n---\n\nbody\n",
        );

        let skills = SkillSet::discover_in(&workspace, None);

        assert_eq!(skills.names(), vec!["good"]);

        std::fs::remove_dir_all(workspace).unwrap();
    }

    #[test]
    fn falls_back_to_directory_name_when_frontmatter_has_no_name() {
        let workspace = temp_root("fallback");
        write_skill(
            &workspace_skills(&workspace),
            "dir-name",
            "---\ndescription: described\n---\n\nbody\n",
        );

        let skills = SkillSet::discover_in(&workspace, None);

        assert_eq!(skills.names(), vec!["dir-name"]);

        std::fs::remove_dir_all(workspace).unwrap();
    }

    /// Mirrors `~/.agents/skills/orchestrate/SKILL.md`: a bare `: ` in the
    /// description, plus embedded double quotes that the retry has to escape
    /// without leaving backslashes in the parsed value.
    #[test]
    fn accepts_a_description_containing_a_bare_colon() {
        let workspace = temp_root("bare-colon");
        write_skill(
            &workspace_skills(&workspace),
            "orchestrate",
            "---\nname: orchestrate\ndescription: applies to any coding task: features, refactors, or asks to \"orchestrate\" a task\n---\n\nbody\n",
        );

        let skills = SkillSet::discover_in(&workspace, None);

        assert_eq!(
            skills.get("orchestrate").unwrap().description,
            "applies to any coding task: features, refactors, or asks to \"orchestrate\" a task"
        );

        std::fs::remove_dir_all(workspace).unwrap();
    }

    /// Quoted and block scalar descriptions are shapes the skills directory
    /// really uses, and they parse strictly — the quoting retry must not touch
    /// them.
    #[test]
    fn reads_quoted_and_block_scalar_descriptions() {
        let workspace = temp_root("scalars");
        write_skill(
            &workspace_skills(&workspace),
            "quoted",
            "---\nname: quoted\ndescription: \"has: a colon\"\n---\n\nbody\n",
        );
        write_skill(
            &workspace_skills(&workspace),
            "block",
            "---\nname: block\ndescription: |\n  line one\n  line two\n---\n\nbody\n",
        );

        let skills = SkillSet::discover_in(&workspace, None);

        assert_eq!(skills.get("quoted").unwrap().description, "has: a colon");
        assert_eq!(
            skills.get("block").unwrap().description,
            "line one\nline two"
        );

        std::fs::remove_dir_all(workspace).unwrap();
    }

    #[test]
    fn no_tool_is_registered_without_skills() {
        assert!(SkillTool::new(SkillSet::default()).is_none());
    }

    #[tokio::test]
    async fn tool_returns_the_body_without_frontmatter() {
        let workspace = temp_root("tool");
        write_skill(
            &workspace_skills(&workspace),
            "demo",
            &skill_file("demo", "demo skill"),
        );
        let tool = SkillTool::new(SkillSet::discover_in(&workspace, None)).unwrap();

        let result = tool
            .execute(input(json!({ "name": "demo" })), CancellationToken::new())
            .await
            .unwrap();

        assert_eq!(result.outcome, ToolOutcome::ExecSuccess);
        assert!(
            result.content.contains("body text"),
            "content={}",
            result.content
        );
        assert!(
            !result.content.contains("description: demo skill"),
            "content={}",
            result.content
        );

        std::fs::remove_dir_all(workspace).unwrap();
    }

    #[tokio::test]
    async fn tool_lists_available_names_for_an_unknown_skill() {
        let workspace = temp_root("unknown");
        write_skill(
            &workspace_skills(&workspace),
            "demo",
            &skill_file("demo", "demo skill"),
        );
        let tool = SkillTool::new(SkillSet::discover_in(&workspace, None)).unwrap();

        let error = tool
            .execute(input(json!({ "name": "nope" })), CancellationToken::new())
            .await
            .unwrap_err();

        let message = error.to_string();
        assert!(
            message.contains("unknown skill \"nope\""),
            "message={message}"
        );
        assert!(message.contains("demo"), "message={message}");

        std::fs::remove_dir_all(workspace).unwrap();
    }
}
