use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Formats a UUID showing only its trailing hex digits.
///
/// IDs in this crate are generated with `Uuid::now_v7`, which packs a
/// millisecond timestamp into the leading bits. Events emitted close
/// together therefore share a long, uninformative common prefix; only the
/// tail carries enough entropy to tell IDs apart at a glance in logs.
fn short_uuid(id: &Uuid) -> impl std::fmt::Display {
    let s = id.simple().to_string();
    format!("…{}", &s[s.len() - 8..])
}

macro_rules! define_id {
    ($name:ident) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
        pub struct $name(Uuid);

        impl $name {
            pub fn new() -> Self {
                Self(Uuid::now_v7())
            }

            pub fn as_uuid(&self) -> Uuid {
                self.0
            }

            pub fn short(&self) -> impl std::fmt::Display {
                short_uuid(&self.0)
            }
        }

        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{}", self.0)
            }
        }

        impl From<Uuid> for $name {
            fn from(uuid: Uuid) -> Self {
                Self(uuid)
            }
        }
    };
}

define_id!(SessionId);
define_id!(MessageId);
define_id!(TurnId);
define_id!(StepId);
define_id!(ToolInvocationId);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Generation(u64);

impl Generation {
    pub fn new() -> Self {
        Self(0)
    }

    pub fn next(&self) -> Self {
        Self(self.0 + 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn id_newtype_keeps_uuid_wire_format() {
        let uuid = Uuid::parse_str("018f47a2-9b1c-7a3d-8e4f-0123456789ab").unwrap();
        let id = MessageId::from(uuid);

        assert_eq!(
            serde_json::to_string(&id).unwrap(),
            "\"018f47a2-9b1c-7a3d-8e4f-0123456789ab\""
        );
        assert_eq!(
            serde_json::from_str::<MessageId>("\"018f47a2-9b1c-7a3d-8e4f-0123456789ab\"").unwrap(),
            id
        );
        assert_eq!(id.as_uuid(), uuid);
    }
}
