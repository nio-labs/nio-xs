//! Shared native/runner decoding policy. The native vocabulary path is greedy.
pub(crate) const HISTORY_WINDOW: usize = 64;

#[derive(Clone, Copy)]
pub(crate) struct DecodingPolicy {
    pub repeat_penalty: f32,
    pub history_window: usize,
    pub suppressed: &'static [u32],
}

impl DecodingPolicy {
    pub fn for_model(
        skate: bool,
        manta: bool,
        beluga: bool,
        blue: bool,
        cheetah: bool,
        tiger: bool,
        fox: bool,
        rabbit: bool,
    ) -> Self {
        Self {
            repeat_penalty: if cheetah || tiger || fox {
                1.10
            } else if beluga || rabbit {
                1.02
            } else if blue {
                1.05
            } else if manta {
                1.15
            } else {
                1.10
            },
            history_window: if manta || tiger || fox || rabbit {
                128
            } else {
                HISTORY_WINDOW
            },
            suppressed: if skate {
                &[3725, 47182]
            } else if manta {
                &[41233, 54275, 73594]
            } else if tiger {
                &[95456, 39814]
            } else {
                &[]
            },
        }
    }

    pub fn runner_args(self) -> Vec<String> {
        let mut args: Vec<String> = [
            "--temp",
            "0",
            "--top-p",
            "1",
            "--top-k",
            "0",
            "--min-p",
            "0",
            "--frequency-penalty",
            "0",
            "--presence-penalty",
            "0",
            "--no-context-shift",
            "--single-turn",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect();
        let repeat_position = args
            .iter()
            .position(|arg| arg == "--no-context-shift")
            .unwrap_or(args.len());
        args.splice(
            repeat_position..repeat_position,
            [
                "--repeat-last-n".to_owned(),
                self.history_window.to_string(),
            ],
        );
        args.extend(["--repeat-penalty".into(), self.repeat_penalty.to_string()]);
        for token in self.suppressed {
            args.extend(["--logit-bias".into(), format!("{token}-inf")]);
        }
        args
    }
}

/// Apply the GGML repetition penalty once per distinct token in the window.
pub(crate) fn repetition_score(score: f32, token: u32, history: &[u32], penalty: f32) -> f32 {
    if penalty <= 1.0 || !history.contains(&token) {
        return score;
    }
    if score > 0.0 {
        score / penalty
    } else {
        score * penalty
    }
}

#[cfg(test)]
pub(crate) fn push_history(history: &mut Vec<u32>, token: u32) {
    push_history_window(history, token, HISTORY_WINDOW);
}

pub(crate) fn push_history_window(history: &mut Vec<u32>, token: u32, window: usize) {
    history.push(token);
    if history.len() > window {
        history.drain(..history.len() - window);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn repetition_matches_ggml_for_positive_negative_and_repeated_tokens() {
        assert_eq!(repetition_score(12.0, 7, &[7], 1.2), 10.0);
        assert_eq!(repetition_score(12.0, 7, &[7, 7, 7], 1.2), 10.0);
        assert_eq!(repetition_score(-10.0, 7, &[7], 1.2), -12.0);
        assert_eq!(repetition_score(12.0, 7, &[8], 1.2), 12.0);
        assert_eq!(repetition_score(12.0, 7, &[7, 7], 1.0), 12.0);
    }
    #[test]
    fn history_expires_old_prompt_and_output_tokens() {
        let mut history = (0..64).collect();
        push_history(&mut history, 64);
        assert_eq!(history, (1..65).collect::<Vec<_>>());
    }
    #[test]
    fn runner_policy_preserves_companion_penalties_and_suppressions() {
        for (flags, expected_penalty, expected_biases) in [
            (
                (true, false, false, false, false, false, false, false),
                "1.1",
                vec!["3725-inf", "47182-inf"],
            ),
            (
                (false, true, false, false, false, false, false, false),
                "1.15",
                vec!["41233-inf", "54275-inf", "73594-inf"],
            ),
            (
                (false, false, true, false, false, false, false, false),
                "1.02",
                vec![],
            ),
            (
                (false, false, false, true, false, false, false, false),
                "1.05",
                vec![],
            ),
            (
                (false, false, false, false, true, false, false, false),
                "1.1",
                vec![],
            ),
            (
                (false, false, false, false, false, true, false, false),
                "1.1",
                vec!["95456-inf", "39814-inf"],
            ),
            (
                (false, false, false, false, false, false, true, false),
                "1.1",
                vec![],
            ),
            (
                (false, false, false, false, false, false, false, true),
                "1.02",
                vec![],
            ),
        ] {
            let args = DecodingPolicy::for_model(
                flags.0, flags.1, flags.2, flags.3, flags.4, flags.5, flags.6, flags.7,
            )
            .runner_args();
            let value = |key| args.windows(2).find(|w| w[0] == key).map(|w| w[1].as_str());
            assert_eq!(value("--repeat-penalty"), Some(expected_penalty));
            assert_eq!(
                value("--repeat-last-n"),
                Some(if flags.1 || flags.5 || flags.6 || flags.7 {
                    "128"
                } else {
                    "64"
                })
            );
            assert_eq!(value("--temp"), Some("0"));
            assert_eq!(value("--frequency-penalty"), Some("0"));
            assert_eq!(value("--presence-penalty"), Some("0"));
            let biases: Vec<_> = args
                .windows(2)
                .filter(|w| w[0] == "--logit-bias")
                .map(|w| w[1].as_str())
                .collect();
            assert_eq!(biases, expected_biases);
        }
    }
}
