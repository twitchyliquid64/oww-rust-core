use crate::MatchConfig;
use std::collections::BTreeMap;
use std::time::Instant;

#[derive(Clone, Debug)]
enum StageResult {
    Noop,
    Matched,
    Timeout,
}

#[derive(Clone, Debug, Default)]
struct MatchStage {
    model: String,
    activation_threshold: f32,
    timeout_ms: usize,
}

impl MatchStage {
    fn eval(
        &mut self,
        started_time: Option<&Instant>,
        activations: &Vec<(String, f32)>,
    ) -> StageResult {
        if let Some((_, amt)) = activations.iter().find(|(n, _)| n == &self.model) {
            if amt >= &self.activation_threshold {
                return StageResult::Matched;
            }
        }

        if let Some(started) = started_time {
            if Instant::now().duration_since(*started).as_millis() > self.timeout_ms as u128 {
                return StageResult::Timeout;
            }
        }

        StageResult::Noop
    }
}

#[derive(Clone, Debug, Default)]
struct MatchState {
    current_stage: Option<(usize, Instant)>,
    stages: Vec<MatchStage>,
    action: String,
}

impl MatchState {
    fn do_action(&mut self, name: &String) {
        let mut spl = self.action.split(":");
        match spl.next() {
            Some("exit") => {
                let code: i32 = spl.next().map(|x| x.parse().unwrap_or(0)).unwrap_or(0);
                std::process::exit(code);
            }
            Some("exec") => {
                use std::process::Command;

                let mut c = Command::new("./".to_owned() + spl.next().unwrap());
                let cmd = c.current_dir(std::env::current_dir().unwrap()).args(spl);
                println!("spawning: {:?}", &cmd);
                println!("result: {:?}", cmd.spawn());
            }
            _ => {
                println!("{}: ignoring unhandled action {}", name, self.action);
            }
        };
    }

    fn eval(&mut self, name: &String, activations: &Vec<(String, f32)>) {
        match self.current_stage {
            Some((idx, started)) => {
                let res = self.stages[idx].eval(Some(&started), activations);

                match res {
                    StageResult::Noop => {
                        println!(
                            "{}[{}]: Awaiting activation ({:.02})",
                            name,
                            idx,
                            self.stages[idx].timeout_ms as f32 / 1000.0
                                - Instant::now().duration_since(started).as_secs_f32()
                        );
                    }
                    StageResult::Timeout => {
                        println!("{}[{}]: Timeout", name, idx);
                        self.current_stage = None;
                    }
                    StageResult::Matched => {
                        println!("{}[{}]: Activated", name, idx);
                        if idx >= self.stages.len() - 1 {
                            self.do_action(name);
                            self.current_stage = None;
                        } else {
                            self.current_stage = Some((idx + 1, Instant::now()));
                        }
                    }
                }
            }

            None => {
                if self.stages.len() > 0 {
                    if let StageResult::Matched = self.stages[0].eval(None, activations) {
                        if self.stages.len() >= 2 {
                            self.current_stage = Some((1, Instant::now()));
                        } else {
                            self.do_action(name);
                        }
                    }
                }
            }
        }
    }
}

#[derive(Debug)]
pub struct Matcher {
    matches: BTreeMap<String, MatchState>,
}

impl Matcher {
    pub fn new() -> Self {
        let matches = BTreeMap::new();

        Self { matches }
    }

    pub fn add_rule(&mut self, name: String, rule: MatchConfig) {
        let mut state = MatchState {
            action: rule.action,
            ..MatchState::default()
        };

        for stage in rule.chain {
            state.stages.push(MatchStage {
                model: stage.model,
                timeout_ms: stage.timeout_ms.unwrap_or(3200),
                activation_threshold: stage.activation_threshold.unwrap_or(0.5),
                ..MatchStage::default()
            })
        }

        self.matches.insert(name, state);
    }

    pub fn eval(&mut self, activations: Vec<(String, f32)>) {
        for (name, m) in self.matches.iter_mut() {
            m.eval(name, &activations);
        }
    }
}
