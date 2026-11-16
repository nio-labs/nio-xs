//! Schedule verified drafting without changing token acceptance rules.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Mode {
    On,
    Off,
    Adaptive,
}

impl Mode {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::On => "on",
            Self::Off => "off",
            Self::Adaptive => "adaptive",
        }
    }
}

pub(crate) struct Schedule {
    mode: Mode,
    misses: usize,
    cooldown: usize,
}

impl Schedule {
    pub(crate) fn new(mode: Mode) -> Self {
        Self {
            mode,
            misses: 0,
            cooldown: 0,
        }
    }

    pub(crate) fn should_draft(&mut self) -> bool {
        if self.mode == Mode::Off {
            return false;
        }
        if self.mode == Mode::Adaptive && self.cooldown > 0 {
            self.cooldown -= 1;
            return false;
        }
        true
    }

    pub(crate) fn observe(&mut self, matched: usize) {
        if self.mode != Mode::Adaptive {
            return;
        }
        if matched > 0 {
            self.misses = 0;
        } else {
            self.misses += 1;
            if self.misses >= 2 {
                self.misses = 0;
                self.cooldown = 8;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repeated_misses_back_off_then_probe_again() {
        let mut schedule = Schedule::new(Mode::Adaptive);
        schedule.observe(0);
        assert!(schedule.should_draft());
        schedule.observe(0);
        for _ in 0..8 {
            assert!(!schedule.should_draft());
        }
        assert!(schedule.should_draft());
        schedule.observe(1);
        assert!(schedule.should_draft());
    }

    #[test]
    fn successful_verification_breaks_a_miss_streak() {
        let mut schedule = Schedule::new(Mode::Adaptive);
        schedule.observe(0);
        schedule.observe(1);
        schedule.observe(0);
        assert!(schedule.should_draft());
    }

    #[test]
    fn explicit_controls_do_not_adapt() {
        for mode in [Mode::On, Mode::Off] {
            let mut schedule = Schedule::new(mode);
            for _ in 0..20 {
                schedule.observe(0);
                assert_eq!(schedule.should_draft(), mode == Mode::On);
            }
        }
    }
}
