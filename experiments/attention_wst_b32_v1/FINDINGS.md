# Balanced WST route for full verify batch 32

Candidate SHA-256 `2235ca0ca95ac673bfb9f4c2013e27743ff3f149c01191f03b1ce327246a0066` changes `ZH_WST_MAX_BATCH` from 24 to 32 in the embedded keell0renz attention source. This extends the already selected balanced WST full-verification route and its workspace allocation to B=32; it is stacked on the scored two-block VP2 winner.

Public judge-image H200, three seeds, 24 alternating read-flushed graph replays per row: all six outputs passed reference checks. Parent/candidate ratios for B32 full verify were 1.008, 1.012, 1.004; unchanged B16 control ratios were 1.003, 0.998, 0.999. This selected-shape result is not a whole-board score. Official full check `b1a3f689acbc49fe` passed 16/16 at 2.62457× (95% CI 2.58520–2.66394); target B32 full verify measured 184.8 µs versus 185.6 µs in the earlier scored run (separate runs). The exact checked file was submitted as scored job `e503062862744b16`, pending. Raw output and driver are preserved here.
