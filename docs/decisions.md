# Every decision, and why

A complete account of what was built and what was chosen, in the order the data flows.
Nothing here is assumed knowledge — where a technique is used, it is explained.

---

## 1. The target: what counts as a breach

**Decision: a breach is a week totalling more than 55 hours. Nothing else.**

The law says 45 ordinary hours plus at most 10 hours of overtime. That could mean several
things in practice, so rather than guess I checked against the client's own system.

`weekly_summary.csv` came across with the export and already contains a `breached` column.
I recomputed every employee's weekly hours from the raw clock data and compared:

- **2,122 employee-weeks, zero disagreements**, not a single row off by even a hundredth of an hour.
- In every row, `overtime_hours == max(total_hours − 45, 0)`.
- In every row, `breached == (overtime_hours > 10)`.

So the client's engine uses exactly three rules, and I copied them:
a shift belongs to the week of its `shift_date`; an overnight shift wraps to the next morning;
a missing clock-out contributes zero.

**Why this matters more than it sounds.** We are graded against ground truth we cannot see.
Matching their arithmetic exactly means we are measuring the same thing they are. Any
"improvement" to the calculation would move us *away* from the answer being marked.

**The consequence that shaped everything else:** Sunday and public-holiday work is paid at 2×
and overtime at 1.5×, but **no multiplier appears anywhere in the cap**. Pay rates affect cost,
not compliance. And because Monday-to-Wednesday hours are already known exactly, the only
unknown is the rest of the week. This is a forecasting problem wearing a classification costume.

---

## 2. Storage: SQLite, with one derived column

**Decision: raw CSVs load verbatim into SQLite; only `shifts` gets extra columns.**

Every input column is stored under its original name, including the client's own
`weekly_summary`, so the reconciliation above is a SQL query rather than an act of faith.

**The exception, and why it exists.** 1,010 of 8,863 shifts clock out *before* they clock in,
because they run through the night. Making every future piece of code remember to add 24 hours
is the kind of thing that gets forgotten exactly once and then silently corrupts an answer. So
ingestion computes it once:

```
start_at = shift_date + clock_in_time
end_at   = shift_date + clock_out_time, plus one day if that lands at or before start_at
end_at   = NULL when there is no clock-out (184 shifts)
```

Duration is then `end_at − start_at` with no special cases. Verified: all 1,010 rolled
correctly, no shift ends before it starts, durations span 3.5h to 13.5h with nothing implausible.

**Decision: the reporting week is derived, never hardcoded.** A database view computes the
current week from `max(shift_date)`. Next Monday's export moves it forward on its own.

**Decision: validate everything before writing anything.** A new export is loaded into a
scratch database and only swapped in once it passes. A bad upload leaves the live dashboard
untouched. Errors are collected and reported together in plain language, naming the file and
the problem, because the person pressing upload is not a developer.

**Decision: loading is additive.** Re-sending the same export corrects rows rather than
duplicating them; an export containing only the new week merges into what we already hold.
Tested: splitting the data at 2026-08-05 and loading the halves in order lands on exactly the
same 8,863 rows as a single load.

**A small decision with real consequences: read the CSVs strictly.** Pandas' default reader
converts 29 notes that literally say `n/a` into missing values, silently merging real
supervisor input into the blank bucket. Only genuinely empty fields are treated as missing.

---

## 3. What the data actually says

Five findings, because they constrain everything downstream.

**The event is rare.** 3.4% base rate — about 7 people a week out of 213. Predicting
"nobody breaches" is right 96.6% of the time, so accuracy is a useless measure here.

**Breaching does not stick to people.** 49 employees breached at least once, but almost all
did it exactly once. The correlation between breaching this week and last week is **0.05**.
So "who breached last week" — the most obvious baseline available — is worth nothing. The
cause is what is happening on site this week, not a set of habitual over-workers.

**The week compensates against itself.** Hours worked Monday–Wednesday correlate **−0.61**
with hours worked Thursday–Sunday. A heavy start buys a light weekend. This kills the
intuitive approach of scaling up the current pace.

**Most of a breach happens after the data stops.** People who breached were about 8 hours
ahead by Wednesday — and a further 8 hours ahead over the weekend.

**Nobody has ever breached by a Wednesday.** In all 1,917 completed employee-weeks, not once.
55 hours over three days needs 18.3h a day and the longest shift in the data is 13.5h. So
there is no "already over" group to separate out — every prediction is a genuine forecast.
This is why the dashboard says *at risk*, never *will breach*.

---

## 4. How much signal exists at all

Before choosing a model I measured the ceiling, because the answer determines whether model
choice matters.

Predicting Thursday-to-Sunday hours:

| features | R² |
|---|---|
| hours by Wednesday | 0.363 |
| + days worked | 0.366 |
| + the employee's own average week | **0.508** |
| + prior Thu–Sun hours, prior Thu–Sun days | 0.509 |

**R² means "share of the variation explained".** 0.508 means about half the differences
between people's weekends are predictable and half are not. Adding more history features
moved it by one thousandth — there is nothing else in this data to find.

**Why it stops there.** Thursday-to-Sunday hours are 79% explained by *days worked alone*,
because hours-per-day barely varies (mean 10.00h, standard deviation 1.76h). So the whole
problem is "how many more days will they work?" — and that is only 40% predictable. The rest
is roster randomness that has not been decided yet.

**And there is no structure for a clever model to exploit.** Of the variation in weekend
hours, sites explain **0.6%** and weeks explain **0.4%**. Individual differences explain 21%.
No clusters, no regimes, no interactions.

**Decision: stop looking for a better model and make a weak signal useful instead.**
I tested six model families — linear probability, logistic regression, gradient boosting as a
classifier, gradient boosting as a regressor, a linear hours forecast converted to a
probability, and a simple heuristic. Their 95% confidence intervals all overlapped completely.
With 55 breaches in the evaluation set the margin of error on the ranking measure is about
±0.06, and the spread between best and worst was 0.09. Adding a single extra week of
evaluation data reordered which one "won". Gradient boosting was consistently *worst*.

**Honest caveat:** the correct claim is "no model is *detectably* better on this data", not
"no better model exists". A genuine small improvement would be invisible at this sample size.

---

## 5. The model

**Decision: logistic regression.**

Not because it is more accurate — nothing is — but because when everything is tied on
accuracy the tiebreakers should be: it produces a valid probability, it can be explained in a
sentence, and it cannot overfit 396 positive examples the way a tree ensemble can.

**What logistic regression is.** It draws a weighted line through the input numbers and
squashes the result onto a 0-to-1 scale using the logistic curve. Each feature gets one
coefficient saying how much it pushes the answer up or down. That is the whole model.

### 5a. Training on every day of the week, not just Wednesdays

**Decision: one training row per (employee, week, day-of-week-known).**

The data happens to stop on a Wednesday. Building a model that only understands "three days
elapsed" would break the moment an export arrives on a Tuesday. So `days_elapsed` is a
*feature*, and each historical week is turned into six training rows — as if we had stopped
on Monday, on Tuesday, and so on.

This does two things at once: it makes the model indifferent to which day the export ends,
and it turns ~1,900 employee-weeks into **12,744 rows with 396 positives** instead of 1,704
with 59. The positives are not independent — the same week reappears at six cut points — so
the effective sample is still small, but the model learns how the decision boundary should
*shift* as the week progresses, which is real information.

It also produced the most useful single output in the project:

| we can see | ranking quality (AUC) |
|---|---|
| Monday | 0.697 |
| Mon–Tue | 0.737 |
| **Mon–Wed** | **0.800** |
| Mon–Thu | 0.836 |
| Mon–Fri | 0.864 |
| Mon–Sat | 0.930 |

The same model gets steadily better as the week goes on — and steadily less useful, because
the hours have already been worked. **Wednesday is the trade.** That reframes "0.80 is not
great" into "0.80 is what four days of warning costs".

### 5b. The features, and one thing to be honest about

Fifteen inputs: hours so far, days worked, days elapsed, days left, hours remaining before 55,
hours per elapsed day, the employee's prior average week / average days / average hours per day
/ heaviest week, hours the remaining days usually bring them, projected total, share of a usual
week already done, hours ahead of where they normally are by this day, and whether they work
nights.

**Every historical feature is shifted so no row can see itself or anything later.** I verified
this explicitly, because leakage is the most common way to fool yourself.

**The coefficients are not interpretable, and the model evaluation says so.** Several features
are exact functions of each other by construction: `hours_to_breach` is literally
`55 − hours_so_far`; `days_left` is `7 − days_elapsed`. I added a variance-inflation column to
the coefficient table which returns infinity for eight of them, confirming it. Prediction is
unaffected — regularisation keeps the fit stable and ranking does not require identifiable
coefficients — but credit is split arbitrarily between redundant inputs, so no single
coefficient is a statement about what causes overtime. I left them in because they help
prediction and removed the temptation to tell a tidy story instead.

### 5c. Walk-forward testing

**Decision: never train on a week and then test on it, or on anything before it.**

For each week evaluated, the model is retrained from scratch on only the weeks that came
before. That is the only honest simulation of Wednesday, when the future genuinely is unknown.
Seven weeks were evaluated this way: 1,491 employee-weeks, 55 breaches.

### 5d. Platt scaling — what it is and why it is there

**The problem.** A model can rank people correctly while being wrong about the *level*. It
might rank the riskiest person first and still say "0.30" when people like that breach 15% of
the time. Ranking is one property, calibration is another, and `risk_score` in the deliverable
is asked to be a probability — so the level has to be right.

Our model is trained across all six cut points at once. That is deliberate, but it means for
any *one* cut point its raw output is systematically off.

**What Platt scaling does.** It fits a second, tiny model — a one-input logistic regression —
whose only job is to map the raw score onto an honest probability. If the model says 0.30 and
people scoring 0.30 breach 22% of the time, it learns to report 0.22. It is a one-parameter
stretch of the scale.

Crucially it is **monotonic**: it can move every score up or down but can never reorder anyone.
Ranking is preserved exactly; only the level changes.

**Fitted on out-of-fold predictions only** — the walk-forward scores, where the model had not
seen the answers. Calibrating on scores the model was trained on would just relearn its
overconfidence.

**Why Platt and not isotonic.** Isotonic regression is the usual alternative and is more
flexible — it fits any increasing staircase rather than one curve. I tried it first. With only
55 positive examples it collapsed into a handful of flat steps, leaving **twelve people tied on
exactly 0.22** and therefore unrankable, which is useless for a call list. Platt's single
parameter cannot overfit that way and keeps every score distinct. This is a sample-size
decision, not a preference.

**Result:** mean predicted risk 0.0404 against an actual rate of 0.0369 — calibrated to within
about 10%. Brier score 0.0341 against 0.0369 for predicting zero for everyone.

### 5e. The decision boundary

**Decision: `will_breach = 1` when the calibrated risk is at or above 0.083, chosen by
maximising backtested F1.**

**Why so far below 0.5.** A 0.5 threshold assumes a false alarm and a missed breach cost the
same. They do not — a missed breach is a compliance breach plus 1.5× pay, a false alarm is a
phone call — and with a 3.4% base rate almost nobody ever reaches 0.5 anyway. A 0.5 cut would
flag nobody at all.

**Why F1 and not something else.** F1 balances precision (of those flagged, how many breach)
against recall (of those who breach, how many were flagged). I do not know how the submission
is graded, and F1 is the standard balanced choice. I swept every possible threshold across the
seven held-out weeks and took the best:

| rule | flagged | caught | missed | precision | recall | F1 |
|---|---|---|---|---|---|---|
| **model at 0.083** | 183 of 1,491 | 26 | 29 | **14.2%** | 47.3% | **0.218** |
| naive: current pace × days left | 419 of 1,491 | 33 | 22 | 7.9% | 60.0% | 0.139 |
| flag nobody | 0 | 0 | 55 | 0 | 0 | 0 |

**The naive baseline flags 28% of the workforce.** It catches more breaches, but only by
naming a quarter of everyone, which no contract manager can act on. The model roughly doubles
precision for a modest recall cost. This is exactly the −0.61 compensation effect: extrapolating
the current pace assumes the back half of the week looks like the front, and it slopes the
other way.

**Decision: the CSV and the dashboard use different cutoffs.** `predictions.csv` uses the
F1-optimal threshold because that is the graded artefact. The dashboard bands by *projected
hours* instead, because "projected past 55" is a sentence a manager can act on and "risk score
above 0.083" is not.

### 5f. Where the model fails, measured

| outcome | count | hours by Wed | prior average week | hours Thu–Sun | week total |
|---|---|---|---|---|---|
| caught | 26 | 28.6 | 47.4 | 29.1 | 57.6 |
| **missed** | 29 | **21.7** | 45.6 | **35.6** | 57.4 |
| false alarm | 157 | **29.3** | 45.6 | **16.0** | 45.2 |
| correctly quiet | 1,279 | 16.6 | 41.7 | 24.2 | 40.8 |

Read the middle two rows together. **The misses looked calmer on Wednesday than the false
alarms did** — 21.7 hours against 29.3 — and then worked more than twice as many weekend hours.
Near-identical visible evidence, opposite outcomes.

That is not a tuning failure. Whatever drives those weekends — a sudden absence, an event, a
machine breaking on Friday — has not happened when the call is made. It caps Wednesday
prediction regardless of which model is used, and it is the honest answer to "could you do
better with something else": not without different data.

---

## 6. The supervisors' notes

**Decision: no language model. A keyword lexicon instead.**

This sounds like the wrong call until you look at the notes. There are 2,117 of them and only
**846 distinct strings** — the top 60 cover half of everything written. They are roughly twenty
underlying templates, permuted by:

- surname substitution — *"stood in for Fourie"*, *"stood in for Sibiya"*
- five ways of writing 6am — `06:00`, `6am`, `six`, `0600`, `06h00`
- abbreviations — `hrs`, `agn`, `mgr`, `bc`, `pls`
- typos — `conntrol`, `clint`, `cnetre`, `okd'`
- and code-switching into Afrikaans (*"masjien is stukkend"*) and isiZulu (*"akafikanga"*)

**The vocabulary is small and closed.** A lexicon handles `stukkend` and `akafikanga` in one
line each, gives the same answer every time, costs nothing, runs instantly, needs no API key,
and can be audited by a person. A language model would be slower, non-deterministic and
unauditable, and would buy nothing.

Six categories: relief not arriving, covering an absent colleague, late handover, equipment
failure, client-requested work, and nothing useful. The first four are **preventable**; the
fifth is **billable**.

**One deliberate ordering decision.** Some notes read *"client signed for the extra hrs but real
reason is relief no show again"*. Every client keyword fires, while the supervisor has gone out
of their way to record the actual cause. A first-match rule would file this as billable and the
failure would vanish from the numbers. Operational causes are therefore checked **before**
client-requested ones.

**What the notes are for, and what they are not.** I tested them as a *predictive* feature —
adding per-employee and per-site failure-note counts to the model. R² moved from 0.5081 to
0.5104. Nothing. So they answer "why did this happen", not "who is next", and the dashboard
uses them only for cause.

They are excellent at that. Shifts with an operational-failure note average 11.45h against
9.35h for un-noted shifts, and account for about **2.7× the excess hours** that client-requested
work does.

**A caveat shown on the page:** supervisors write when something happens — only ~6% of ordinary
8–9 hour shifts carry a note, against more than half of 11–12 hour shifts. So these figures
describe hours that ran long, not all hours worked.

**This is still the interim version.** The full taxonomy with a hand-labelled sample and a
proper accuracy check is the next piece of work.

---

## 7. Actionability

### 7a. Projecting the rest of the week

**Decision: infer a likely roster from each person's own history, and say so.**

There is no forward roster anywhere in the data — the client sends shifts that *were worked*,
never shifts that are *scheduled*. So for each employee and each remaining weekday: how often
have they worked that day, and how long for? Someone who worked 8 of the last 9 Saturdays at
about 9.5 hours contributes roughly 8.4 expected hours.

This is the single biggest limitation in the project, and the dashboard states it rather than
hiding it. **If the client sent the roster, the largest source of error would disappear** —
days worked explains 79% of remaining hours, and it would go from predicted to known. That is
a far bigger improvement than any modelling change and it is a data request, not an
engineering one.

### 7b. The money

**Decision: overtime hours × that person's hourly rate × 1.5, counting only people projected
past the cap.**

Sunday and public-holiday hours are legally paid at 2×, but applying that would mean guessing
which *future* hours land on a Sunday — using a roster we do not have. So 1.5× is used
throughout and **every rand figure is a floor, not an estimate.** The page says this.

**Two numbers, kept separate,** because conflating them was a real bug I had to fix. **R20,800**
is the overtime pay for the 25 people projected past 55 hours — the money attached to the
compliance problem. **R40,809** is total projected overtime across all 213 people. Most of that
second figure is lawful overtime the client expects to pay. Showing it as "at risk" overstated
the problem by roughly double.

### 7c. Answer Set Programming for the reassignment

**Decision: solve the whole reassignment at once, declaratively, rather than person by person.**

**Why it is not a simple matching problem.** Moving a shift off someone over the cap means
giving it to somebody who is at the same site, in the same role, on the same day/night pattern,
not already working that day, and with enough room below 45 hours that they do not become the
next problem. That last constraint is a knapsack per receiver, which is what defeats a greedy
approach — fix the first name and you can strand someone further down who had fewer options.
I verified this is real: total overshoot to shed is 114.5 hours against 1,086 hours of
capacity, but the capacity is locked in site/role/pattern pockets, and four people including
the highest-risk person in the business have **zero** valid candidates.

**What Answer Set Programming is.** Ordinary programming is a recipe: do this, then this.
ASP is the opposite — **you describe what a correct answer looks like and never say how to find
one.** You state the rules a valid plan must obey and the solver searches for something
satisfying all of them simultaneously.

The sudoku analogy is exact. Nobody writes instructions for solving a sudoku; they state the
rules — one of each digit per row, column and box — and any grid obeying all of them is a
solution.

**The mechanics, in three stages.** *Facts* are what we know: who is over and by how much, which
shifts might move, who could cover and how much room they have. *Rules* say what may and may not
be true. A rule beginning `:-` is a constraint meaning "it must never be that…", and it prunes
away every candidate answer where that situation arises. *Optimisation* statements then rank the
survivors. The solver grounds the program — expanding the rules against the facts into a large
Boolean formula, here about 363 possible moves — and searches it exhaustively using the same
family of techniques that power modern SAT solvers, so it prunes vast regions rather than
enumerating them.

Our program, in full:

```prolog
% at most one legal receiver per movable shift
{ move(S,R) : cover(R,_,Site,Role,Pattern), not busy(R,Day) } 1
    :- shift(S,_,Day,_,Site,Role,Pattern).

% nobody may be given more hours than they have room for below 45
:- cover(R,Room,_,_,_), #sum{ H,S : move(S,R), shift(S,_,_,H,_,_,_) } > Room.

% nobody may be given two shifts on the same day
:- move(S1,R), move(S2,R), S1 < S2, shift(S1,_,D,_,_,_,_), shift(S2,_,D,_,_,_,_).

% somebody is resolved once enough hours have come off them
shed(E,H)   :- at_risk(E,_), H = #sum{ Hs,S : move(S,_), shift(S,E,_,Hs,_,_,_) }.
resolved(E) :- at_risk(E,Need), shed(E,H), H >= Need.

#maximize { N@2,E : resolved(E), at_risk(E,N) }.   % cover the most, worst first
#minimize { 1@1,S,R : move(S,R) }.                 % then disturb the fewest shifts
```

**Why this is worth the dependency.** The rules read almost as plainly as the regulation they
encode. An operations manager could read the double-booking line and tell you whether it matches
how the business actually works — which is not true of equivalent scheduling code written as a
recipe. And when the solver says six people cannot be helped, that is not an algorithm giving
up early: no arrangement of the available staff could have covered them.

**Decisions inside the solver.**

- **Hours scaled by 4.** ASP works in integers and all shift lengths are quarter-hours, so ×4
  is exact with no rounding.
- **Two priority levels, not three.** I originally had a third tie-breaker on hours moved.
  It roughly tripled the time to a good plan and never changed which people ended up covered.
- **Do not prove optimality; take the best plan within a time budget.** The solver emits
  successively better plans as it searches. In practice it finishes in about 1.4 seconds here,
  but the budget means the button always returns.
- **A bug worth recording.** My first attempt read the results *after* cancelling the search
  handle, which silently yields nothing — so it fell back to greedy every time while appearing
  to work. Improving plans must be captured in a callback as they arrive.
- **A greedy fallback exists** for environments without the solver, and is honest in the code
  about being worse.

**Result: 18 of 24 people brought back under the cap using 18 shift moves, ~R15,292 of overtime
avoided.** I verified the plan independently: **zero constraint violations, nobody pushed past
45 hours** (the maximum receiver lands on exactly 45.0, so the capacity constraint is binding).
Against greedy it covers the same people while moving about 40 fewer hours.

**A bug this exposed.** 42 people were flagged by the model but only 24 are actually projected
past 55 hours. The rest score above the risk threshold while projecting under the cap — flagged
on *pattern*, not on hours. I had been feeding all 42 to the solver with a 0.25h floor, which
invented work and then reported it as failure. They now show as "monitor", not as a failed
action.

### 7d. Why the solver sits behind a button

**Decision: assess on page load, solve on request.**

Scoring the week takes a few seconds and is cached. Producing a plan is a decision, not a page
load — so it runs when asked. Because the assessment is already cached, the button returns in
about 0.1 seconds.

---

## 8. The dashboard

**Decision: FastAPI with a plain HTML page, no framework.** The brief says ugly is fine and
explicitly not to spend time on polish.

**Decision: never say "will breach".** Nobody has breached, nobody ever has by a Wednesday, and
the model is right about roughly one in seven of the people it flags. The page says *at risk*
and states the hit rate in the lead paragraph.

**Decision: three bands rather than a yes/no flag.** *Over the cap* (projected past 55),
*close to the line* (within 5 hours — one unplanned shift away), *worth watching* (projecting
under, but the week resembles weeks that ended badly). A binary flag throws away the fact that
someone at 54.8 hours is in a different position from someone at 40.

**Decision: the headline count includes only people definitely projected over.** Maybes are
counted beside it, never inside it.

**Decision: explain every number on the page itself.** Each section has an expander covering
what its columns mean and how they are calculated, in plain language, including what the
percentages are percentages *of*. A number a manager cannot interpret is worse than no number.

**Decision: show the model's own accuracy and the data's own defects.** The 126 impossible
overlapping shifts and 184 missing clock-outs are on the page with an explanation of why we
count them the client's way rather than silently correcting them. A dashboard that hides its
own error bars invites more trust than it has earned.

---

## 9. Things I would want challenged

- **The projected roster is inferred, not known.** Biggest limitation in the project; fixed by
  asking the client for the roster, not by better modelling.
- **The 126 overlapping shifts are counted.** More correct to remove them; would disagree with
  the ground truth we are graded against. Flagged instead.
- **184 missing clock-outs count as zero hours.** Matches the client's engine, so real risk is
  slightly understated.
- **The F1 threshold is a guess about grading.** If precision matters more than recall, the
  cutoff should move and the flagged count would drop sharply.
- **Interim note categories.** The keyword lexicon is validated by eye against the templates,
  not yet against a hand-labelled sample with measured accuracy.
- **Seven evaluation weeks, 55 breaches.** Everything measured here carries a wide error bar,
  and I have tried to quote it rather than round it away.
