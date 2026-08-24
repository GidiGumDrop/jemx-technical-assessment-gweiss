# Applied AI Intern — Technical Assessment

**Jem / JemX · Cape Town**

Please do not spend more than a couple of hours on this. You have **48 hours** from when this lands in your inbox.

We would rather see what you can do in an afternoon and hear an honest account of what you would do next than receive
something that takes you a whle day. Running out of time is expected. Hiding it is not ;-)

This is part of the whole assessment. What you build here is what we sit down and work through together afterwards, so build something you can explain and demo live.

---

## The situation

Jem builds WhatsApp-native HR and payroll for South Africa's frontline workforce - the people who clean, guard, cook, drive and serve. Our clients are large employers
with thousands of shift workers spread across sites.

One of them is a national facilities-management company. Every week they send us raw clock-in and clock-out data from six sites. They have a problem they cannot see
coming: **overtime**.

Under the Basic Conditions of Employment Act, an employee may work a maximum of **45 ordinary hours** and a maximum of **10 hours of overtime** in a week. Going past
that is a compliance breach. It is also expensive — overtime is paid at 1.5x, and Sunday and public-holiday work at 2x.

Today the client finds out on Monday, when payroll runs and the overtime cost is already accrued. By then nothing can be done about it.

**In the scenario it is Wednesday, and the week is not over** - that holds whatever day you actually sit down to do this. The data runs to the end of that Wednesday.
Weeks run Monday to Sunday. The week that is still in progress when the data stops is the week we care about.

> **Who is going to breach the 10-hour overtime cap by Sunday, and what should somebody do about it today?**

Every question below is in service of that one.

---

## The data

In `data/`. All of it is synthetic — no real client or employee data is used
anywhere in this exercise.

| File | What it is |
|---|---|
| `shifts.csv` | One row per clock-in / clock-out record, roughly ten weeks of it. |
| `employees.csv` | The employee register. |
| `sites.csv` | The six sites. |
| `shift_notes.csv` | Free-text notes typed by supervisors against some shifts. |
| `public_holidays.csv` | Public holidays falling in the period. |
| `weekly_summary.csv` | A weekly roll-up exported by the client's existing system. |
| `payroll_details.csv` | Came across with the extract from the client's payroll system. |

It is real-world data in the sense that matters: it was collected by people at 3am on a phone with cracked glass, not assembled for a tutorial. Treat it accordingly.

If you make an assumption about the data, write it down. A stated assumption is never wrong; a silent one usually is.

---

## What to build

**An ops room dashboard, deployed, on a URL we can open.**

The person using it is the contract manager who owns the six sites. They are not an analyst. They are in a car between sites or standing in a control room, on a phone, and they have about ten minutes before their next call.

**How you build it is entirely your call.** Any language, any framework, any host. We are not prescribing features or architecture, and there is no answer we are waiting for you to arrive at. These four requirements are the whole specification:

**1. It shows who is going over by Sunday.** This is the answer to the question above. It has to be correct - we check it against ground truth you do not have.

**2. It says what to do about it.** Something the contract manager could act on without opening a spreadsheet. A suggestion that could have been written without
looking at the data is worth nothing to them.

**3. It says why the hours happened.** This comes out of the supervisors' notes - see the next section.

**4. It accepts next week's data without you.** On Monday the client sends a new export: the same files, the same columns, a later week, different rows. Loading it must not require a developer. We should be able to hand you a fresh export and ask you to load it.

**Rough is fine. Ugly is fine.** Do not spend your hours making it pretty; everyone can make something pretty with an agent now, and that is not what we are reading.


Free tiers for everything is fine - **Please do not spend money on this.**

---

## The supervisors' notes — required

`shift_notes.csv` holds what supervisors typed against shifts. It is messy: typos, abbreviations, isiZulu and Afrikaans mixed in, plenty of rows that say nothing at
all. Nobody cleaned it up for you. That is the point.

Three things to do with it:

**1. Sort the notes into a small set of reasons** for the extra hours. You decide what the categories are. It is fine if plenty end up as "nothing useful here".

**2. Split the overtime into two piles:** hours the client *asked for* (they pay for those, that is fine) versus hours caused by operational failures — no-shows, broken
machines, late handovers (that is cost somebody could fix). Work out what that split says and where it is concentrated.

**3. Check whether your sorting is any good.** There is no answer sheet, so you have to design your own check. Hand-label a sample and measure against it, run two methods
and compare, whatever you think is right. We want to know what your check found - including where your sorting turned out to be wrong.

Whether you use an LLM, something simpler, or both is entirely your call - and a well-argued "a language model is the wrong tool here, and here is what I did instead" scores just as well as a clever use of one. What loses marks is a sorting with no evidence about whether it works. 

---

## What to send us

**Send us three links:** your dashboard, your repo, and your video.

### 1. The dashboard
A working URL we can open.

### 2. The repo
Public, or invite `southafricanrob`. It must contain these four files.

**`predictions.csv`** — who breaches in the week that was still in progress when the
data stopped. One row for every `employee_id` in `employees.csv`, exactly these
columns:

```
employee_id,will_breach,risk_score
E1001,0,0.03
E1002,1,0.81
```

`will_breach` is `1` if you think they will exceed 10 hours of overtime by Sunday, otherwise `0`. `risk_score` is a number between 0 and 1.

**`note_classifications.csv`** — your sorting of the supervisors' notes, one row per note in `shift_notes.csv`, exactly these columns:

```
shift_id,category,note
S100737,relief_no_show,Next shift guard did not pitch. Had to cover.
```

`category` is whatever your own taxonomy calls it. We are going to read a sample of these, so send the real output rather than a summary.

**`NOTES.md`** - short. Half a page is plenty, and it is not a report. Three things:

- Any assumptions you made about the data.
- How you checked your note-sorting, and what that check found.
- What a trained model **would learn here that your approach does not** - or, if you trained one, what yours actually learned - and how you would test it on a couple of hundred people without fooling yourself. A few honest sentences beat a page of theory. This is where we find out whether you understand what is happening under the hood, so do not skip it.


### 3. The video
Loom, or Drive link. Camera on, screen shared. **five minutes**

There is no written report in this assessment, so the video is carrying that weight. Not a polished pitch - we want to hear you think. Show us the thing working, and cover four things:

- What you found in the data that you did not expect.
- What you compared your answer against. What is the naive baseline, and do you beat it? Which metric did you optimise for, and **why that one and not another**?
- Where you would not trust your own output.
- What you would do next with another two days.

And somewhere in the five minutes please do this: pick **one number your system produced** - one person's risk score, one note your system sorted — and explain where it came from, step by step, simply enough that a site manager with no data background would follow it. Someone who truly understands their own work can explain it very simply.

---

## Then we sit down together

If we like what you send, we will spend **45–60 minutes** working through it with you, usually within a week of your submission, in person where that is possible.

We are not going to grill you on trivia. We will open what you built and let you talk us through it

What we are watching is how you work, the tools change every few months - how you think about a problem does not.

---

## Ground rules

- **Use coding agents.** Claude Code, Cursor, Codex, whatever you use daily. This is  how we work. Be ready to explain what you submit.
- Any language, any libraries, any host. It must run on a normal laptop, and must not require a paid API key.
- Free tiers only. Do not spend money on this.
- Ask us anything. Reply to the email and we will answer quickly - asking is not a mark against you.

## If you run out of time

Expected. Cut in this order, and say in the video what you cut:

1. Polish on the dashboard. We mean it.
2. **Requirement 2** - what to do about it. A correct list of names beats a pretty
   list of actions.
3. Depth on the trained-model question in `NOTES.md` - but write *something*.

Do not cut: the answer itself, the check on your note-sorting, or the ability to load a new week of data. Those three are what we are actually reading.

---

## How we will assess it

No hidden criteria. We score four things:

| | |
|---|---|
| **Passion and self-direction** | Did you chase something past the point where the brief made you? |
| **Fundamentals** | Judgement about method, data and architecture — not which library you reached for. |
| **Hard problems** | Did you find what was actually going on in the data, and reason from first principles when it got ambiguous? |
| **Communication** | Can you explain a technical decision so a non-technical person follows it first time? |


The most common way to lose is to build something impressive that answers a question nobody asked. The most common way to win is to be right about a small thing and say
so clearly.

Good luck!